import CoreGraphics
import CoreML
import Foundation

struct CoreMLPromptPreset: Decodable, Identifiable {
    let key: String
    let title: String
    let prompt: String
    let aliases: [String]

    var id: String { key }
}

enum CoreMLDecoderMode: String, CaseIterable, Identifiable {
    case compressed4Bit
    case fp16

    var id: String { rawValue }

    var title: String {
        switch self {
        case .compressed4Bit:
            return "VAE 4-bit"
        case .fp16:
            return "VAE FP16"
        }
    }
}

private struct PromptPresetFile: Decodable {
    let embeddingShape: [Int]
    let uncondEmbeddingShape: [Int]
    let presets: [CoreMLPromptPreset]
}

private struct DPMSolverSchedulerFile: Decodable {
    let timesteps: [Int]
    let sigmas: [Float]
    let scheduler: String
    let algorithmType: String
    let solverOrder: Int
    let solverType: String
    let finalSigmasType: String
    let guidanceScale: Float
}

final class LCMCoreMLImageGenerator: @unchecked Sendable {
    static let imageSize = 128
    static let latentSide = 16
    static let latentCount = 4 * latentSide * latentSide
    static let debugLogging = true

    let presets: [CoreMLPromptPreset]
    let stepCount: Int
    let decoderModes: [CoreMLDecoderMode]

    private let unetURL: URL
    private let compressedVAEURL: URL
    private let fp16VAEURL: URL?
    private let embeddingData: Data
    private let uncondEmbeddingData: Data
    private let scheduler: DPMSolverSchedulerFile
    private let embeddingStrideBytes: Int

    init(bundle: Bundle = .main) throws {
        unetURL = try Self.resourceURL("unet_sd_16x16_6bit", "mlmodelc", bundle: bundle)
        compressedVAEURL = try Self.resourceURL("vae_decoder_128x128_noattn_4bit", "mlmodelc", bundle: bundle)
        fp16VAEURL = bundle.url(forResource: "vae_decoder_128x128_noattn", withExtension: "mlmodelc")

        let presetFile = try JSONDecoder().decode(
            PromptPresetFile.self,
            from: Data(contentsOf: Self.resourceURL("prompt_presets", "json", bundle: bundle))
        )
        scheduler = try JSONDecoder().decode(
            DPMSolverSchedulerFile.self,
            from: Data(contentsOf: Self.resourceURL("sd_ddim_scheduler", "json", bundle: bundle))
        )
        presets = presetFile.presets
        stepCount = scheduler.timesteps.count
        decoderModes = fp16VAEURL == nil ? [.compressed4Bit] : [.compressed4Bit, .fp16]
        embeddingData = try Data(contentsOf: Self.resourceURL("prompt_embeddings_f16", "bin", bundle: bundle))
        uncondEmbeddingData = try Data(contentsOf: Self.resourceURL("uncond_embedding_f16", "bin", bundle: bundle))

        let tokenCount = presetFile.embeddingShape.dropFirst().reduce(1, *)
        embeddingStrideBytes = tokenCount * MemoryLayout<Float16>.stride

        Self.debugLog(
            "ready presets=\(presets.count) scheduler=\(scheduler.scheduler) steps=\(stepCount) " +
            "guidance=\(scheduler.guidanceScale) solver=\(scheduler.algorithmType)/\(scheduler.solverType) " +
            "embeddingBytes=\(embeddingData.count) uncondBytes=\(uncondEmbeddingData.count) " +
            "decoders=\(decoderModes.map(\.title).joined(separator: ","))"
        )
    }

    func bestPresetIndex(for prompt: String) -> Int {
        let normalized = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return 0 }
        for (index, preset) in presets.enumerated() {
            if preset.key == normalized { return index }
            if preset.aliases.contains(where: { normalized.contains($0.lowercased()) }) {
                return index
            }
        }
        return Int(Self.fnv1a(normalized) % UInt64(max(presets.count, 1)))
    }

    func generate(
        prompt: String,
        seed: UInt64,
        guidanceScale: Float? = nil,
        decoderMode: CoreMLDecoderMode = .compressed4Bit
    ) throws -> TinyGeneratedImage {
        let startedAt = CFAbsoluteTimeGetCurrent()
        let presetIndex = bestPresetIndex(for: prompt)
        let preset = presets[presetIndex]
        let guidanceScale = guidanceScale ?? scheduler.guidanceScale
        Self.debugLog(
            "generate start prompt=\"\(prompt)\" seed=\(seed) preset=\(preset.key) " +
            "title=\"\(preset.title)\" guidance=\(guidanceScale) decoder=\(decoderMode.title)"
        )
        var rng = SeededRandom(seed: seed ^ Self.fnv1a(presets[presetIndex].key))
        let condEmbedding = try makePromptEmbedding(index: presetIndex)
        let uncondEmbedding = try makeUncondEmbedding()
        var latents = makeLatents(rng: &rng)
        Self.debugLog("initialLatents \(Self.stats(latents).summary)")
        latents = try runUNet(
            latents: latents,
            condEmbedding: condEmbedding,
            uncondEmbedding: uncondEmbedding,
            guidanceScale: guidanceScale
        )
        let image = try decode(latents: latents, decoderMode: decoderMode)
        Self.debugLog("generate done elapsedMs=\(Self.elapsedMS(since: startedAt))")
        return image
    }

    private func runUNet(
        latents initialLatents: [Float],
        condEmbedding: MLMultiArray,
        uncondEmbedding: MLMultiArray,
        guidanceScale: Float
    ) throws -> [Float] {
        try autoreleasepool {
            let loadStartedAt = CFAbsoluteTimeGetCurrent()
            let unet = try MLModel(contentsOf: unetURL, configuration: Self.memoryConstrainedConfiguration())
            Self.debugLog("unet loaded elapsedMs=\(Self.elapsedMS(since: loadStartedAt))")
            var latents = initialLatents
            var previousModelOutput: [Float]?
            let options = MLPredictionOptions()

            for (index, timestep) in scheduler.timesteps.enumerated() {
                try autoreleasepool {
                    let stepStartedAt = CFAbsoluteTimeGetCurrent()
                    let uncondNoise = try predictNoiseValues(
                        unet: unet,
                        latents: latents,
                        timestep: timestep,
                        embedding: uncondEmbedding,
                        options: options
                    )
                    let condNoise = try predictNoiseValues(
                        unet: unet,
                        latents: latents,
                        timestep: timestep,
                        embedding: condEmbedding,
                        options: options
                    )
                    let noiseStats = updateLatents(
                        &latents,
                        previousModelOutput: &previousModelOutput,
                        uncondNoise: uncondNoise,
                        condNoise: condNoise,
                        guidanceScale: guidanceScale,
                        stepIndex: index
                    )
                    if Self.shouldLogStep(index: index, total: scheduler.timesteps.count) {
                        Self.debugLog(
                            "step \(index + 1)/\(scheduler.timesteps.count) t=\(timestep) " +
                            "noise \(noiseStats.summary) latents \(Self.stats(latents).summary) " +
                            "elapsedMs=\(Self.elapsedMS(since: stepStartedAt))"
                        )
                    }
                }
            }

            return latents
        }
    }

    private func predictNoiseValues(
        unet: MLModel,
        latents: [Float],
        timestep: Int,
        embedding: MLMultiArray,
        options: MLPredictionOptions
    ) throws -> [Float] {
        let sample = try multiArray(shape: [1, 4, NSNumber(value: Self.latentSide), NSNumber(value: Self.latentSide)])
        copy(latents, to: sample)
        let timestepArray = try multiArray(shape: [1])
        timestepArray.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(Float(timestep))
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "sample": MLFeatureValue(multiArray: sample),
            "timestep": MLFeatureValue(multiArray: timestepArray),
            "encoder_hidden_states": MLFeatureValue(multiArray: embedding),
        ])
        let output = try unet.prediction(from: provider, options: options)
            .featureValue(for: "noise_pred")!
            .multiArrayValue!
        if timestep == scheduler.timesteps.first {
            Self.debugLog("noise_pred \(Self.multiArrayDescription(output))")
        }
        return values(from: output, count: Self.latentCount)
    }

    private func decode(latents: [Float], decoderMode: CoreMLDecoderMode) throws -> TinyGeneratedImage {
        try autoreleasepool {
            Self.debugLog("finalLatents \(Self.stats(latents).summary)")
            let loadStartedAt = CFAbsoluteTimeGetCurrent()
            let vaeURL = url(for: decoderMode)
            let vae = try MLModel(contentsOf: vaeURL, configuration: Self.memoryConstrainedConfiguration())
            Self.debugLog("vae loaded mode=\(decoderMode.title) elapsedMs=\(Self.elapsedMS(since: loadStartedAt))")
            let options = MLPredictionOptions()
            let finalLatents = try multiArray(shape: [1, 4, NSNumber(value: Self.latentSide), NSNumber(value: Self.latentSide)])
            copy(latents, to: finalLatents)
            let decodedProvider = try MLDictionaryFeatureProvider(dictionary: [
                "latents": MLFeatureValue(multiArray: finalLatents)
            ])
            let decoded = try vae.prediction(from: decodedProvider, options: options)
                .featureValue(for: "decoded")!
                .multiArrayValue!
            Self.debugLog("decodedArray \(Self.multiArrayDescription(decoded))")
            return image(from: decoded)
        }
    }

    private static func memoryConstrainedConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        return configuration
    }

    private func makePromptEmbedding(index: Int) throws -> MLMultiArray {
        let output = try multiArray(shape: [1, 77, 768])
        let offset = index * embeddingStrideBytes
        embeddingData.withUnsafeBytes { bytes in
            output.dataPointer.copyMemory(from: bytes.baseAddress!.advanced(by: offset), byteCount: embeddingStrideBytes)
        }
        return output
    }

    private func makeUncondEmbedding() throws -> MLMultiArray {
        let output = try multiArray(shape: [1, 77, 768])
        uncondEmbeddingData.withUnsafeBytes { bytes in
            output.dataPointer.copyMemory(from: bytes.baseAddress!, byteCount: uncondEmbeddingData.count)
        }
        return output
    }

    private func makeLatents(rng: inout SeededRandom) -> [Float] {
        (0..<Self.latentCount).map { _ in rng.normal() }
    }

    private func updateLatents(
        _ latents: inout [Float],
        previousModelOutput: inout [Float]?,
        uncondNoise: [Float],
        condNoise: [Float],
        guidanceScale: Float,
        stepIndex: Int
    ) -> TensorStats {
        var guidedNoise = [Float]()
        guidedNoise.reserveCapacity(latents.count)
        let sigmaSRaw = scheduler.sigmas[stepIndex]
        let sigmaTRaw = scheduler.sigmas[stepIndex + 1]
        let (alphaS, sigmaS) = Self.alphaSigma(from: sigmaSRaw)
        let (alphaT, sigmaT) = Self.alphaSigma(from: sigmaTRaw)
        let lambdaS = Self.lambda(alpha: alphaS, sigma: sigmaS)
        let lambdaT = Self.lambda(alpha: alphaT, sigma: sigmaT)
        let h = lambdaT - lambdaS
        let expNegH = Self.expNeg(h)
        var modelOutput = [Float]()
        modelOutput.reserveCapacity(latents.count)
        for i in latents.indices {
            let noise = uncondNoise[i] + guidanceScale * (condNoise[i] - uncondNoise[i])
            guidedNoise.append(noise)
            modelOutput.append((latents[i] - sigmaS * noise) / max(alphaS, 0.000001))
        }

        let useFirstOrder = previousModelOutput == nil || (stepIndex == scheduler.timesteps.count - 1 && scheduler.finalSigmasType == "zero")
        if useFirstOrder {
            for i in latents.indices {
                latents[i] = (sigmaT / sigmaS) * latents[i] - alphaT * (expNegH - 1.0) * modelOutput[i]
            }
        } else if let previous = previousModelOutput {
            let sigmaS1Raw = scheduler.sigmas[stepIndex - 1]
            let (alphaS1, sigmaS1) = Self.alphaSigma(from: sigmaS1Raw)
            let lambdaS1 = Self.lambda(alpha: alphaS1, sigma: sigmaS1)
            let h0 = lambdaS - lambdaS1
            let r0 = h0 / h
            for i in latents.indices {
                let d1 = (modelOutput[i] - previous[i]) / r0
                let coeff = alphaT * (expNegH - 1.0)
                latents[i] = (sigmaT / sigmaS) * latents[i] - coeff * modelOutput[i] - 0.5 * coeff * d1
            }
        }
        previousModelOutput = modelOutput
        return Self.stats(guidedNoise)
    }

    private func image(from decoded: MLMultiArray) -> TinyGeneratedImage {
        let values = values(from: decoded, count: Self.imageSize * Self.imageSize * 3)
        Self.debugLog("decoded \(Self.stats(values).summary)")
        var rgba = [UInt8](repeating: 255, count: Self.imageSize * Self.imageSize * 4)
        let plane = Self.imageSize * Self.imageSize
        var clipped = 0
        for y in 0..<Self.imageSize {
            for x in 0..<Self.imageSize {
                let src = y * Self.imageSize + x
                let dst = src * 4
                let red = values[src]
                let green = values[plane + src]
                let blue = values[plane * 2 + src]
                clipped += Self.isOutsideImageRange(red) ? 1 : 0
                clipped += Self.isOutsideImageRange(green) ? 1 : 0
                clipped += Self.isOutsideImageRange(blue) ? 1 : 0
                rgba[dst] = toByte(red)
                rgba[dst + 1] = toByte(green)
                rgba[dst + 2] = toByte(blue)
            }
        }
        let totalChannels = Self.imageSize * Self.imageSize * 3
        Self.debugLog("image clippedChannels=\(clipped)/\(totalChannels)")
        return TinyGeneratedImage(width: Self.imageSize, height: Self.imageSize, rgba: rgba)
    }

    private func toByte(_ value: Float) -> UInt8 {
        UInt8(max(0, min(255, Int(((value + 1.0) * 127.5).rounded()))))
    }

    private func multiArray(shape: [NSNumber]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: .float16)
    }

    private func copy(_ values: [Float], to array: MLMultiArray) {
        let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in values.indices {
            pointer[i] = Float16(values[i])
        }
    }

    private func values(from array: MLMultiArray, count: Int) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var output = [Float]()
        output.reserveCapacity(shape.reduce(1, *))

        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            Self.appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                Float(pointer[$0])
            }
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            Self.appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                pointer[$0]
            }
        case .double:
            let pointer = array.dataPointer.assumingMemoryBound(to: Double.self)
            Self.appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                Float(pointer[$0])
            }
        default:
            Self.debugLog("unexpected MLMultiArray dtype=\(array.dataType.rawValue), treating as Float16")
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            Self.appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                Float(pointer[$0])
            }
        }

        if output.count != count {
            Self.debugLog("logical value count mismatch expected=\(count) actual=\(output.count)")
        }
        return output
    }

    private func url(for decoderMode: CoreMLDecoderMode) -> URL {
        switch decoderMode {
        case .compressed4Bit:
            compressedVAEURL
        case .fp16:
            fp16VAEURL ?? compressedVAEURL
        }
    }

    private static func shouldLogStep(index: Int, total: Int) -> Bool {
        index == 0 || index == total - 1 || (index + 1).isMultiple(of: 5)
    }

    private static func isOutsideImageRange(_ value: Float) -> Bool {
        value < -1.0 || value > 1.0
    }

    private static func multiArrayDescription(_ array: MLMultiArray) -> String {
        "dtype=\(array.dataType.rawValue) shape=\(array.shape.map(\.intValue)) strides=\(array.strides.map(\.intValue))"
    }

    private static func appendLogicalValues(
        shape: [Int],
        strides: [Int],
        dimension: Int,
        offset: Int,
        into output: inout [Float],
        read: (Int) -> Float
    ) {
        if dimension == shape.count {
            output.append(read(offset))
            return
        }
        for index in 0..<shape[dimension] {
            appendLogicalValues(
                shape: shape,
                strides: strides,
                dimension: dimension + 1,
                offset: offset + index * strides[dimension],
                into: &output,
                read: read
            )
        }
    }

    private static func alphaSigma(from rawSigma: Float) -> (alpha: Float, sigma: Float) {
        let alpha = 1.0 / sqrt(rawSigma * rawSigma + 1.0)
        return (alpha, rawSigma * alpha)
    }

    private static func lambda(alpha: Float, sigma: Float) -> Float {
        if sigma == 0 {
            return .infinity
        }
        return log(alpha) - log(sigma)
    }

    private static func expNeg(_ value: Float) -> Float {
        value.isInfinite ? 0.0 : exp(-value)
    }

    private static func stats(_ values: [Float]) -> TensorStats {
        var minValue = Float.infinity
        var maxValue = -Float.infinity
        var sum: Float = 0
        var sumSquares: Float = 0
        var finiteCount = 0
        for value in values where value.isFinite {
            finiteCount += 1
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
            sum += value
            sumSquares += value * value
        }
        guard finiteCount > 0 else {
            return TensorStats(min: .nan, max: .nan, mean: .nan, rms: .nan, finiteCount: 0, totalCount: values.count)
        }
        let count = Float(finiteCount)
        return TensorStats(
            min: minValue,
            max: maxValue,
            mean: sum / count,
            rms: sqrt(sumSquares / count),
            finiteCount: finiteCount,
            totalCount: values.count
        )
    }

    private static func elapsedMS(since start: CFAbsoluteTime) -> Int {
        Int(((CFAbsoluteTimeGetCurrent() - start) * 1000).rounded())
    }

    private static func debugLog(_ message: String) {
        guard debugLogging else { return }
        print("[TinyImageCoreML] \(message)")
    }

    private static func resourceURL(_ name: String, _ ext: String, bundle: Bundle) throws -> URL {
        if let url = bundle.url(forResource: name, withExtension: ext) {
            return url
        }
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: "\(name).\(ext)"])
    }

    private static func fnv1a(_ text: String) -> UInt64 {
        var value: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            value ^= UInt64(byte)
            value &*= 0x100000001b3
        }
        return value
    }
}

private struct SeededRandom {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func normal() -> Float {
        let u1 = max(uniform(), 0.000001)
        let u2 = uniform()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }

    private mutating func uniform() -> Float {
        let value = nextUInt64() >> 40
        return Float(value) / Float(0x0100_0000)
    }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }
}

private struct TensorStats {
    let min: Float
    let max: Float
    let mean: Float
    let rms: Float
    let finiteCount: Int
    let totalCount: Int

    var summary: String {
        "min=\(Self.format(min)) max=\(Self.format(max)) mean=\(Self.format(mean)) rms=\(Self.format(rms)) finite=\(finiteCount)/\(totalCount)"
    }

    private static func format(_ value: Float) -> String {
        guard value.isFinite else { return "\(value)" }
        return String(format: "%.4f", value)
    }
}
