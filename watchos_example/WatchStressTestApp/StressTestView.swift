import CoreML
import Foundation
import os
import SwiftUI

struct StressLogLine: Identifiable {
    let id = UUID()
    let text: String
}

private struct TextEncoderProbeManifest: Decodable {
    let prompts: [String]
    let inputIdsShape: [Int]
    let inputIdsDtype: String
    let hiddenStatesShape: [Int]
    let hiddenStatesDtype: String
}

enum StressComputeMode: String, CaseIterable, Identifiable {
    case cpuOnly
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .cpuOnly:
            return "CPU"
        case .all:
            return "All"
        }
    }

    var computeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly:
            return .cpuOnly
        case .all:
            return .all
        }
    }
}

@MainActor
final class StressTestViewModel: ObservableObject {
    @Published var isRunning = false
    @Published var status = "Idle"
    @Published var retainedMB = 0
    @Published var lastStableMB = 0
    @Published var modelCount = 0
    @Published var retainedModelCount = 0
    @Published var logLines: [StressLogLine] = []
    @Published var computeMode: StressComputeMode = .cpuOnly

    private let logger = Logger(subsystem: "dev.local.WatchStressTestApp", category: "stress")
    private var retainedBuffers: [Data] = []
    private var retainedModels: [(label: String, model: MLModel)] = []
    private var didPrepareForLaunch = false
    private var textEncoderPromptIndex = 0
    private var cachedTextEncoderProbeManifest: TextEncoderProbeManifest?

    let isTextEncoderSmokeOnly = ProcessInfo.processInfo.environment["WATCH_TEXT_ENCODER_AUTORUN"] == "1"

    func prepareForLaunch() async {
        guard !didPrepareForLaunch else { return }
        didPrepareForLaunch = true
        scanModels()
        guard isTextEncoderSmokeOnly else { return }
        await runSeparatedTextEncoderCycle()
    }

    func scanModels() {
        let urls = bundledModelURLs()
        modelCount = urls.count
        if urls.isEmpty {
            log("scan: no bundled .mlmodelc found")
        } else {
            log("scan: found \(urls.count) model(s)")
            for url in urls {
                log("model: \(url.lastPathComponent)")
            }
        }
    }

    func releaseBuffers() {
        retainedBuffers.removeAll(keepingCapacity: false)
        retainedMB = 0
        lastStableMB = 0
        status = "Released"
        log("memory: released retained buffers")
    }

    func releaseModels() {
        retainedModels.removeAll(keepingCapacity: false)
        retainedModelCount = 0
        status = "Released"
        log("models: released retained models")
    }

    func addMemory(megabytes: Int) async {
        await runGuarded("memory +\(megabytes) MB") {
            try await allocateAndRetain(megabytes: megabytes)
        }
    }

    func runFineMemoryLadder() async {
        await runGuarded("memory fine ladder") {
            for _ in 0..<20 {
                try await allocateAndRetain(megabytes: 16)
                try await Task.sleep(nanoseconds: 450_000_000)
            }
        }
    }

    func runAggressiveMemoryLadder() async {
        await runGuarded("memory aggressive ladder") {
            for mb in [16, 32, 64, 96, 128, 192, 256] {
                try await allocateAndRetain(megabytes: mb)
                try await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    func loadModelsOnly() async {
        await runGuarded("load models") {
            let urls = bundledModelURLs()
            modelCount = urls.count
            guard !urls.isEmpty else {
                log("load: no bundled .mlmodelc found")
                return
            }

            for url in urls {
                _ = try loadModel(url: url)
                try await Task.sleep(nanoseconds: 300_000_000)
            }
        }
    }

    func loadAndRetainModels() async {
        await runGuarded("load and retain models") {
            let urls = bundledModelURLs()
            modelCount = urls.count
            guard !urls.isEmpty else {
                log("retain: no bundled .mlmodelc found")
                return
            }

            retainedModels.removeAll(keepingCapacity: false)
            for url in urls {
                let model = try loadModel(url: url)
                retainedModels.append((label: url.lastPathComponent, model: model))
                retainedModelCount = retainedModels.count
                log("retain: \(url.lastPathComponent) count=\(retainedModelCount)")
                try await Task.sleep(nanoseconds: 300_000_000)
            }
        }
    }

    func predictOnce() async {
        await runGuarded("predict once") {
            let urls = bundledModelURLs()
            modelCount = urls.count
            guard !urls.isEmpty else {
                log("predict: no bundled .mlmodelc found")
                return
            }

            for url in urls {
                let model = try loadModel(url: url)
                try predict(model: model, label: url.lastPathComponent, iteration: 1)
                try await Task.sleep(nanoseconds: 300_000_000)
            }
        }
    }

    func predictLoop(iterations: Int) async {
        await runGuarded("predict x\(iterations)") {
            let urls = bundledModelURLs()
            modelCount = urls.count
            guard !urls.isEmpty else {
                log("loop: no bundled .mlmodelc found")
                return
            }

            for url in urls {
                let model = try loadModel(url: url)
                for iteration in 1...iterations {
                    try predict(model: model, label: url.lastPathComponent, iteration: iteration)
                    try await Task.sleep(nanoseconds: 250_000_000)
                }
            }
        }
    }

    func predictRetainedLoop(iterations: Int) async {
        await runGuarded("predict retained x\(iterations)") {
            guard !retainedModels.isEmpty else {
                log("retained loop: no retained models")
                return
            }

            for retained in retainedModels {
                for iteration in 1...iterations {
                    try predict(model: retained.model, label: retained.label, iteration: iteration)
                    try await Task.sleep(nanoseconds: 250_000_000)
                }
            }
        }
    }

    func loadTextEncoderOnly() async {
        await runGuarded("load text encoder") {
            let urls = textEncoderModelURLs()
            guard !urls.isEmpty else {
                log("text encoder: no bundled text encoder .mlmodelc found")
                return
            }

            retainedModels.removeAll(keepingCapacity: false)
            for url in urls {
                let model = try loadModel(url: url)
                retainedModels.append((label: url.lastPathComponent, model: model))
                retainedModelCount = retainedModels.count
                log("text encoder retain: \(url.lastPathComponent) count=\(retainedModelCount)")
            }
        }
    }

    func predictTextEncoderOnce() async {
        await runGuarded("predict text encoder once") {
            let model: (label: String, model: MLModel)
            if let retained = retainedModels.first(where: { isTextEncoderName($0.label) }) {
                model = retained
            } else if let url = textEncoderModelURLs().first {
                model = (url.lastPathComponent, try loadModel(url: url))
            } else {
                log("text encoder predict: no bundled text encoder .mlmodelc found")
                return
            }

            let result = try predict(model: model.model, label: model.label, iteration: 1)
            try compareTextEncoderOutput(result.output, promptIndex: textEncoderPromptIndex, prompt: nil)
        }
    }

    func runSeparatedTextEncoderCycle() async {
        await runGuarded("text encoder separated cycle") {
            guard let url = textEncoderModelURLs().first else {
                log("text encoder separated: no bundled text encoder .mlmodelc found")
                return
            }

            retainedModels.removeAll(keepingCapacity: false)
            retainedModelCount = 0
            let probeManifest = try loadTextEncoderProbeManifest()
            log("text encoder separated: begin transient load/predict/release")
            log("text encoder prompts: count=\(probeManifest.prompts.count)")

            try autoreleasepool {
                let model = try loadModel(url: url)
                defer { textEncoderPromptIndex = 0 }
                for (index, prompt) in probeManifest.prompts.enumerated() {
                    textEncoderPromptIndex = index
                    log("text encoder prompt: \(index + 1)/\(probeManifest.prompts.count) \"\(prompt)\"")
                    let result = try predict(model: model, label: url.lastPathComponent, iteration: index + 1)
                    try compareTextEncoderOutput(result.output, promptIndex: index, prompt: prompt)
                }
            }

            log("text encoder separated: transient model scope ended")
            retainedModels.removeAll(keepingCapacity: false)
            retainedModelCount = 0
            try await Task.sleep(nanoseconds: 500_000_000)
            purgeCoreMLCache(context: "after text encoder release")
            log("text encoder separated: ready for generation load")
        }
    }

    func predictPipeline(unetSteps: Int) async {
        await runGuarded("pipeline \(unetSteps)+decode") {
            guard let unet = retainedModels.first(where: { $0.label.lowercased().contains("unet") }) else {
                log("pipeline: no retained UNet model")
                return
            }
            guard let decoder = retainedModels.first(where: { $0.label.lowercased().contains("decoder") }) else {
                log("pipeline: no retained decoder model")
                return
            }

            var modelElapsed: TimeInterval = 0
            for step in 1...unetSteps {
                modelElapsed += try predict(model: unet.model, label: unet.label, iteration: step).elapsed
                await Task.yield()
            }
            modelElapsed += try predict(model: decoder.model, label: decoder.label, iteration: 1).elapsed
            log("pipeline: model_time=\(format(seconds: modelElapsed)) steps=\(unetSteps)+decode")
        }
    }

    private func runGuarded(_ name: String, body: () async throws -> Void) async {
        guard !isRunning else { return }
        isRunning = true
        status = "Running"
        log("start: \(name)")
        let start = Date()

        do {
            try await body()
            let elapsed = Date().timeIntervalSince(start)
            status = "Done"
            log("done: \(name) \(format(seconds: elapsed))")
        } catch {
            status = "Failed"
            log("error: \(name) \(error.localizedDescription)")
        }

        isRunning = false
    }

    private func allocateAndRetain(megabytes: Int) async throws {
        let byteCount = megabytes * 1024 * 1024
        let targetMB = retainedMB + megabytes
        let start = Date()
        status = "Alloc \(targetMB) MB"
        log("memory: attempt +\(megabytes) MB target=\(targetMB) MB")
        var data = Data(count: byteCount)
        data.withUnsafeMutableBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }
            let pointer = base.assumingMemoryBound(to: UInt8.self)
            var offset = 0
            while offset < byteCount {
                pointer[offset] = UInt8(truncatingIfNeeded: offset / 4096)
                offset += 4096
            }
            if byteCount > 0 {
                pointer[byteCount - 1] = 0xA5
            }
        }
        retainedBuffers.append(data)
        retainedMB += megabytes
        lastStableMB = retainedMB
        await Task.yield()
        log("memory: retained +\(megabytes) MB stable=\(retainedMB) MB \(format(seconds: Date().timeIntervalSince(start)))")
    }

    private func loadModel(url: URL) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeMode.computeUnits

        let start = Date()
        log("load: \(url.lastPathComponent) compute=\(computeMode.title)")
        let model = try MLModel(contentsOf: url, configuration: config)
        let elapsed = Date().timeIntervalSince(start)
        log("load: \(url.lastPathComponent) \(format(seconds: elapsed))")
        logModelDescription(model)
        return model
    }

    @discardableResult
    private func predict(model: MLModel, label: String, iteration: Int) throws -> (elapsed: TimeInterval, output: MLFeatureProvider) {
        let input = try makeInputProvider(for: model)
        let start = Date()
        let output = try model.prediction(from: input)
        let elapsed = Date().timeIntervalSince(start)
        let outputs = output.featureNames.sorted().joined(separator: ",")
        log("predict: \(label) #\(iteration) \(format(seconds: elapsed)) outputs=[\(outputs)]")
        return (elapsed, output)
    }

    private func makeInputProvider(for model: MLModel) throws -> MLFeatureProvider {
        var values: [String: MLFeatureValue] = [:]
        let inputs = model.modelDescription.inputDescriptionsByName

        for (name, description) in inputs {
            switch description.type {
            case .multiArray:
                guard let constraint = description.multiArrayConstraint else {
                    throw StressTestError.unsupportedInput(name)
                }
                let shape = constraint.shape.isEmpty ? [1] : constraint.shape
                let array = try MLMultiArray(shape: shape, dataType: constraint.dataType)
                if name == "input_ids", constraint.dataType == .int32 {
                    try fillTextEncoderInputIDs(array)
                }
                values[name] = MLFeatureValue(multiArray: array)
                log("input: \(name) multiArray shape=\(shape.map(\.stringValue).joined(separator: "x")) type=\(constraint.dataType.rawValue)")
            case .int64:
                values[name] = MLFeatureValue(int64: 0)
                log("input: \(name) int64=0")
            case .double:
                values[name] = MLFeatureValue(double: 0)
                log("input: \(name) double=0")
            case .string:
                values[name] = MLFeatureValue(string: "")
                log("input: \(name) string=''")
            default:
                throw StressTestError.unsupportedInput(name)
            }
        }

        return DictionaryFeatureProvider(values: values)
    }

    private func fillTextEncoderInputIDs(_ array: MLMultiArray) throws {
        guard array.dataType == .int32 else {
            throw StressTestError.unsupportedInput("input_ids")
        }
        guard let url = bundledResourceURL(named: "input_ids_i32", extension: "bin") else {
            log("input_ids: no bundled ids asset; using zeros")
            return
        }
        let data = try Data(contentsOf: url)
        let availableCount = data.count / MemoryLayout<Int32>.size
        let startIndex = textEncoderPromptIndex * array.count
        guard startIndex < availableCount else {
            log("input_ids: prompt index \(textEncoderPromptIndex) out of range; using zeros")
            return
        }
        let copyCount = min(array.count, availableCount - startIndex)
        data.withUnsafeBytes { rawBuffer in
            guard let source = rawBuffer.baseAddress?.assumingMemoryBound(to: Int32.self) else { return }
            let destination = array.dataPointer.assumingMemoryBound(to: Int32.self)
            for index in 0..<copyCount {
                destination[index] = source[startIndex + index]
            }
        }
        log("input_ids: loaded \(copyCount) ids prompt=\(textEncoderPromptIndex + 1) from \(url.lastPathComponent)")
    }

    private func compareTextEncoderOutput(_ output: MLFeatureProvider, promptIndex: Int, prompt: String?) throws {
        guard let array = output.featureValue(for: "hidden_states")?.multiArrayValue
            ?? output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first
        else {
            log("text encoder compare: no multiArray output")
            return
        }

        guard let url = bundledResourceURL(named: "reference_hidden_states_f16", extension: "bin") else {
            log("text encoder compare: no bundled reference")
            return
        }

        let data = try Data(contentsOf: url)
        let referenceCount = data.count / MemoryLayout<UInt16>.size
        let startIndex = promptIndex * array.count
        guard startIndex < referenceCount else {
            log("text encoder compare: prompt index \(promptIndex) out of reference range")
            return
        }
        let compareCount = min(array.count, referenceCount - startIndex)
        guard compareCount > 0 else {
            log("text encoder compare: empty reference")
            return
        }

        var sumSquared = 0.0
        var maxAbs = 0.0
        for index in 0..<compareCount {
            let actual = Double(multiArrayFloatValue(array, flatIndex: index))
            let expected = Double(referenceFloat16(data, index: startIndex + index))
            let diff = actual - expected
            sumSquared += diff * diff
            maxAbs = max(maxAbs, abs(diff))
        }

        let rms = sqrt(sumSquared / Double(compareCount))
        let promptLabel = prompt.map { " prompt=\"\($0)\"" } ?? ""
        log(String(format: "text encoder compare:%@ count=%d rms=%.6f max=%.6f", promptLabel, compareCount, rms, maxAbs))
    }

    private func referenceFloat16(_ data: Data, index: Int) -> Float {
        let byteIndex = index * MemoryLayout<UInt16>.size
        let low = UInt16(data[byteIndex])
        let high = UInt16(data[byteIndex + 1]) << 8
        return float16ToFloat(low | high)
    }

    private func multiArrayFloatValue(_ array: MLMultiArray, flatIndex: Int) -> Float {
        let dimensions = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var remaining = flatIndex
        var offset = 0

        for dimensionIndex in stride(from: dimensions.count - 1, through: 0, by: -1) {
            let dimension = max(dimensions[dimensionIndex], 1)
            let coordinate = remaining % dimension
            remaining /= dimension
            offset += coordinate * strides[dimensionIndex]
        }

        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            return float16ToFloat(pointer[offset])
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            return pointer[offset]
        case .double:
            let pointer = array.dataPointer.assumingMemoryBound(to: Double.self)
            return Float(pointer[offset])
        default:
            return 0
        }
    }

    private func float16ToFloat(_ value: UInt16) -> Float {
        let sign = UInt32(value & 0x8000) << 16
        let exponent = Int((value & 0x7C00) >> 10)
        var mantissa = UInt32(value & 0x03FF)
        let bits: UInt32

        if exponent == 0 {
            if mantissa == 0 {
                bits = sign
            } else {
                var adjustedExponent = -14
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1
                    adjustedExponent -= 1
                }
                mantissa &= 0x03FF
                bits = sign | (UInt32(adjustedExponent + 127) << 23) | (mantissa << 13)
            }
        } else if exponent == 0x1F {
            bits = sign | 0x7F80_0000 | (mantissa << 13)
        } else {
            bits = sign | (UInt32(exponent - 15 + 127) << 23) | (mantissa << 13)
        }

        return Float(bitPattern: bits)
    }

    private func logModelDescription(_ model: MLModel) {
        let inputs = model.modelDescription.inputDescriptionsByName.keys.sorted().joined(separator: ",")
        let outputs = model.modelDescription.outputDescriptionsByName.keys.sorted().joined(separator: ",")
        log("desc: inputs=[\(inputs)] outputs=[\(outputs)]")
    }

    private func bundledModelURLs() -> [URL] {
        guard let resourceURL = Bundle.main.resourceURL else { return [] }
        let keys: [URLResourceKey] = [.isDirectoryKey, .nameKey]
        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var urls: [URL] = []
        for case let url as URL in enumerator where url.pathExtension == "mlmodelc" {
            urls.append(url)
        }
        return urls.sorted { $0.path < $1.path }
    }

    private func textEncoderModelURLs() -> [URL] {
        bundledModelURLs().filter { isTextEncoderName($0.lastPathComponent) }
    }

    private func isTextEncoderName(_ name: String) -> Bool {
        name.lowercased().contains("text_encoder")
    }

    private func bundledResourceURL(named name: String, extension pathExtension: String) -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else { return nil }
        let targetFileName = "\(name).\(pathExtension)"
        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL,
            includingPropertiesForKeys: [.nameKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        for case let url as URL in enumerator where url.lastPathComponent == targetFileName {
            return url
        }
        return nil
    }

    private func loadTextEncoderProbeManifest() throws -> TextEncoderProbeManifest {
        if let cachedTextEncoderProbeManifest {
            return cachedTextEncoderProbeManifest
        }
        guard let url = bundledResourceURL(named: "text_encoder_probe_prompts", extension: "json") else {
            throw StressTestError.missingResource("text_encoder_probe_prompts.json")
        }
        let manifest = try JSONDecoder().decode(TextEncoderProbeManifest.self, from: Data(contentsOf: url))
        guard manifest.inputIdsShape.count == 2,
              manifest.inputIdsShape[0] == manifest.prompts.count,
              manifest.inputIdsShape[1] == 77,
              manifest.hiddenStatesShape == [manifest.prompts.count, 77, 768],
              manifest.inputIdsDtype == "int32",
              manifest.hiddenStatesDtype == "float16"
        else {
            throw StressTestError.invalidProbeManifest
        }
        cachedTextEncoderProbeManifest = manifest
        return manifest
    }

    private func purgeCoreMLCache(context: String) {
        let fileManager = FileManager.default
        guard let cachesURL = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            log("cache: missing caches directory \(context)")
            return
        }

        var candidates = [
            cachesURL.appendingPathComponent("com.apple.e5rt.e5bundlecache", isDirectory: true)
        ]
        if let bundleIdentifier = Bundle.main.bundleIdentifier {
            candidates.append(
                cachesURL
                    .appendingPathComponent(bundleIdentifier, isDirectory: true)
                    .appendingPathComponent("com.apple.e5rt.e5bundlecache", isDirectory: true)
            )
        }

        var removedCount = 0
        for url in candidates where fileManager.fileExists(atPath: url.path) {
            do {
                try fileManager.removeItem(at: url)
                removedCount += 1
                log("cache: removed \(shortPath(url)) \(context)")
            } catch {
                log("cache: remove failed \(shortPath(url)) \(context) \(error.localizedDescription)")
            }
        }
        if removedCount == 0 {
            log("cache: no Core ML cache found \(context)")
        }
    }

    private func shortPath(_ url: URL) -> String {
        url.path.replacingOccurrences(of: NSHomeDirectory(), with: "~")
    }

    private func log(_ message: String) {
        let line = "\(timestamp()) \(message)"
        print("[WatchStress] \(line)")
        logger.info("\(line, privacy: .public)")
        logLines.insert(StressLogLine(text: line), at: 0)
        if logLines.count > 80 {
            logLines.removeLast(logLines.count - 80)
        }
    }

    private func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }

    private func format(seconds: TimeInterval) -> String {
        String(format: "%.3fs", seconds)
    }
}

struct StressTestView: View {
    @StateObject private var viewModel = StressTestViewModel()

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        Text(viewModel.status)
                        Spacer()
                        Text("\(viewModel.retainedMB) MB")
                            .monospacedDigit()
                    }
                    HStack {
                        Text("Stable")
                        Spacer()
                        Text("\(viewModel.lastStableMB) MB")
                            .monospacedDigit()
                    }
                    HStack {
                        Text("Models")
                        Spacer()
                        Text("\(viewModel.modelCount)")
                            .monospacedDigit()
                    }
                    HStack {
                        Text("Retained")
                        Spacer()
                        Text("\(viewModel.retainedModelCount)")
                            .monospacedDigit()
                    }
                    Picker("Compute", selection: $viewModel.computeMode) {
                        ForEach(StressComputeMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                }

                Section {
                    Button(viewModel.isTextEncoderSmokeOnly ? "Run Text Encoder" : "Encode & Release") {
                        Task { await viewModel.runSeparatedTextEncoderCycle() }
                    }
                }
                .disabled(viewModel.isRunning)

                if !viewModel.isTextEncoderSmokeOnly {
                    Section {
                        Button("Scan Models") {
                            viewModel.scanModels()
                        }
                        Button("+4 MB") {
                            Task { await viewModel.addMemory(megabytes: 4) }
                        }
                        Button("+8 MB") {
                            Task { await viewModel.addMemory(megabytes: 8) }
                        }
                        Button("+16 MB") {
                            Task { await viewModel.addMemory(megabytes: 16) }
                        }
                        Button("+32 MB") {
                            Task { await viewModel.addMemory(megabytes: 32) }
                        }
                        Button("Fine Ladder") {
                            Task { await viewModel.runFineMemoryLadder() }
                        }
                        Button("Aggressive Ladder") {
                            Task { await viewModel.runAggressiveMemoryLadder() }
                        }
                        Button("Release Memory") {
                            viewModel.releaseBuffers()
                        }
                    }
                    .disabled(viewModel.isRunning)

                    Section {
                        Button("Load Only") {
                            Task { await viewModel.loadModelsOnly() }
                        }
                        Button("Load & Retain") {
                            Task { await viewModel.loadAndRetainModels() }
                        }
                        Button("Release Models") {
                            viewModel.releaseModels()
                        }
                        Button("Predict Once") {
                            Task { await viewModel.predictOnce() }
                        }
                        Button("Predict x4") {
                            Task { await viewModel.predictLoop(iterations: 4) }
                        }
                        Button("Retained x4") {
                            Task { await viewModel.predictRetainedLoop(iterations: 4) }
                        }
                        Button("Pipeline 4+Decode") {
                            Task { await viewModel.predictPipeline(unetSteps: 4) }
                        }
                    }
                    .disabled(viewModel.isRunning)

                    Section {
                        Button("Load Text Encoder") {
                            Task { await viewModel.loadTextEncoderOnly() }
                        }
                        Button("Predict Text Encoder") {
                            Task { await viewModel.predictTextEncoderOnce() }
                        }
                    }
                    .disabled(viewModel.isRunning)
                }

                Section {
                    ForEach(viewModel.logLines.prefix(12)) { line in
                        Text(line.text)
                            .font(.system(size: 9, design: .monospaced))
                            .lineLimit(3)
                    }
                }
            }
            .navigationTitle(viewModel.isTextEncoderSmokeOnly ? "Text Encoder" : "Stress")
            .task {
                await viewModel.prepareForLaunch()
            }
        }
    }
}

private final class DictionaryFeatureProvider: MLFeatureProvider {
    let values: [String: MLFeatureValue]

    init(values: [String: MLFeatureValue]) {
        self.values = values
    }

    var featureNames: Set<String> {
        Set(values.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

private enum StressTestError: LocalizedError {
    case unsupportedInput(String)
    case missingResource(String)
    case invalidProbeManifest

    var errorDescription: String? {
        switch self {
        case .unsupportedInput(let name):
            return "unsupported input: \(name)"
        case .missingResource(let name):
            return "missing resource: \(name)"
        case .invalidProbeManifest:
            return "invalid text encoder probe manifest"
        }
    }
}
