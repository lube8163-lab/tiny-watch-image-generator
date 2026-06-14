import Accelerate
import Foundation

public struct TinyGeneratedImage {
    public let width: Int
    public let height: Int
    public let rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) {
        self.width = width
        self.height = height
        self.rgba = rgba
    }
}

public enum TinyImagePostprocess {
    case none
    case watchDenoise
}

private struct TinyPromptSlots {
    let subjects: [String]
    let colors: [String]
    let actions: [String]
    let views: [String]
    let modifiers: [String]
    let styles: [String]
    let unknownTokens: [String]

    func phraseTokens(includeViews: Bool) -> [String] {
        var tokens: [String] = []
        tokens.append(contentsOf: styles)
        tokens.append(contentsOf: colors)
        tokens.append(contentsOf: modifiers)
        tokens.append(contentsOf: subjects)
        tokens.append(contentsOf: actions)
        if includeViews {
            for view in views {
                tokens.append(contentsOf: Self.viewPhraseTokens(view))
            }
        }
        tokens.append(contentsOf: unknownTokens)
        return tokens
    }

    private static func viewPhraseTokens(_ view: String) -> [String] {
        switch view {
        case "front", "side", "back", "top":
            return [view, "view"]
        default:
            return [view]
        }
    }
}

public struct TinyImageGenerator {
    public static let defaultSize = 128
    private static let inferenceBatchSize = 64

    public init() {}

    public func generate(
        prompt: String = "",
        seed: UInt64,
        size: Int = Self.defaultSize,
        postprocess: TinyImagePostprocess = .watchDenoise
    ) -> TinyGeneratedImage {
        let latent = makeLatent(prompt: prompt, seed: seed)
        let frequencies = TinyWeights.coordFrequencies
        let w1 = TinyWeights.w1
        let b1 = TinyWeights.b1
        let w2 = TinyWeights.w2
        let b2 = TinyWeights.b2
        let w3 = TinyWeights.w3
        let b3 = TinyWeights.b3
        let batchLimit = Self.inferenceBatchSize
        var input = [Float](repeating: 0, count: batchLimit * TinyWeights.inputCount)
        var h1 = [Float](repeating: 0, count: batchLimit * TinyWeights.hiddenCount)
        var h2 = [Float](repeating: 0, count: batchLimit * TinyWeights.hiddenCount)
        var out = [Float](repeating: 0, count: batchLimit * TinyWeights.outputCount)
        var pixels = [UInt8]()
        pixels.reserveCapacity(size * size * 4)

        let pixelCount = size * size
        var batchStart = 0
        while batchStart < pixelCount {
            let batchCount = min(batchLimit, pixelCount - batchStart)
            for batchIndex in 0..<batchCount {
                let pixelIndex = batchStart + batchIndex
                let px = pixelIndex % size
                let py = pixelIndex / size
                let fx = (Float(px) / Float(max(size - 1, 1))) * 2.0 - 1.0
                let fy = (Float(py) / Float(max(size - 1, 1))) * 2.0 - 1.0
                let r = min(1.0, sqrtf(fx * fx + fy * fy))

                var inputIndex = batchIndex * TinyWeights.inputCount
                if frequencies.isEmpty {
                    input[inputIndex] = fx
                    inputIndex += 1
                    input[inputIndex] = fy
                    inputIndex += 1
                    input[inputIndex] = r
                    inputIndex += 1
                    input[inputIndex] = sinf(fx * 6.0 + latent[0] * 3.0)
                    inputIndex += 1
                    input[inputIndex] = cosf(fy * 6.0 + latent[1] * 3.0)
                    inputIndex += 1
                    input[inputIndex] = 1.0
                    inputIndex += 1
                } else {
                    input[inputIndex] = fx
                    inputIndex += 1
                    input[inputIndex] = fy
                    inputIndex += 1
                    input[inputIndex] = r
                    inputIndex += 1
                    input[inputIndex] = 1.0
                    inputIndex += 1
                    for frequency in frequencies {
                        let phaseX = latent[0] * (0.35 * frequency)
                        let phaseY = latent[1] * (0.35 * frequency)
                        let scaledX = fx * (Float.pi * frequency)
                        let scaledY = fy * (Float.pi * frequency)
                        input[inputIndex] = sinf(scaledX + phaseX)
                        inputIndex += 1
                        input[inputIndex] = cosf(scaledX + phaseX)
                        inputIndex += 1
                        input[inputIndex] = sinf(scaledY + phaseY)
                        inputIndex += 1
                        input[inputIndex] = cosf(scaledY + phaseY)
                        inputIndex += 1
                    }
                }
                for value in latent {
                    input[inputIndex] = value
                    inputIndex += 1
                }
            }

            inferBatch(
                input,
                batchCount: batchCount,
                w1: w1,
                b1: b1,
                w2: w2,
                b2: b2,
                w3: w3,
                b3: b3,
                h1: &h1,
                h2: &h2,
                out: &out
            )

            for batchIndex in 0..<batchCount {
                let outputIndex = batchIndex * TinyWeights.outputCount
                pixels.append(toByte(out[outputIndex]))
                pixels.append(toByte(out[outputIndex + 1]))
                pixels.append(toByte(out[outputIndex + 2]))
                pixels.append(255)
            }
            batchStart += batchCount
        }

        switch postprocess {
        case .none:
            return TinyGeneratedImage(width: size, height: size, rgba: pixels)
        case .watchDenoise:
            return TinyGeneratedImage(
                width: size,
                height: size,
                rgba: Self.watchDenoise(rgba: pixels, width: size, height: size)
            )
        }
    }

    private static func watchDenoise(rgba: [UInt8], width: Int, height: Int) -> [UInt8] {
        guard width > 2, height > 2 else { return rgba }

        let pixelCount = width * height
        let background = estimateBackground(rgba: rgba, width: width, height: height)
        var luminance = [Float](repeating: 0, count: pixelCount)
        var cb = [Float](repeating: 0, count: pixelCount)
        var cr = [Float](repeating: 0, count: pixelCount)
        var colorMask = [Float](repeating: 0, count: pixelCount)

        for index in 0..<pixelCount {
            let base = index * 4
            let red = Float(rgba[base])
            let green = Float(rgba[base + 1])
            let blue = Float(rgba[base + 2])
            let y = 0.299 * red + 0.587 * green + 0.114 * blue
            luminance[index] = y
            cb[index] = blue - y
            cr[index] = red - y
            let distance = colorDistance(y: y, cb: cb[index], cr: cr[index], background: background)
            colorMask[index] = smoothstep(18.0, 56.0, distance)
        }

        var yBlur = [Float](repeating: 0, count: pixelCount)
        var cbBlur = [Float](repeating: 0, count: pixelCount)
        var crBlur = [Float](repeating: 0, count: pixelCount)
        var maskBlur = [Float](repeating: 0, count: pixelCount)

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x

                var ySum: Float = 0
                for dy in -1...1 {
                    let sampleY = min(height - 1, max(0, y + dy))
                    for dx in -1...1 {
                        let sampleX = min(width - 1, max(0, x + dx))
                        ySum += luminance[sampleY * width + sampleX]
                    }
                }
                yBlur[index] = ySum / 9.0

                var cbSum: Float = 0
                var crSum: Float = 0
                for dy in -2...2 {
                    let sampleY = min(height - 1, max(0, y + dy))
                    for dx in -2...2 {
                        let sampleX = min(width - 1, max(0, x + dx))
                        let sampleIndex = sampleY * width + sampleX
                        cbSum += cb[sampleIndex]
                        crSum += cr[sampleIndex]
                    }
                }
                cbBlur[index] = cbSum / 25.0
                crBlur[index] = crSum / 25.0

                var maskMax: Float = 0
                var maskSum: Float = 0
                for dy in -1...1 {
                    let sampleY = min(height - 1, max(0, y + dy))
                    for dx in -1...1 {
                        let sampleX = min(width - 1, max(0, x + dx))
                        let sampleMask = colorMask[sampleY * width + sampleX]
                        maskMax = max(maskMax, sampleMask)
                        maskSum += sampleMask
                    }
                }
                maskBlur[index] = max(maskSum / 9.0, maskMax * 0.72)
            }
        }

        var output = rgba
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let left = y * width + max(0, x - 1)
                let right = y * width + min(width - 1, x + 1)
                let up = max(0, y - 1) * width + x
                let down = min(height - 1, y + 1) * width + x
                let gradient = abs(luminance[index] - luminance[left])
                    + abs(luminance[index] - luminance[right])
                    + abs(luminance[index] - luminance[up])
                    + abs(luminance[index] - luminance[down])
                let edge = min(1.0 as Float, gradient / 128.0)
                let edgeMask = smoothstep(34.0, 120.0, gradient)
                let edgeSupport = edgeMask * min(1.0 as Float, maskBlur[index] + 0.18)
                let foregroundMask = min(
                    1.0 as Float,
                    max(colorMask[index], max(maskBlur[index], edgeSupport))
                )
                let backgroundMask = (1.0 - foregroundMask) * (1.0 - edge * 0.55)
                let foregroundSmooth = foregroundMask * (1.0 - edge)

                let lumaAmount = 0.08 * foregroundSmooth + 0.62 * backgroundMask
                let chromaAmount = 0.18 * foregroundSmooth + 0.82 * backgroundMask
                let newY = luminance[index] + (yBlur[index] - luminance[index]) * lumaAmount
                var newCb = cb[index] + (cbBlur[index] - cb[index]) * chromaAmount
                var newCr = cr[index] + (crBlur[index] - cr[index]) * chromaAmount
                var matteY = newY
                let matteAmount = 0.28 * backgroundMask
                matteY += (background.y - matteY) * matteAmount
                newCb += (background.cb - newCb) * (0.50 * backgroundMask)
                newCr += (background.cr - newCr) * (0.50 * backgroundMask)

                let red = matteY + newCr
                let blue = matteY + newCb
                let green = (matteY - 0.299 * red - 0.114 * blue) / 0.587
                let base = index * 4
                output[base] = byteClamped(red)
                output[base + 1] = byteClamped(green)
                output[base + 2] = byteClamped(blue)
            }
        }
        return output
    }

    private struct BackgroundEstimate {
        let red: Float
        let green: Float
        let blue: Float
        let y: Float
        let cb: Float
        let cr: Float
    }

    private static func estimateBackground(rgba: [UInt8], width: Int, height: Int) -> BackgroundEstimate {
        let patchSize = max(2, min(12, min(width, height) / 8))
        let cornerMeans = [
            meanColor(rgba: rgba, width: width, xRange: 0..<patchSize, yRange: 0..<patchSize),
            meanColor(rgba: rgba, width: width, xRange: (width - patchSize)..<width, yRange: 0..<patchSize),
            meanColor(rgba: rgba, width: width, xRange: 0..<patchSize, yRange: (height - patchSize)..<height),
            meanColor(rgba: rgba, width: width, xRange: (width - patchSize)..<width, yRange: (height - patchSize)..<height)
        ]
        let initial = pairedCornerMean(cornerMeans) ?? cornerMeans.min { left, right in
            left.variance < right.variance
        } ?? meanColor(rgba: rgba, width: width, xRange: 0..<width, yRange: 0..<height)

        let borderWidth = max(2, min(8, min(width, height) / 16))
        var distanceSum: Float = 0
        var distanceSquareSum: Float = 0
        var sampleCount: Float = 0

        for y in 0..<height {
            for x in 0..<width where isBorder(x: x, y: y, width: width, height: height, borderWidth: borderWidth) {
                let base = (y * width + x) * 4
                let red = Float(rgba[base])
                let green = Float(rgba[base + 1])
                let blue = Float(rgba[base + 2])
                let distance = rgbDistance(red, green, blue, initial.red, initial.green, initial.blue)
                distanceSum += distance
                distanceSquareSum += distance * distance
                sampleCount += 1.0
            }
        }

        guard sampleCount > 0 else {
            return backgroundEstimate(red: initial.red, green: initial.green, blue: initial.blue)
        }

        let meanDistance = distanceSum / sampleCount
        let variance = max(0.0 as Float, distanceSquareSum / sampleCount - meanDistance * meanDistance)
        let threshold = max(14.0 as Float, meanDistance + sqrtf(variance) * 0.85)
        var redSum: Float = 0
        var greenSum: Float = 0
        var blueSum: Float = 0
        var keptCount: Float = 0

        for y in 0..<height {
            for x in 0..<width where isBorder(x: x, y: y, width: width, height: height, borderWidth: borderWidth) {
                let base = (y * width + x) * 4
                let red = Float(rgba[base])
                let green = Float(rgba[base + 1])
                let blue = Float(rgba[base + 2])
                let distance = rgbDistance(red, green, blue, initial.red, initial.green, initial.blue)
                if distance <= threshold {
                    redSum += red
                    greenSum += green
                    blueSum += blue
                    keptCount += 1.0
                }
            }
        }

        if keptCount < max(8.0 as Float, sampleCount * 0.12) {
            return backgroundEstimate(red: initial.red, green: initial.green, blue: initial.blue)
        }
        return backgroundEstimate(red: redSum / keptCount, green: greenSum / keptCount, blue: blueSum / keptCount)
    }

    private struct ColorSummary {
        let red: Float
        let green: Float
        let blue: Float
        let variance: Float
    }

    private static func meanColor(
        rgba: [UInt8],
        width: Int,
        xRange: Range<Int>,
        yRange: Range<Int>
    ) -> ColorSummary {
        var redSum: Float = 0
        var greenSum: Float = 0
        var blueSum: Float = 0
        var redSquareSum: Float = 0
        var greenSquareSum: Float = 0
        var blueSquareSum: Float = 0
        var count: Float = 0

        for y in yRange {
            for x in xRange {
                let base = (y * width + x) * 4
                let red = Float(rgba[base])
                let green = Float(rgba[base + 1])
                let blue = Float(rgba[base + 2])
                redSum += red
                greenSum += green
                blueSum += blue
                redSquareSum += red * red
                greenSquareSum += green * green
                blueSquareSum += blue * blue
                count += 1.0
            }
        }

        guard count > 0 else {
            return ColorSummary(red: 0, green: 0, blue: 0, variance: Float.greatestFiniteMagnitude)
        }

        let redMean = redSum / count
        let greenMean = greenSum / count
        let blueMean = blueSum / count
        let variance = max(
            0.0 as Float,
            (redSquareSum + greenSquareSum + blueSquareSum) / count
                - (redMean * redMean + greenMean * greenMean + blueMean * blueMean)
        ) / 3.0

        return ColorSummary(red: redMean, green: greenMean, blue: blueMean, variance: variance)
    }

    private static func pairedCornerMean(_ summaries: [ColorSummary]) -> ColorSummary? {
        guard summaries.count >= 2 else { return nil }

        var bestLeft = 0
        var bestRight = 1
        var bestScore = Float.greatestFiniteMagnitude

        for left in summaries.indices {
            for right in summaries.indices where right > left {
                let distance = rgbDistance(
                    summaries[left].red,
                    summaries[left].green,
                    summaries[left].blue,
                    summaries[right].red,
                    summaries[right].green,
                    summaries[right].blue
                )
                let variancePenalty = sqrtf(max(0.0 as Float, summaries[left].variance + summaries[right].variance))
                let score = distance + variancePenalty * 0.15
                if score < bestScore {
                    bestScore = score
                    bestLeft = left
                    bestRight = right
                }
            }
        }

        return ColorSummary(
            red: (summaries[bestLeft].red + summaries[bestRight].red) * 0.5,
            green: (summaries[bestLeft].green + summaries[bestRight].green) * 0.5,
            blue: (summaries[bestLeft].blue + summaries[bestRight].blue) * 0.5,
            variance: (summaries[bestLeft].variance + summaries[bestRight].variance) * 0.5
        )
    }

    private static func backgroundEstimate(red: Float, green: Float, blue: Float) -> BackgroundEstimate {
        let y = 0.299 * red + 0.587 * green + 0.114 * blue
        return BackgroundEstimate(red: red, green: green, blue: blue, y: y, cb: blue - y, cr: red - y)
    }

    private static func colorDistance(
        y: Float,
        cb: Float,
        cr: Float,
        background: BackgroundEstimate
    ) -> Float {
        let luma = abs(y - background.y)
        let chroma = (abs(cb - background.cb) + abs(cr - background.cr)) * 0.5
        return luma * 0.68 + chroma * 0.62
    }

    private static func rgbDistance(
        _ red: Float,
        _ green: Float,
        _ blue: Float,
        _ otherRed: Float,
        _ otherGreen: Float,
        _ otherBlue: Float
    ) -> Float {
        let dr = red - otherRed
        let dg = green - otherGreen
        let db = blue - otherBlue
        return sqrtf((dr * dr + dg * dg + db * db) / 3.0)
    }

    private static func isBorder(x: Int, y: Int, width: Int, height: Int, borderWidth: Int) -> Bool {
        x < borderWidth || y < borderWidth || x >= width - borderWidth || y >= height - borderWidth
    }

    private static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        guard edge0 != edge1 else {
            return x < edge0 ? 0.0 : 1.0
        }
        let t = max(0.0 as Float, min(1.0 as Float, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)
    }

    private func inferBatch(
        _ input: [Float],
        batchCount: Int,
        w1: [Float],
        b1: [Float],
        w2: [Float],
        b2: [Float],
        w3: [Float],
        b3: [Float],
        h1: inout [Float],
        h2: inout [Float],
        out: inout [Float]
    ) {
        denseBatch(
            input,
            batchCount: batchCount,
            inputCount: TinyWeights.inputCount,
            weights: w1,
            outputCount: TinyWeights.hiddenCount,
            bias: b1,
            output: &h1
        )
        for i in 0..<(batchCount * TinyWeights.hiddenCount) {
            h1[i] = fastTanh(h1[i])
        }

        denseBatch(
            h1,
            batchCount: batchCount,
            inputCount: TinyWeights.hiddenCount,
            weights: w2,
            outputCount: TinyWeights.hiddenCount,
            bias: b2,
            output: &h2
        )
        for i in 0..<(batchCount * TinyWeights.hiddenCount) {
            h2[i] = fastTanh(h2[i])
        }

        denseBatch(
            h2,
            batchCount: batchCount,
            inputCount: TinyWeights.hiddenCount,
            weights: w3,
            outputCount: TinyWeights.outputCount,
            bias: b3,
            output: &out
        )
        for i in 0..<(batchCount * TinyWeights.outputCount) {
            out[i] = sigmoid(out[i] * 1.8)
        }
    }

    private func denseBatch(
        _ input: [Float],
        batchCount: Int,
        inputCount: Int,
        weights: [Float],
        outputCount: Int,
        bias: [Float],
        output: inout [Float]
    ) {
        input.withUnsafeBufferPointer { inputBuffer in
            weights.withUnsafeBufferPointer { weightBuffer in
                output.withUnsafeMutableBufferPointer { outputBuffer in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        Int32(batchCount),
                        Int32(outputCount),
                        Int32(inputCount),
                        1.0,
                        inputBuffer.baseAddress!,
                        Int32(inputCount),
                        weightBuffer.baseAddress!,
                        Int32(inputCount),
                        0.0,
                        outputBuffer.baseAddress!,
                        Int32(outputCount)
                    )
                }
            }
        }
        for row in 0..<batchCount {
            let rowStart = row * outputCount
            for column in 0..<outputCount {
                output[rowStart + column] += bias[column]
            }
        }
    }

    private func makeLatent(prompt: String, seed: UInt64) -> [Float] {
        if TinyWeights.promptEncoder == "compositional_v1" {
            return makeCompositionalLatent(prompt: prompt, seed: seed, version: 1)
        }
        if TinyWeights.promptEncoder == "compositional_v2" {
            return makeCompositionalLatent(prompt: prompt, seed: seed, version: 2)
        }
        return makeHashLatent(prompt: prompt, seed: seed)
    }

    private func makeHashLatent(prompt: String, seed: UInt64) -> [Float] {
        var state = seed ^ hash(canonicalPrompt(prompt))
        if state == 0 {
            state = 0x9E3779B97F4A7C15
        }
        var latent = [Float]()
        latent.reserveCapacity(TinyWeights.latentCount)

        for _ in 0..<TinyWeights.latentCount {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            let v = Float(state & 0xffff) / 32767.5 - 1.0
            latent.append(v)
        }

        return latent
    }

    private func makeCompositionalLatent(prompt: String, seed: UInt64, version: Int) -> [Float] {
        let rawTokens = tokenizePrompt(prompt)
        let rawPhrase = rawTokens.joined(separator: " ")
        let canonical = canonicalPrompt(prompt)
        let slots = promptSlots(tokens: rawTokens, phrase: rawPhrase, includeViews: version >= 2)
        let slotTokens = slots.phraseTokens(includeViews: version >= 2)
        let tokens = slotTokens.isEmpty ? rawTokens : slotTokens
        let tokenSet = Set(tokens)
        let phrase = tokens.joined(separator: " ")
        var latent = [Float](repeating: 0, count: TinyWeights.latentCount)

        let seedWeight: Float = version == 1 ? 0.35 : 0.22
        let subjectWeight: Float = version == 1 ? 0.75 : 0.95
        let featureRepeats = version == 1 ? 2 : 4
        let secondaryRepeats = version == 1 ? 2 : 3

        addSeedNoise(to: &latent, seed: seed, canonical: canonical, weight: seedWeight)
        if slots.subjects.isEmpty {
            if !canonical.isEmpty {
                addFeature(to: &latent, name: "subject:\(canonical)", weight: 0.45, repeats: secondaryRepeats)
            }
        } else {
            for subject in slots.subjects {
                addFeature(to: &latent, name: "subject:\(subject)", weight: subjectWeight, repeats: featureRepeats)
            }
        }

        for color in slots.colors {
            addFeature(to: &latent, name: "color:\(color)", weight: 0.60, repeats: secondaryRepeats)
        }
        for action in slots.actions {
            addFeature(to: &latent, name: "action:\(action)", weight: 0.65, repeats: secondaryRepeats)
        }
        if version >= 2 {
            for view in slots.views {
                addFeature(to: &latent, name: "view:\(view)", weight: 0.60, repeats: secondaryRepeats)
            }
        }
        for modifier in slots.modifiers {
            addFeature(to: &latent, name: "modifier:\(modifier)", weight: 0.42, repeats: secondaryRepeats)
        }
        for style in slots.styles {
            addFeature(to: &latent, name: "style:\(style)", weight: 0.38, repeats: secondaryRepeats)
        }

        let recognized = recognizedTokens(tokens: tokenSet, phrase: phrase, includeViews: version >= 2)
        let stopwords = promptStopwords()
        for token in tokens.filter({ !stopwords.contains($0) && !tokenIsRecognized($0, recognized: recognized) }).prefix(8) {
            addFeature(to: &latent, name: "token:\(token)", weight: 0.18)
        }
        for index in 0..<max(tokens.count - 1, 0) {
            let left = tokens[index]
            let right = tokens[index + 1]
            if !stopwords.contains(left) && !stopwords.contains(right) {
                addFeature(to: &latent, name: "bigram:\(left)_\(right)", weight: 0.12)
            }
        }
        if !phrase.isEmpty {
            addFeature(to: &latent, name: "full:\(phrase)", weight: 0.10)
        }

        for index in latent.indices {
            latent[index] = max(-1.0, min(1.0, latent[index]))
        }
        return latent
    }

    private func promptSlots(tokens: [String], phrase: String, includeViews: Bool) -> TinyPromptSlots {
        let tokenSet = Set(tokens)
        let recognized = recognizedTokens(tokens: tokenSet, phrase: phrase, includeViews: includeViews)
        let stopwords = promptStopwords()
        let subjects = matchedAliasKeys(tokens: tokenSet, phrase: phrase, aliases: subjectAliases())
        let colors = matchedAliasKeys(tokens: tokenSet, phrase: phrase, aliases: colorAliases())
        let actions = matchedAliasKeys(tokens: tokenSet, phrase: phrase, aliases: actionAliases())
        let views = includeViews ? matchedAliasKeys(tokens: tokenSet, phrase: phrase, aliases: viewAliases()) : []
        let modifiers = matchedAliasKeys(tokens: tokenSet, phrase: phrase, aliases: modifierAliases())
        let styles = matchedAliasKeys(tokens: tokenSet, phrase: phrase, aliases: styleAliases())
        let hasKnownSlots = !subjects.isEmpty || !colors.isEmpty || !actions.isEmpty ||
            !views.isEmpty || !modifiers.isEmpty || !styles.isEmpty
        let unknownTokens = hasKnownSlots ? [] : tokens.filter { token in
            !stopwords.contains(token) && !tokenIsRecognized(token, recognized: recognized)
        }
        return TinyPromptSlots(
            subjects: subjects,
            colors: colors,
            actions: actions,
            views: views,
            modifiers: modifiers,
            styles: styles,
            unknownTokens: unknownTokens
        )
    }

    private func canonicalPrompt(_ prompt: String) -> String {
        let text = normalizedPrompt(prompt)
        let tokens = tokenizePrompt(text)
        if tokens.isEmpty {
            return ""
        }
        let tokenSet = Set(tokens)
        let aliases = subjectAliases()
        let phrase = tokens.joined(separator: " ")
        let composites: [(String, [String])] = [
            ("astronaut horse", ["astronaut", "horse"])
        ]
        for (key, requiredKeys) in composites {
            if requiredKeys.allSatisfy({ hasAlias(tokenSet, phrase: phrase, key: $0, aliases: aliases) }) {
                return key
            }
        }
        for (key, words) in aliases {
            if aliasMatches(tokens: tokenSet, phrase: phrase, alias: key) ||
                words.contains(where: { aliasMatches(tokens: tokenSet, phrase: phrase, alias: $0) }) {
                return key
            }
        }
        return phrase
    }

    private func hasAlias(
        _ tokens: Set<String>,
        phrase: String,
        key: String,
        aliases: [(String, [String])]
    ) -> Bool {
        if aliasMatches(tokens: tokens, phrase: phrase, alias: key) {
            return true
        }
        guard let words = aliases.first(where: { $0.0 == key })?.1 else {
            return false
        }
        return words.contains(where: { aliasMatches(tokens: tokens, phrase: phrase, alias: $0) })
    }

    private func normalizedPrompt(_ prompt: String) -> String {
        prompt
            .lowercased()
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
    }

    private func tokenizePrompt(_ prompt: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        for scalar in normalizedPrompt(prompt).unicodeScalars {
            if isPromptTokenScalar(scalar) {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }
        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }

    private func isPromptTokenScalar(_ scalar: UnicodeScalar) -> Bool {
        let value = scalar.value
        return (value >= 48 && value <= 57) ||
            (value >= 97 && value <= 122) ||
            (value >= 0x3040 && value <= 0x30ff) ||
            (value >= 0x3400 && value <= 0x9fff)
    }

    private func addSeedNoise(to values: inout [Float], seed: UInt64, canonical: String, weight: Float) {
        var state = seed ^ hash("seed:\(canonical)")
        if state == 0 {
            state = 0x9E3779B97F4A7C15
        }
        for index in values.indices {
            nextXorshift(&state)
            values[index] += (Float(state & 0xffff) / 32767.5 - 1.0) * weight
        }
    }

    private func addFeature(to values: inout [Float], name: String, weight: Float, repeats: Int = 2) {
        if values.isEmpty {
            return
        }
        var state = hash(name)
        for _ in 0..<repeats {
            nextXorshift(&state)
            let index = Int(state % UInt64(values.count))
            nextXorshift(&state)
            let sign: Float = (state & 1) == 1 ? 1.0 : -1.0
            values[index] += sign * weight
        }
    }

    private func nextXorshift(_ state: inout UInt64) {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
    }

    private func matchedAliasKeys(
        tokens: Set<String>,
        phrase: String,
        aliases: [(String, [String])]
    ) -> [String] {
        var output: [String] = []
        for (key, words) in aliases {
            if aliasMatches(tokens: tokens, phrase: phrase, alias: key) ||
                words.contains(where: { aliasMatches(tokens: tokens, phrase: phrase, alias: $0) }) {
                output.append(key)
            }
        }
        return output
    }

    private func recognizedTokens(tokens: Set<String>, phrase: String, includeViews: Bool = false) -> Set<String> {
        var output = Set<String>()
        var aliasGroups = [subjectAliases(), colorAliases(), actionAliases(), modifierAliases(), styleAliases()]
        if includeViews {
            aliasGroups.append(viewAliases())
        }
        for aliases in aliasGroups {
            for (key, words) in aliases {
                if aliasMatches(tokens: tokens, phrase: phrase, alias: key) {
                    output.formUnion(tokenizePrompt(key))
                }
                for word in words where aliasMatches(tokens: tokens, phrase: phrase, alias: word) {
                    output.formUnion(tokenizePrompt(word))
                }
            }
        }
        return output
    }

    private func tokenIsRecognized(_ token: String, recognized: Set<String>) -> Bool {
        if recognized.contains(token) {
            return true
        }
        if token.unicodeScalars.allSatisfy({ $0.value <= 127 }) {
            return false
        }
        return recognized.contains { part in
            !part.isEmpty &&
                part.unicodeScalars.contains(where: { $0.value > 127 }) &&
                token.contains(part)
        }
    }

    private func aliasMatches(tokens: Set<String>, phrase: String, alias: String) -> Bool {
        let aliasTokens = tokenizePrompt(alias)
        if aliasTokens.isEmpty {
            return false
        }
        if aliasTokens.allSatisfy({ tokens.contains($0) }) {
            return true
        }
        if alias.unicodeScalars.contains(where: { $0.value > 127 }) {
            let compactAlias = normalizedPrompt(alias).replacingOccurrences(of: " ", with: "")
            let compactPhrase = phrase.replacingOccurrences(of: " ", with: "")
            return compactPhrase.contains(compactAlias)
        }
        return false
    }

    private func subjectAliases() -> [(String, [String])] {
        [
            ("astronaut", ["astronaut", "spaceperson", "spaceman", "宇宙飛行士"]),
            ("alien", ["alien", "aliens", "extraterrestrial", "宇宙人"]),
            ("dragon", ["dragon", "dragons", "竜", "ドラゴン"]),
            ("penguin", ["penguin", "penguins", "ペンギン"]),
            ("turtle", ["turtle", "turtles", "亀", "カメ"]),
            ("elephant", ["elephant", "elephants", "象", "ゾウ"]),
            ("lion", ["lion", "lions", "ライオン"]),
            ("monkey", ["monkey", "monkeys", "猿", "サル"]),
            ("frog", ["frog", "frogs", "蛙", "カエル"]),
            ("duck", ["duck", "ducks", "アヒル", "鴨"]),
            ("deer", ["deer", "鹿", "シカ"]),
            ("whale", ["whale", "whales", "くじら", "クジラ"]),
            ("cat", ["cat", "cats", "kitten", "kitty", "ねこ", "ネコ", "猫"]),
            ("dog", ["dog", "dogs", "puppy", "いぬ", "イヌ", "犬"]),
            ("apple", ["apple", "apples", "りんご", "リンゴ"]),
            ("robot", ["robot", "robots", "android", "ロボット"]),
            ("rabbit", ["rabbit", "rabbits", "bunny", "うさぎ", "兎"]),
            ("horse", ["horse", "horses", "pony", "馬"]),
            ("bear", ["bear", "bears", "熊"]),
            ("fox", ["fox", "foxes", "きつね", "狐"]),
            ("owl", ["owl", "owls", "ふくろう"]),
            ("butterfly", ["butterfly", "butterflies", "蝶"]),
            ("star", ["star", "stars", "星"]),
            ("sun", ["sun", "sunny", "太陽"]),
            ("moon", ["moon", "lunar", "月"]),
            ("car", ["car", "cars", "auto", "automobile", "vehicle", "車"]),
            ("bus", ["bus", "buses", "バス"]),
            ("bicycle", ["bicycle", "bicycles", "bike", "cycle", "自転車"]),
            ("airplane", ["airplane", "airplanes", "plane", "aircraft", "飛行機"]),
            ("boat", ["boat", "boats", "ship", "船"]),
            ("tree", ["tree", "trees", "forest", "木", "森"]),
            ("mountain", ["mountain", "mountains", "山"]),
            ("cloud", ["cloud", "clouds", "雲"]),
            ("flower", ["flower", "flowers", "floral", "blossom", "rose", "tulip", "sunflower", "daisy", "orchid", "花"]),
            ("house", ["house", "houses", "home", "building", "家"]),
            ("bird", ["bird", "birds", "cardinal", "peacock", "parrot", "eagle", "sparrow", "鳥"]),
            ("fish", ["fish", "fishes", "魚"]),
            ("train", ["train", "trains", "railway", "電車", "列車"]),
            ("castle", ["castle", "castles", "城"]),
            ("banana", ["banana", "bananas", "バナナ"]),
            ("orange", ["orange", "oranges", "オレンジ"]),
            ("strawberry", ["strawberry", "strawberries", "いちご"]),
            ("cake", ["cake", "cakes", "ケーキ"]),
            ("pizza", ["pizza", "pizzas", "ピザ"]),
            ("bread", ["bread", "loaf", "パン"]),
            ("book", ["book", "books", "本"]),
            ("chair", ["chair", "chairs", "椅子"]),
            ("clock", ["clock", "clocks", "watch", "時計"]),
            ("cup", ["cup", "cups", "mug", "コップ"]),
            ("mushroom", ["mushroom", "mushrooms", "きのこ"]),
            ("heart", ["heart", "hearts", "ハート"]),
            ("ball", ["ball", "balls", "ボール"]),
            ("guitar", ["guitar", "guitars", "ギター"]),
            ("camera", ["camera", "cameras", "カメラ"]),
            ("shoe", ["shoe", "shoes", "sneaker", "靴"]),
            ("umbrella", ["umbrella", "umbrellas", "傘", "かさ"]),
            ("key", ["key", "keys", "鍵", "かぎ"]),
            ("bottle", ["bottle", "bottles", "ボトル", "瓶"]),
            ("pencil", ["pencil", "pencils", "鉛筆", "えんぴつ"]),
            ("lamp", ["lamp", "lamps", "light", "ライト", "ランプ"]),
            ("phone", ["phone", "phones", "smartphone", "スマホ", "携帯"]),
            ("computer", ["computer", "computers", "laptop", "pc", "パソコン"]),
            ("crown", ["crown", "crowns", "王冠"]),
            ("diamond", ["diamond", "diamonds", "gem", "jewel", "宝石", "ダイヤ"]),
            ("sword", ["sword", "swords", "剣"]),
            ("shield", ["shield", "shields", "盾"]),
            ("cactus", ["cactus", "cacti", "サボテン"]),
            ("volcano", ["volcano", "volcanoes", "火山"]),
            ("fire", ["fire", "flame", "flames", "炎", "火"]),
            ("icecream", ["icecream", "ice cream", "ice-cream", "アイス", "アイスクリーム"]),
            ("donut", ["donut", "donuts", "doughnut", "doughnuts", "ドーナツ"]),
            ("sushi", ["sushi", "寿司", "すし"]),
            ("face", ["face", "faces", "portrait", "person", "girl", "boy", "顔", "人物", "女の子", "男の子"])
        ]
    }

    private func colorAliases() -> [(String, [String])] {
        [
            ("red", ["red", "scarlet", "赤", "赤い"]),
            ("orange", ["orange", "橙", "オレンジ色"]),
            ("yellow", ["yellow", "gold", "golden", "黄色", "金色"]),
            ("green", ["green", "緑", "緑色"]),
            ("blue", ["blue", "cyan", "青", "青い", "水色"]),
            ("purple", ["purple", "violet", "紫"]),
            ("pink", ["pink", "ピンク"]),
            ("brown", ["brown", "茶色"]),
            ("black", ["black", "黒", "黒い"]),
            ("white", ["white", "白", "白い"]),
            ("gray", ["gray", "grey", "silver", "銀色", "灰色"])
        ]
    }

    private func actionAliases() -> [(String, [String])] {
        [
            ("sitting", ["sitting", "sit", "seated", "座る", "座っている"]),
            ("standing", ["standing", "stand", "立つ", "立っている"]),
            ("running", ["running", "run", "走る", "走っている"]),
            ("walking", ["walking", "walk", "歩く", "歩いている"]),
            ("flying", ["flying", "fly", "飛ぶ", "飛んでいる"]),
            ("swimming", ["swimming", "swim", "泳ぐ", "泳いでいる"]),
            ("sleeping", ["sleeping", "sleep", "眠る", "寝ている"]),
            ("eating", ["eating", "eat", "食べる", "食べている"]),
            ("holding", ["holding", "hold", "持つ", "持っている"]),
            ("wearing", ["wearing", "wear", "着る", "着ている"]),
            ("jumping", ["jumping", "jump", "跳ぶ", "ジャンプ"]),
            ("parked", ["parked", "parking", "駐車"]),
            ("floating", ["floating", "float", "drifting", "drift", "浮く", "浮いている"]),
            ("tilted", ["tilted", "tilted slightly", "leaning", "lean", "swaying", "sway", "turning", "turn", "傾く", "斜め"]),
            ("shining", ["shining", "shine", "glowing", "glow", "sparkling", "sparkle", "輝く", "光る"]),
            ("rolling", ["rolling", "roll", "転がる", "転がっている"]),
            ("bouncing", ["bouncing", "bounce", "跳ねる", "弾む"]),
            ("smiling", ["smiling", "smile", "happy", "笑顔", "笑う"]),
            ("looking", ["looking", "look", "facing", "facing forward", "looking left", "looking right", "見る", "向く"]),
            ("dancing", ["dancing", "dance", "踊る", "踊っている"]),
            ("climbing", ["climbing", "climb", "登る", "登っている"]),
            ("spinning", ["spinning", "spin", "回る", "回転"]),
            ("sliding", ["sliding", "slide", "滑る", "滑っている"]),
            ("open", ["open", "opening", "開く", "開いている"])
        ]
    }

    private func viewAliases() -> [(String, [String])] {
        [
            ("front", ["front", "front view", "正面"]),
            ("side", ["side", "side view", "profile", "横向き", "横"]),
            ("back", ["back", "rear", "back view", "後ろ"]),
            ("top", ["top", "top view", "overhead", "上から"]),
            ("closeup", ["closeup", "close up", "close-up", "macro", "アップ"])
        ]
    }

    private func modifierAliases() -> [(String, [String])] {
        [
            ("small", ["small", "tiny", "mini", "little", "小さい", "小さな"]),
            ("large", ["large", "big", "giant", "huge", "大きい", "大きな"]),
            ("cute", ["cute", "kawaii", "かわいい"]),
            ("simple", ["simple", "clean", "minimal", "plain", "シンプル"]),
            ("detailed", ["detailed", "intricate", "細かい"]),
            ("round", ["round", "circular", "丸い"]),
            ("bright", ["bright", "shiny", "glowing", "明るい"]),
            ("dark", ["dark", "night", "暗い", "夜"]),
            ("fluffy", ["fluffy", "soft", "fuzzy", "ふわふわ"]),
            ("striped", ["striped", "stripe", "stripes", "しましま", "縞"]),
            ("spotted", ["spotted", "spots", "polka dot", "水玉", "斑点"]),
            ("open", ["open", "opened", "開いた"]),
            ("closed", ["closed", "shut", "閉じた"]),
            ("broken", ["broken", "cracked", "壊れた", "割れた"]),
            ("wooden", ["wooden", "wood", "木製"]),
            ("metallic", ["metallic", "metal", "silver", "金属", "金属製"]),
            ("transparent", ["transparent", "clear", "glass", "透明"]),
            ("sharp", ["sharp", "pointy", "尖った"]),
            ("spiky", ["spiky", "spike", "spines", "とげ", "トゲ"])
        ]
    }

    private func styleAliases() -> [(String, [String])] {
        [
            ("photo", ["photo", "photograph", "realistic", "写真"]),
            ("anime", ["anime", "manga", "アニメ"]),
            ("cartoon", ["cartoon", "toon", "comic"]),
            ("icon", ["icon", "symbol", "emoji", "sticker", "アイコン"]),
            ("toy", ["toy", "plush", "figurine", "おもちゃ"]),
            ("watercolor", ["watercolor", "painting", "painted", "水彩"]),
            ("sketch", ["sketch", "drawing", "lineart", "線画"])
        ]
    }

    private func promptStopwords() -> Set<String> {
        [
            "a", "an", "the", "of", "on", "in", "with", "and", "or", "to",
            "for", "by", "at", "from", "one", "single", "centered", "clear",
            "background", "image", "picture", "illustration"
        ]
    }

    private func hash(_ prompt: String) -> UInt64 {
        var value: UInt64 = 0xcbf29ce484222325
        for byte in prompt.utf8 {
            value ^= UInt64(byte)
            value &*= 0x100000001b3
        }
        return value
    }

    private func fastTanh(_ x: Float) -> Float {
        let clamped = max(-3.0, min(3.0, x))
        return tanhf(clamped)
    }

    private func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + expf(-x))
    }

    private func toByte(_ x: Float) -> UInt8 {
        UInt8(max(0, min(255, Int((x * 255.0).rounded()))))
    }

    private static func byteClamped(_ x: Float) -> UInt8 {
        UInt8(max(0, min(255, Int(x.rounded()))))
    }
}
