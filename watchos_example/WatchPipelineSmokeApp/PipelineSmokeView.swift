import CoreGraphics
@preconcurrency import CoreML
import Darwin
import Foundation
import SwiftUI

struct PipelinePromptPreset: Decodable, Identifiable, Hashable {
    let key: String
    let title: String
    let prompt: String
    let aliases: [String]

    var id: String { key }
}

private struct PipelinePromptPresetFile: Decodable {
    let embeddingShape: [Int]
    let uncondEmbeddingShape: [Int]
    let embeddingDtype: String
    let presets: [PipelinePromptPreset]
}

private struct PipelineScheduler: Decodable {
    let timesteps: [Int]
    let sigmas: [Float]
    let latentShape: [Int]
    let decodedShape: [Int]
    let predictionType: String
    let finalSigmasType: String
    let guidanceScale: Float
}

private struct PipelineLCMPresetFile: Decodable {
    let embeddingShape: [Int]
    let embeddingDtype: String
    let presets: [PipelinePromptPreset]
}

private struct PipelineLCMScheduler: Decodable {
    let steps: [PipelineLCMStep]
    let guidanceScale: Float
    let timestepCondShape: [Int]
    let latentShape: [Int]
    let decodedShape: [Int]
    let predictionType: String
}

private struct PipelineLCMStep: Decodable {
    let timestep: Int
    let sqrtAlpha: Float
    let sqrtBeta: Float
    let sqrtAlphaPrev: Float
    let sqrtBetaPrev: Float
    let cSkip: Float
    let cOut: Float
}

enum PipelineFamily: String, CaseIterable, Identifiable {
    case sd128
    case lcm64
    case lcm64SixBit
    case lcm128
    case lcm128SixBit
    case lcm192SixBit

    static let selectableCases: [PipelineFamily] = [.lcm192SixBit]

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sd128:
            return "SD128"
        case .lcm64:
            return "LCM64"
        case .lcm64SixBit:
            return "LCM64 6b"
        case .lcm128:
            return "LCM128"
        case .lcm128SixBit:
            return "LCM128 6b"
        case .lcm192SixBit:
            return "LCM192 6b"
        }
    }

    var isLCM: Bool {
        switch self {
        case .sd128:
            return false
        case .lcm64, .lcm64SixBit, .lcm128, .lcm128SixBit, .lcm192SixBit:
            return true
        }
    }

    var lcmUNetLabel: String {
        switch self {
        case .sd128:
            return ""
        case .lcm64:
            return "LCM 4-bit"
        case .lcm64SixBit:
            return "LCM 6-bit"
        case .lcm128:
            return "LCM 128 4-bit"
        case .lcm128SixBit:
            return "LCM 128 6-bit 16p"
        case .lcm192SixBit:
            return "LCM 192 6-bit 16p"
        }
    }

    var lcmUNetResourceNames: [String] {
        switch self {
        case .sd128:
            return []
        case .lcm64:
            return [
                "lcm_unet_8x8_4bit_part1",
                "lcm_unet_8x8_4bit_part2",
                "lcm_unet_8x8_4bit_part3",
                "lcm_unet_8x8_4bit_part4"
            ]
        case .lcm64SixBit:
            return [
                "lcm_unet_8x8_6bit_part1",
                "lcm_unet_8x8_6bit_part2",
                "lcm_unet_8x8_6bit_part3",
                "lcm_unet_8x8_6bit_part4",
                "lcm_unet_8x8_6bit_part5",
                "lcm_unet_8x8_6bit_part6",
                "lcm_unet_8x8_6bit_part7",
                "lcm_unet_8x8_6bit_part8"
            ]
        case .lcm128:
            return [
                "lcm_unet_16x16_4bit_part1",
                "lcm_unet_16x16_4bit_part2",
                "lcm_unet_16x16_4bit_part3",
                "lcm_unet_16x16_4bit_part4",
                "lcm_unet_16x16_4bit_part5",
                "lcm_unet_16x16_4bit_part6",
                "lcm_unet_16x16_4bit_part7",
                "lcm_unet_16x16_4bit_part8"
            ]
        case .lcm128SixBit:
            return [
                "lcm_unet_16x16_6bit_16p_part1",
                "lcm_unet_16x16_6bit_16p_part2",
                "lcm_unet_16x16_6bit_16p_part3",
                "lcm_unet_16x16_6bit_16p_part4",
                "lcm_unet_16x16_6bit_16p_part5",
                "lcm_unet_16x16_6bit_16p_part6",
                "lcm_unet_16x16_6bit_16p_part7",
                "lcm_unet_16x16_6bit_16p_part8",
                "lcm_unet_16x16_6bit_16p_part9",
                "lcm_unet_16x16_6bit_16p_part10",
                "lcm_unet_16x16_6bit_16p_part11",
                "lcm_unet_16x16_6bit_16p_part12",
                "lcm_unet_16x16_6bit_16p_part13",
                "lcm_unet_16x16_6bit_16p_part14",
                "lcm_unet_16x16_6bit_16p_part15",
                "lcm_unet_16x16_6bit_16p_part16"
            ]
        case .lcm192SixBit:
            return [
                "lcm_unet_24x24_6bit_16p_part1",
                "lcm_unet_24x24_6bit_16p_part2",
                "lcm_unet_24x24_6bit_16p_part3",
                "lcm_unet_24x24_6bit_16p_part4",
                "lcm_unet_24x24_6bit_16p_part5",
                "lcm_unet_24x24_6bit_16p_part6",
                "lcm_unet_24x24_6bit_16p_part7",
                "lcm_unet_24x24_6bit_16p_part8",
                "lcm_unet_24x24_6bit_16p_part9",
                "lcm_unet_24x24_6bit_16p_part10",
                "lcm_unet_24x24_6bit_16p_part11",
                "lcm_unet_24x24_6bit_16p_part12",
                "lcm_unet_24x24_6bit_16p_part13",
                "lcm_unet_24x24_6bit_16p_part14",
                "lcm_unet_24x24_6bit_16p_part15",
                "lcm_unet_24x24_6bit_16p_part16"
            ]
        }
    }

    var lcmDecoderResourceName: String {
        switch self {
        case .sd128:
            return ""
        case .lcm64, .lcm64SixBit:
            return "lcm_vae_decoder_64x64_noattn_4bit"
        case .lcm128, .lcm128SixBit:
            return "vae_decoder_128x128_noattn_4bit"
        case .lcm192SixBit:
            return "vae_decoder_192x192_noattn_4bit"
        }
    }

    var lcmLatentShape: [Int] {
        switch self {
        case .sd128:
            return []
        case .lcm64, .lcm64SixBit:
            return [1, 4, 8, 8]
        case .lcm128, .lcm128SixBit:
            return [1, 4, 16, 16]
        case .lcm192SixBit:
            return [1, 4, 24, 24]
        }
    }

    var lcmDecodedShape: [Int] {
        switch self {
        case .sd128:
            return []
        case .lcm64, .lcm64SixBit:
            return [1, 3, 64, 64]
        case .lcm128, .lcm128SixBit:
            return [1, 3, 128, 128]
        case .lcm192SixBit:
            return [1, 3, 192, 192]
        }
    }

    var lcmVAELabel: String {
        switch self {
        case .sd128:
            return ""
        case .lcm64, .lcm64SixBit:
            return "64 4-bit"
        case .lcm128, .lcm128SixBit:
            return "128 4-bit"
        case .lcm192SixBit:
            return "192 4-bit"
        }
    }

    var lcmSchedulerAssetSubdirectory: String {
        switch self {
        case .lcm192SixBit:
            return "LCM192Assets"
        case .sd128, .lcm64, .lcm64SixBit, .lcm128, .lcm128SixBit:
            return "LCMAssets"
        }
    }
}

enum PipelineStepMode: String, CaseIterable, Identifiable {
    case quality30
    case smoke4

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quality30:
            return "30"
        case .smoke4:
            return "4"
        }
    }

    var stepCount: Int {
        switch self {
        case .quality30:
            return 30
        case .smoke4:
            return 4
        }
    }
}

enum PipelinePreviewMode: String, CaseIterable, Identifiable {
    case smooth
    case crisp
    case sharp2x

    var id: String { rawValue }

    var title: String {
        switch self {
        case .smooth:
            return "Smooth"
        case .crisp:
            return "Crisp"
        case .sharp2x:
            return "Sharp x2"
        }
    }

    var interpolation: Image.Interpolation {
        switch self {
        case .smooth:
            return .medium
        case .crisp:
            return .none
        case .sharp2x:
            return .medium
        }
    }

    var shouldInterpolateCGImage: Bool {
        switch self {
        case .smooth, .sharp2x:
            return true
        case .crisp:
            return false
        }
    }

    var usesImagePostprocess: Bool {
        switch self {
        case .smooth, .crisp:
            return false
        case .sharp2x:
            return true
        }
    }
}

enum PipelineUNetMode: String, CaseIterable, Identifiable {
    case fourBit
    case sixBit

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fourBit:
            return "4-bit"
        case .sixBit:
            return "6-bit"
        }
    }

    var resourceName: String {
        switch self {
        case .fourBit:
            return "unet_sd_16x16_4bit"
        case .sixBit:
            return "unet_sd_16x16_6bit"
        }
    }
}

enum PipelineDecoderMode: String, CaseIterable, Identifiable {
    case fourBit
    case fp16

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fourBit:
            return "4-bit"
        case .fp16:
            return "FP16"
        }
    }

    var resourceName: String {
        switch self {
        case .fourBit:
            return "vae_decoder_128x128_noattn_4bit"
        case .fp16:
            return "vae_decoder_128x128_noattn"
        }
    }
}

private struct PipelineDenoiseSchedule {
    let timesteps: [Int]
    let sigmas: [Float]

    var stepCount: Int {
        timesteps.count
    }
}

struct PipelineMetric: Identifiable {
    let id = UUID()
    let label: String
    let value: String
}

struct PipelineLogLine: Identifiable {
    let id = UUID()
    let text: String
}

struct PipelineSeedOption: Identifiable, Hashable {
    let id: String
    let title: String
    let seed: UInt64?
    let isRandom: Bool

    init(id: String, title: String, seed: UInt64?, isRandom: Bool = false) {
        self.id = id
        self.title = title
        self.seed = seed
        self.isRandom = isRandom
    }
}

struct PipelineGuidanceOption: Identifiable, Hashable {
    let id: String
    let title: String
    let value: Float
}

struct PipelineCandidateRun: Identifiable, Hashable {
    let id: String
    let title: String
    let presetKey: String
    let seed: UInt64
    let guidanceScale: Float

    init(title: String, presetKey: String, seed: UInt64, guidanceScale: Float) {
        self.id = "\(presetKey)-\(seed)-\(guidanceScale)"
        self.title = title
        self.presetKey = presetKey
        self.seed = seed
        self.guidanceScale = guidanceScale
    }
}

private struct PipelineTensorStats {
    let min: Float
    let max: Float
    let mean: Float
    let rms: Float

    var summary: String {
        String(format: "min=%.3f max=%.3f mean=%.3f rms=%.3f", min, max, mean, rms)
    }
}

private struct PipelineImageResult {
    let image: CGImage
    let decodedStats: PipelineTensorStats
    let clippedChannels: Int
    let totalChannels: Int
}

private struct PipelinePackageSizeSummary {
    let count: Int
    let totalBytes: Int64
    let maxBytes: Int64
    let maxIndex: Int
}

private struct PipelineChunkTiming {
    let step: Int
    let chunk: Int
    let loadSeconds: TimeInterval
    let predictSeconds: TimeInterval
}

private struct PipelineTemporaryPredictionResult {
    let provider: PipelineDictionaryFeatureProvider
    let loadSeconds: TimeInterval
    let predictSeconds: TimeInterval
}

private struct PipelineConditioningEmbedding {
    let embedding: MLMultiArray
    let source: String
    let runKey: String
    let seedKey: String
}

private struct PipelineTextEncoderResult {
    let embedding: MLMultiArray
    let tokenCount: Int
    let loadSeconds: TimeInterval
    let predictSeconds: TimeInterval
}

private struct PipelineTokenization {
    let ids: [Int32]
    let tokenCount: Int
}

@MainActor
final class PipelineSmokeViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var status = "Idle"
    @Published var promptText = "cat mascot"
    @Published var presets: [PipelinePromptPreset] = []
    @Published var selectedFamily: PipelineFamily = .lcm192SixBit
    @Published var selectedPresetIndex = 0
    @Published var stepMode: PipelineStepMode = .quality30
    @Published var selectedUNetMode: PipelineUNetMode = .fourBit
    @Published var selectedDecoderMode: PipelineDecoderMode = .fourBit
    @Published var selectedSeedIndex = 1
    @Published var selectedGuidanceIndex = 1
    @Published var selectedPreviewMode: PipelinePreviewMode = .sharp2x
    @Published var generatedImage: CGImage?
    @Published var metrics: [PipelineMetric] = []
    @Published var logLines: [PipelineLogLine] = []
    @Published private(set) var lastGeneratedPrompt = ""
    @Published private(set) var lastRunID: String?
    @Published private(set) var lastResolvedSeed: UInt64?

    private let minimumPromptMatchScore = 200
    private let logsDetailedLCMChunks = false
    private let textConditioningStyleClauses = [
        "centered",
        "full object visible",
        "clean anime illustration",
        "simple background"
    ]

    let seedOptions: [PipelineSeedOption] = [
        PipelineSeedOption(id: "curated", title: "Curated", seed: nil),
        PipelineSeedOption(id: "random", title: "Random", seed: nil, isRandom: true),
        PipelineSeedOption(id: "0", title: "0", seed: 0),
        PipelineSeedOption(id: "1", title: "1", seed: 1),
        PipelineSeedOption(id: "2", title: "2", seed: 2),
        PipelineSeedOption(id: "3", title: "3", seed: 3),
        PipelineSeedOption(id: "4", title: "4", seed: 4),
        PipelineSeedOption(id: "5", title: "5", seed: 5),
        PipelineSeedOption(id: "6", title: "6", seed: 6),
        PipelineSeedOption(id: "7", title: "7", seed: 7),
        PipelineSeedOption(id: "8", title: "8", seed: 8),
        PipelineSeedOption(id: "9", title: "9", seed: 9),
        PipelineSeedOption(id: "10", title: "10", seed: 10),
        PipelineSeedOption(id: "11", title: "11", seed: 11),
        PipelineSeedOption(id: "12", title: "12", seed: 12),
        PipelineSeedOption(id: "13", title: "13", seed: 13),
        PipelineSeedOption(id: "14", title: "14", seed: 14),
        PipelineSeedOption(id: "15", title: "15", seed: 15),
        PipelineSeedOption(id: "16", title: "16", seed: 16),
        PipelineSeedOption(id: "17", title: "17", seed: 17),
        PipelineSeedOption(id: "18", title: "18", seed: 18),
        PipelineSeedOption(id: "19", title: "19", seed: 19),
        PipelineSeedOption(id: "20", title: "20", seed: 20),
        PipelineSeedOption(id: "21", title: "21", seed: 21),
        PipelineSeedOption(id: "22", title: "22", seed: 22),
        PipelineSeedOption(id: "23", title: "23", seed: 23),
        PipelineSeedOption(id: "24", title: "24", seed: 24),
        PipelineSeedOption(id: "25", title: "25", seed: 25),
        PipelineSeedOption(id: "26", title: "26", seed: 26),
        PipelineSeedOption(id: "27", title: "27", seed: 27),
        PipelineSeedOption(id: "28", title: "28", seed: 28),
        PipelineSeedOption(id: "29", title: "29", seed: 29),
        PipelineSeedOption(id: "30", title: "30", seed: 30),
        PipelineSeedOption(id: "31", title: "31", seed: 31),
        PipelineSeedOption(id: "42", title: "42", seed: 42),
        PipelineSeedOption(id: "1234", title: "1234", seed: 1234)
    ]

    let guidanceOptions: [PipelineGuidanceOption] = [
        PipelineGuidanceOption(id: "4", title: "4", value: 4),
        PipelineGuidanceOption(id: "6", title: "6", value: 6),
        PipelineGuidanceOption(id: "8", title: "8", value: 8),
        PipelineGuidanceOption(id: "10", title: "10", value: 10),
        PipelineGuidanceOption(id: "12", title: "12", value: 12),
        PipelineGuidanceOption(id: "16", title: "16", value: 16)
    ]

    let candidateRuns: [PipelineCandidateRun] = [
        PipelineCandidateRun(title: "Mascot 1", presetKey: "cat_mascot", seed: 1, guidanceScale: 6),
        PipelineCandidateRun(title: "Mascot 2", presetKey: "cat_mascot", seed: 2, guidanceScale: 6),
        PipelineCandidateRun(title: "Lucky 1", presetKey: "lucky_cat", seed: 1, guidanceScale: 6),
        PipelineCandidateRun(title: "Sticker 16", presetKey: "cat_sticker", seed: 16, guidanceScale: 6),
        PipelineCandidateRun(title: "White 14", presetKey: "white_mascot", seed: 14, guidanceScale: 6),
        PipelineCandidateRun(title: "Tabby 30", presetKey: "tabby_icon", seed: 30, guidanceScale: 6),
        PipelineCandidateRun(title: "Orange 30", presetKey: "orange_cat", seed: 30, guidanceScale: 6),
        PipelineCandidateRun(title: "Logo 3", presetKey: "cat_logo", seed: 3, guidanceScale: 6)
    ]

    var selectedPresetForDisplay: PipelinePromptPreset? {
        guard !presets.isEmpty else { return nil }
        return presets[clampedPresetIndex(selectedPresetIndex)]
    }

    var selectedSeedDisplay: String {
        guard let preset = selectedPresetForDisplay else { return "-" }
        let seedOption = seedOptions[min(selectedSeedIndex, max(0, seedOptions.count - 1))]
        if seedOption.isRandom {
            return "Random"
        }
        let resolved = resolvedLCMSeed(for: preset, seedOption: seedOption)
        return resolved.label
    }

    var curatedSeedDisplay: String {
        guard let preset = selectedPresetForDisplay else { return "-" }
        return "\(curatedLCMSeed(for: preset.key))"
    }

    var expectedRunIDForDisplay: String {
        guard let preset = selectedPresetForDisplay else { return "-" }
        let seedOption = seedOptions[min(selectedSeedIndex, max(0, seedOptions.count - 1))]
        let guidance = guidanceOptions[min(selectedGuidanceIndex, max(0, guidanceOptions.count - 1))]
        let seedToken: String
        if seedOption.isRandom {
            seedToken = "random"
        } else {
            seedToken = "\(resolvedLCMSeed(for: preset, seedOption: seedOption).seed)"
        }
        return makeRunID(
            preset: preset,
            seedToken: seedToken,
            guidanceScale: guidance.value
        )
    }

    var generateButtonTitle: String {
        if isGenerating {
            return "Generating"
        }
        if generatedImage != nil && normalizedPromptInput() == lastGeneratedPrompt {
            return "Reroll Seed"
        }
        return "Generate"
    }

    func applyCandidateRun(_ run: PipelineCandidateRun) {
        guard selectedFamily.isLCM else { return }
        guard presets.contains(where: { $0.key == run.presetKey }) else {
            log("candidate: missing preset \(run.presetKey)")
            return
        }
        selectPreset(key: run.presetKey)
        if let seedIndex = seedOptions.firstIndex(where: { $0.seed == run.seed }) {
            selectedSeedIndex = seedIndex
        }
        if let guidanceIndex = guidanceOptions.firstIndex(where: { abs($0.value - run.guidanceScale) < 0.0001 }) {
            selectedGuidanceIndex = guidanceIndex
        }
        selectedPreviewMode = .sharp2x
        generatedImage = nil
        metrics = []
        log("candidate: \(run.title) preset=\(run.presetKey) seed=\(run.seed) guidance=\(format(float: run.guidanceScale))")
    }

    private var promptFile: PipelinePromptPresetFile?
    private var scheduler: PipelineScheduler?
    private var sdPresets: [PipelinePromptPreset] = []
    private var promptEmbeddingData = Data()
    private var uncondEmbeddingData = Data()
    private var latentSeedData = Data()
    private var lcmPromptFile: PipelineLCMPresetFile?
    private var lcmScheduler: PipelineLCMScheduler?
    private var lcmPromptEmbeddingData = Data()
    private var lcmTimestepCondData = Data()
    private var lcmPresets: [PipelinePromptPreset] = []
    private var lcmUnetChunkURLs: [URL] = []
    private var lcmDecoderURL: URL?
    private var clipTokenizer: PipelineCLIPTokenizer?
    nonisolated(unsafe) private var unetModels: [MLModel] = []
    nonisolated(unsafe) private var decoderModel: MLModel?
    private var retainedFamily: PipelineFamily?
    private var retainedUNetMode: PipelineUNetMode?
    private var retainedDecoderMode: PipelineDecoderMode?
    private var didPrepare = false
    private let defaultLCMPresetKey = "cat_mascot"
    private let defaultSDPresetKey = "cat"

    var canGenerate: Bool {
        !isLoading &&
        !isGenerating &&
        hasLoadedResourcesForSelectedFamily &&
        retainedFamily == selectedFamily &&
        retainedUNetMode == selectedUNetMode &&
        retainedDecoderMode == selectedDecoderMode &&
        !presets.isEmpty
    }

    private var hasLoadedResourcesForSelectedFamily: Bool {
        switch selectedFamily {
        case .sd128:
            return !unetModels.isEmpty && decoderModel != nil
        case .lcm64, .lcm64SixBit, .lcm128, .lcm128SixBit, .lcm192SixBit:
            return !lcmUnetChunkURLs.isEmpty && lcmDecoderURL != nil
        }
    }

    func prepare() {
        guard !didPrepare else { return }
        didPrepare = true

        do {
            try loadAssets()
            Task {
                await loadAndRetainModels()
            }
        } catch {
            status = "Asset error"
            log("error: assets \(error.localizedDescription)")
        }
    }

    func reloadModels() async {
        releaseRetainedModels()
        await loadAndRetainModels()
    }

    func familyDidChange() {
        releaseRetainedModels()
        activatePresetsForSelectedFamily(selectDefault: true)
        generatedImage = nil
        metrics = []
        log("pipeline: selected \(selectedFamily.title); reload models before generate")
    }

    func loadAndRetainModels() async {
        guard !isLoading else { return }
        isLoading = true
        status = "Loading"
        let totalStart = Date()
        let family = selectedFamily
        let unetMode = selectedUNetMode
        let decoderMode = selectedDecoderMode

        do {
            if !unetModels.isEmpty || decoderModel != nil {
                log("load: releasing retained models before loading pipeline=\(family.title)")
                releaseRetainedModels()
            }
            if family.isLCM {
                purgeCoreMLCacheBeforeHeavyLoad()
            }

            switch family {
            case .sd128:
                lcmUnetChunkURLs = []
                lcmDecoderURL = nil
                let unetResources = [(name: unetMode.resourceName, label: "unet \(unetMode.title)")]
                let decoderResource = decoderMode.resourceName
                let decoderLabel = "decoder \(decoderMode.title)"
                let loadedUNets = try unetResources.map { resource in
                    try loadModel(
                        url: try bundledURL(named: resource.name, extension: "mlmodelc"),
                        label: resource.label
                    )
                }
                let decoderURL = try bundledURL(named: decoderResource, extension: "mlmodelc")
                unetModels = loadedUNets
                decoderModel = try loadModel(url: decoderURL, label: decoderLabel)
                let unetLabels = unetResources.map(\.label).joined(separator: "+")
                log("load: retained models pipeline=\(family.title) unets=\(unetLabels) decoder=\(decoderLabel) total=\(format(seconds: Date().timeIntervalSince(totalStart)))")
            case .lcm64, .lcm64SixBit, .lcm128, .lcm128SixBit, .lcm192SixBit:
                unetModels = []
                decoderModel = nil
                lcmUnetChunkURLs = try family.lcmUNetResourceNames.map {
                    try bundledURL(named: $0, extension: "mlmodelc")
                }
                lcmDecoderURL = try bundledURL(
                    named: family.lcmDecoderResourceName,
                    extension: "mlmodelc"
                )
                log(
                    "load: prepared streamed models pipeline=\(family.title) " +
                    "unets=\(family.lcmUNetLabel) chunks=\(lcmUnetChunkURLs.count) decoder=\(family.lcmVAELabel) " +
                    "total=\(format(seconds: Date().timeIntervalSince(totalStart)))"
                )
            }
            retainedFamily = family
            retainedUNetMode = unetMode
            retainedDecoderMode = decoderMode
            status = "Ready"
        } catch {
            status = "Load failed"
            log("error: load \(error.localizedDescription)")
        }

        isLoading = false
    }

    private func releaseRetainedModels() {
        unetModels.removeAll(keepingCapacity: false)
        decoderModel = nil
        lcmUnetChunkURLs = []
        lcmDecoderURL = nil
        retainedFamily = nil
        retainedUNetMode = nil
        retainedDecoderMode = nil
    }

    private func purgeCoreMLCacheBeforeHeavyLoad() {
        purgeCoreMLCache(context: "before LCM load")
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

    func generate() async {
        guard !isGenerating else { return }
        guard hasLoadedResourcesForSelectedFamily else {
            log("generate: missing models or assets")
            return
        }
        guard retainedFamily == selectedFamily else {
            log("generate: reload models for pipeline=\(selectedFamily.title)")
            return
        }
        guard retainedUNetMode == selectedUNetMode else {
            log("generate: reload models for unet=\(selectedUNetMode.title)")
            return
        }
        guard retainedDecoderMode == selectedDecoderMode else {
            log("generate: reload models for decoder=\(selectedDecoderMode.title)")
            return
        }
        guard !presets.isEmpty else {
            log("generate: no prompt presets")
            return
        }

        let hadGeneratedImage = generatedImage != nil
        let previousPrompt = lastGeneratedPrompt
        let previousRunID = lastRunID
        let promptInput = normalizedPromptInput()

        isGenerating = true
        generatedImage = nil
        metrics = []
        status = "Generating"
        let totalStart = Date()

        do {
            if hadGeneratedImage && promptInput == previousPrompt {
                log("reroll: prompt=\"\(promptInput)\" previousRun=\(previousRunID ?? "-")")
            }
            let promptMatch = resolvedPromptPreset(for: promptInput)
            let hasPresetMatch = promptMatch.score >= minimumPromptMatchScore
            guard hasPresetMatch || selectedFamily.isLCM else {
                status = "Unsupported"
                log(
                    "prompt: input=\"\(promptInput)\" unsupported score=\(promptMatch.score) " +
                    "minimum=\(minimumPromptMatchScore)"
                )
                log("generate: unsupported prompt for preset-only pipeline")
                isGenerating = false
                return
            }
            if hasPresetMatch {
                selectedPresetIndex = promptMatch.index
            }
            let presetIndex = promptMatch.index
            let preset = presets[presetIndex]
            let seedIndex = min(selectedSeedIndex, max(0, seedOptions.count - 1))
            let seedOption = seedOptions[seedIndex]
            let guidanceIndex = min(selectedGuidanceIndex, max(0, guidanceOptions.count - 1))
            let guidance = guidanceOptions[guidanceIndex]
            let unetTitle = selectedFamily.isLCM ? selectedFamily.lcmUNetLabel : selectedUNetMode.title
            let decoderTitle = selectedFamily.isLCM ? selectedFamily.lcmVAELabel : selectedDecoderMode.title
            if hasPresetMatch {
                log(
                    "prompt: input=\"\(promptInput)\" resolvedPreset=\(preset.key) score=\(promptMatch.score) " +
                    "embeddingPrompt=\"\(preset.prompt)\""
                )
            } else {
                log(
                    "prompt: input=\"\(promptInput)\" no preset match score=\(promptMatch.score); " +
                    "using text encoder promptKey=\(promptRunKey(promptInput))"
                )
            }
            let presetRole = hasPresetMatch ? "matchedPreset" : "fallbackPreset"
            log("generate: pipeline=\(selectedFamily.title) unet=\(unetTitle) decoder=\(decoderTitle) \(presetRole)=\(preset.key) seedMode=\(seedOption.title) guidance=\(guidance.title) prompt=\"\(promptInput)\"")

            if selectedFamily.isLCM {
                guard let lcmDecoderURL else {
                    throw PipelineSmokeError.missingResource("\(selectedFamily.lcmDecoderResourceName).mlmodelc")
                }
                try await generateLCM(
                    unetURLs: lcmUnetChunkURLs,
                    decoderURL: lcmDecoderURL,
                    unetLabel: selectedFamily.lcmUNetLabel,
                    vaeLabel: selectedFamily.lcmVAELabel,
                    latentShape: selectedFamily.lcmLatentShape,
                    decodedShape: selectedFamily.lcmDecodedShape,
                    promptInput: promptInput,
                    presetIndex: presetIndex,
                    preset: preset,
                    seedOption: seedOption,
                    guidanceScale: guidance.value,
                    totalStart: totalStart
                )
                isGenerating = false
                return
            }

            guard let decoderModel, let promptFile, let scheduler else {
                log("generate: missing SD models or assets")
                isGenerating = false
                return
            }
            let unetModel = unetModels[0]
            let promptEmbedding = try makePromptEmbedding(
                promptFile: promptFile,
                presetIndex: presetIndex
            )
            let uncondEmbedding = try makeUncondEmbedding(promptFile: promptFile)

            var latents = try makeInitialLatents(
                scheduler: scheduler,
                presetKey: preset.key,
                seedOption: seedOption
            )
            var previousModelOutput: [Float]?
            let schedule = try makeDenoiseSchedule(scheduler: scheduler, mode: stepMode)
            log("schedule: mode=\(stepMode.title) steps=\(schedule.stepCount) guidance=\(format(float: guidance.value)) timesteps=[\(scheduleSummary(schedule.timesteps))]")
            log("initial: latents \(stats(latents).summary)")

            var stepMetrics: [PipelineMetric] = []
            for stepIndex in 0..<schedule.stepCount {
                let timestep = schedule.timesteps[stepIndex]
                let stepStart = Date()

                let uncondResult = try await predictNoiseValues(
                    unetModel: unetModel,
                    latents: latents,
                    timestep: timestep,
                    embedding: uncondEmbedding,
                    scheduler: scheduler
                )
                let condResult = try await predictNoiseValues(
                    unetModel: unetModel,
                    latents: latents,
                    timestep: timestep,
                    embedding: promptEmbedding,
                    scheduler: scheduler
                )
                if stepIndex == 0 {
                    log("unet: uncond \(uncondResult.description)")
                    log("unet: cond \(condResult.description)")
                }
                let noiseStats = try updateLatents(
                    latents: &latents,
                    previousModelOutput: &previousModelOutput,
                    uncondNoise: uncondResult.values,
                    condNoise: condResult.values,
                    guidanceScale: guidance.value,
                    schedule: schedule,
                    stepIndex: stepIndex
                )
                let elapsed = Date().timeIntervalSince(stepStart)

                let metric = PipelineMetric(
                    label: "UNet \(stepIndex + 1)",
                    value: format(seconds: elapsed)
                )
                stepMetrics.append(metric)
                log(
                    "step: \(stepIndex + 1)/\(schedule.stepCount) t=\(timestep) " +
                    "sigma=\(format(float: schedule.sigmas[stepIndex]))->\(format(float: schedule.sigmas[stepIndex + 1])) " +
                    "uncond=\(format(seconds: uncondResult.elapsed)) cond=\(format(seconds: condResult.elapsed)) " +
                    "total=\(format(seconds: elapsed)) noise \(noiseStats.summary) latents \(stats(latents).summary)"
                )
                await Task.yield()
            }

            log("final: latents \(stats(latents).summary)")
            let decodeInput = PipelineDictionaryFeatureProvider(values: [
                "latents": MLFeatureValue(multiArray: try makeLatentArray(values: latents, shape: scheduler.latentShape))
            ])
            let decoderStart = Date()
            let decodeOutput = try await decoderModel.prediction(from: decodeInput)
            let decoderElapsed = Date().timeIntervalSince(decoderStart)
            let decoded = try decodeOutput.multiArray(named: "decoded")
            log("decoder: output \(multiArrayDescription(decoded))")
            let imageResult = try makeImage(fromDecodedArray: decoded)
            generatedImage = imageResult.image

            let totalElapsed = Date().timeIntervalSince(totalStart)
            let runID = makeRunID(
                preset: preset,
                seedToken: seedOption.seed.map { String($0) } ?? "asset",
                guidanceScale: guidance.value
            )
            lastGeneratedPrompt = promptInput
            lastRunID = runID
            lastResolvedSeed = seedOption.seed
            metrics = stepMetrics + [
                PipelineMetric(label: "Run ID", value: runID),
                PipelineMetric(label: "Preset", value: preset.title),
                PipelineMetric(label: "UNet", value: selectedUNetMode.title),
                PipelineMetric(label: "VAE", value: selectedDecoderMode.title),
                PipelineMetric(label: "Seed", value: seedOption.title),
                PipelineMetric(label: "Guidance", value: guidance.title),
                PipelineMetric(label: "Preview", value: selectedPreviewMode.title),
                PipelineMetric(label: "Decoder", value: format(seconds: decoderElapsed)),
                PipelineMetric(label: "Total", value: format(seconds: totalElapsed)),
                PipelineMetric(label: "Final RMS", value: format(float: stats(latents).rms)),
                PipelineMetric(label: "Decoded RMS", value: format(float: imageResult.decodedStats.rms)),
                PipelineMetric(label: "Clipped", value: "\(imageResult.clippedChannels)/\(imageResult.totalChannels)")
            ]
            status = "Done"
            log("decode: \(format(seconds: decoderElapsed))")
            log("run: \(runID)")
            log("done: total=\(format(seconds: totalElapsed))")
        } catch {
            status = "Failed"
            log("error: generate \(error.localizedDescription)")
        }

        isGenerating = false
    }

    private func loadAssets() throws {
        let promptURL = try bundledURL(named: "prompt_presets", extension: "json", subdirectory: "PromptAssets")
        let schedulerURL = try bundledURL(named: "sd_ddim_scheduler", extension: "json", subdirectory: "PromptAssets")
        let promptEmbeddingURL = try bundledURL(named: "prompt_embeddings_f16", extension: "bin", subdirectory: "PromptAssets")
        let uncondURL = try bundledURL(named: "uncond_embedding_f16", extension: "bin", subdirectory: "PromptAssets")
        let latentSeedURL = try bundledURL(named: "latent_seed_f16", extension: "bin", subdirectory: "PipelineAssets")
        let lcmPromptURL = try bundledURL(named: "prompt_presets", extension: "json", subdirectory: "LCMAssets")
        let lcmSchedulerURL = try bundledURL(
            named: "lcm_scheduler",
            extension: "json",
            subdirectory: selectedFamily.lcmSchedulerAssetSubdirectory
        )
        let lcmPromptEmbeddingURL = try bundledURL(named: "prompt_embeddings_f16", extension: "bin", subdirectory: "LCMAssets")
        let lcmTimestepCondURL = try bundledURL(named: "timestep_cond_f16", extension: "bin", subdirectory: "LCMAssets")

        let decoder = JSONDecoder()
        let loadedPromptFile = try decoder.decode(PipelinePromptPresetFile.self, from: Data(contentsOf: promptURL))
        let loadedScheduler = try decoder.decode(PipelineScheduler.self, from: Data(contentsOf: schedulerURL))
        let loadedLCMPromptFile = try decoder.decode(PipelineLCMPresetFile.self, from: Data(contentsOf: lcmPromptURL))
        let loadedLCMScheduler = try decoder.decode(PipelineLCMScheduler.self, from: Data(contentsOf: lcmSchedulerURL))

        promptFile = loadedPromptFile
        scheduler = loadedScheduler
        sdPresets = loadedPromptFile.presets
        promptEmbeddingData = try Data(contentsOf: promptEmbeddingURL)
        uncondEmbeddingData = try Data(contentsOf: uncondURL)
        latentSeedData = try Data(contentsOf: latentSeedURL)
        lcmPromptFile = loadedLCMPromptFile
        lcmScheduler = loadedLCMScheduler
        lcmPresets = loadedLCMPromptFile.presets
        lcmPromptEmbeddingData = try Data(contentsOf: lcmPromptEmbeddingURL)
        lcmTimestepCondData = try Data(contentsOf: lcmTimestepCondURL)
        try validateLoadedAssets(
            promptFile: loadedPromptFile,
            scheduler: loadedScheduler,
            lcmPromptFile: loadedLCMPromptFile,
            lcmScheduler: loadedLCMScheduler
        )
        activatePresetsForSelectedFamily(selectDefault: true)

        log("assets: presets=\(presets.count) embeddingBytes=\(promptEmbeddingData.count)")
        log("assets: scheduler steps=\(loadedScheduler.timesteps.count) latent=\(loadedScheduler.latentShape.map(String.init).joined(separator: "x"))")
        log("assets: lcm presets=\(lcmPresets.count) embeddingBytes=\(lcmPromptEmbeddingData.count) steps=\(loadedLCMScheduler.steps.count) latent=\(loadedLCMScheduler.latentShape.map(String.init).joined(separator: "x"))")
        if presets.indices.contains(selectedPresetIndex) {
            let selectedPreset = presets[selectedPresetIndex]
            log("assets: default preset=\(selectedPreset.key) title=\"\(selectedPreset.title)\"")
        }
    }

    private func validateLoadedAssets(
        promptFile: PipelinePromptPresetFile,
        scheduler: PipelineScheduler,
        lcmPromptFile: PipelineLCMPresetFile,
        lcmScheduler: PipelineLCMScheduler
    ) throws {
        try validatePromptPresetAsset(
            embeddingShape: promptFile.embeddingShape,
            embeddingDtype: promptFile.embeddingDtype,
            presetCount: promptFile.presets.count,
            embeddingBytes: promptEmbeddingData.count
        )
        guard uncondEmbeddingData.count == (try f16ByteCount(shape: promptFile.uncondEmbeddingShape)) else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        guard latentSeedData.count == (try f16ByteCount(shape: scheduler.latentShape)) else {
            throw PipelineSmokeError.invalidLatentSeed
        }
        try validatePromptPresetAsset(
            embeddingShape: lcmPromptFile.embeddingShape,
            embeddingDtype: lcmPromptFile.embeddingDtype,
            presetCount: lcmPromptFile.presets.count,
            embeddingBytes: lcmPromptEmbeddingData.count
        )
        guard lcmTimestepCondData.count == (try f16ByteCount(shape: lcmScheduler.timestepCondShape)) else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        log(
            "assets: validated sdPrompts=\(promptFile.presets.count) " +
            "lcmPrompts=\(lcmPromptFile.presets.count)"
        )
    }

    private func validatePromptPresetAsset(
        embeddingShape: [Int],
        embeddingDtype: String,
        presetCount: Int,
        embeddingBytes: Int
    ) throws {
        guard
            embeddingDtype == "float16",
            embeddingShape.count == 3,
            embeddingShape[0] == presetCount,
            embeddingBytes == (try f16ByteCount(shape: embeddingShape))
        else {
            throw PipelineSmokeError.invalidPromptAsset
        }
    }

    private func f16ByteCount(shape: [Int]) throws -> Int {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw PipelineSmokeError.invalidArrayShape
        }
        return shape.reduce(1, *) * MemoryLayout<UInt16>.size
    }

    private func activatePresetsForSelectedFamily(selectDefault: Bool = false) {
        switch selectedFamily {
        case .sd128:
            presets = sdPresets
        case .lcm64, .lcm64SixBit, .lcm128, .lcm128SixBit, .lcm192SixBit:
            presets = lcmPresets
        }
        if selectDefault {
            selectPreset(key: defaultPresetKey(for: selectedFamily))
        } else {
            selectedPresetIndex = clampedPresetIndex(selectedPresetIndex)
        }
    }

    private func defaultPresetKey(for family: PipelineFamily) -> String {
        switch family {
        case .sd128:
            return defaultSDPresetKey
        case .lcm64, .lcm64SixBit, .lcm128, .lcm128SixBit, .lcm192SixBit:
            return defaultLCMPresetKey
        }
    }

    private func selectPreset(key: String) {
        if let index = presets.firstIndex(where: { $0.key == key }) {
            selectedPresetIndex = index
        } else {
            selectedPresetIndex = clampedPresetIndex(selectedPresetIndex)
        }
    }

    private func clampedPresetIndex(_ index: Int) -> Int {
        min(index, max(0, presets.count - 1))
    }

    private func normalizedPromptInput() -> String {
        let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return trimmed
        }
        if let preset = selectedPresetForDisplay {
            return preset.prompt
        }
        return "cat mascot"
    }

    private func resolvedPromptPreset(for prompt: String) -> (index: Int, score: Int) {
        guard !presets.isEmpty else {
            return (0, 0)
        }

        let selectedIndex = clampedPresetIndex(selectedPresetIndex)
        let query = normalizedSearchText(prompt)
        let queryTokens = tokenSet(query)
        guard !query.isEmpty else {
            return (selectedIndex, 0)
        }

        var bestIndex = selectedIndex
        var bestScore = -1
        for (index, preset) in presets.enumerated() {
            let score = promptMatchScore(query: query, queryTokens: queryTokens, preset: preset)
            if score > bestScore {
                bestIndex = index
                bestScore = score
            }
        }

        if bestScore < minimumPromptMatchScore {
            return (selectedIndex, 0)
        }
        return (bestIndex, bestScore)
    }

    private func promptMatchScore(
        query: String,
        queryTokens: Set<String>,
        preset: PipelinePromptPreset
    ) -> Int {
        let candidates = [preset.key, preset.title, preset.prompt] + preset.aliases
        var bestScore = 0

        for candidate in candidates {
            let normalized = normalizedSearchText(candidate)
            guard !normalized.isEmpty else { continue }
            if normalized == query {
                bestScore = max(bestScore, 1_000 + normalized.count)
            } else if normalized.contains(query) || query.contains(normalized) {
                bestScore = max(bestScore, 700 + normalized.count)
            }

            let candidateTokens = tokenSet(normalized)
            let overlap = queryTokens.intersection(candidateTokens).count
            if overlap > 0 {
                bestScore = max(bestScore, overlap * 100 + candidateTokens.count)
            }
        }

        return bestScore
    }

    private func normalizedSearchText(_ text: String) -> String {
        let lowercased = text.lowercased()
        var scalars: [UnicodeScalar] = []
        var previousWasSpace = true

        for scalar in lowercased.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                scalars.append(scalar)
                previousWasSpace = false
            } else if !previousWasSpace {
                scalars.append(" ")
                previousWasSpace = true
            }
        }

        return String(String.UnicodeScalarView(scalars)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func tokenSet(_ text: String) -> Set<String> {
        Set(text.split(separator: " ").map(String.init).filter { $0.count > 1 })
    }

    private func promptRunKey(_ prompt: String) -> String {
        let slug = normalizedSearchText(prompt)
            .split(separator: " ")
            .joined(separator: "_")
        return slug.isEmpty ? "prompt" : slug
    }

    private func expandedTextConditioningPrompt(for prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return "cat mascot, single subject, " + textConditioningStyleClauses.joined(separator: ", ")
        }

        let normalized = normalizedSearchText(trimmed)
        let preservesPluralIntent = containsAnyNormalizedTerm(
            in: normalized,
            words: ["two", "three", "multiple", "many", "group", "crowd", "pair"]
        )
        let hasCompositionalRelation = containsAnyNormalizedTerm(
            in: normalized,
            words: ["in", "on", "with", "riding", "holding", "wearing", "inside", "under", "over", "near"]
        )
        let isCloseupPrompt = containsAnyNormalizedTerm(
            in: normalized,
            words: ["face", "head", "portrait", "closeup", "close"]
        )
        var clauses = [trimmed]

        if !preservesPluralIntent && !containsAnyNormalizedTerm(in: normalized, words: ["single", "one", "solo"]) {
            clauses.append(hasCompositionalRelation ? "single centered composition" : "single subject")
        }

        if !containsAnyNormalizedTerm(in: normalized, words: ["center", "centered"]) {
            clauses.append("centered")
        }
        if !isCloseupPrompt && !containsAnyNormalizedTerm(in: normalized, words: ["full", "visible", "body"]) {
            clauses.append("full object visible")
        }
        if !containsAnyNormalizedTerm(
            in: normalized,
            words: [
                "anime",
                "illustration",
                "illustrated",
                "icon",
                "logo",
                "sticker",
                "photo",
                "photograph",
                "photorealistic",
                "pixel",
                "3d"
            ]
        ) {
            clauses.append("clean anime illustration")
        } else if !containsAnyNormalizedTerm(in: normalized, words: ["clean", "simple"]) {
            clauses.append("clean illustration")
        }
        if !containsAnyNormalizedTerm(in: normalized, words: ["background", "scene", "landscape"]) {
            clauses.append("simple background")
        }

        return clauses.joined(separator: ", ")
    }

    private func containsAnyNormalizedTerm(in normalizedText: String, words: [String]) -> Bool {
        let tokens = tokenSet(normalizedText)
        return words.contains { word in
            let normalizedWord = normalizedSearchText(word)
            guard !normalizedWord.isEmpty else { return false }
            if normalizedWord.contains(" ") {
                return normalizedText.contains(normalizedWord)
            }
            return tokens.contains(normalizedWord)
        }
    }

    private func loadModel(
        url: URL,
        label: String,
        logsLifecycle: Bool = true,
        logsDescription: Bool = true
    ) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        let start = Date()
        if logsLifecycle {
            log("load: \(label) \(url.lastPathComponent) compute=CPU")
        }
        let model = try MLModel(contentsOf: url, configuration: config)
        if logsLifecycle {
            let elapsed = format(seconds: Date().timeIntervalSince(start))
            if logsDescription {
                log("load: \(label) \(elapsed) inputs=[\(model.inputNames)] outputs=[\(model.outputNames)]")
            } else {
                log("load: \(label) \(elapsed)")
            }
        }
        return model
    }

    private func makePromptEmbedding(promptFile: PipelinePromptPresetFile, presetIndex: Int) throws -> MLMultiArray {
        guard promptFile.embeddingDtype == "float16", promptFile.embeddingShape.count == 3 else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        let presetCount = promptFile.embeddingShape[0]
        let tokenCount = promptFile.embeddingShape[1]
        let hiddenSize = promptFile.embeddingShape[2]
        guard presetIndex >= 0 && presetIndex < presetCount else {
            throw PipelineSmokeError.invalidPromptAsset
        }

        let singleByteCount = tokenCount * hiddenSize * MemoryLayout<UInt16>.size
        let start = presetIndex * singleByteCount
        let end = start + singleByteCount
        guard promptEmbeddingData.count >= end else {
            throw PipelineSmokeError.invalidPromptAsset
        }

        let array = try makeEmptyArray(shape: [1, tokenCount, hiddenSize])
        try copyBytes(from: promptEmbeddingData, offset: start, count: singleByteCount, to: array)
        return array
    }

    private func makeUncondEmbedding(promptFile: PipelinePromptPresetFile) throws -> MLMultiArray {
        guard promptFile.uncondEmbeddingShape == [1, 77, 768] else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        let array = try makeEmptyArray(shape: promptFile.uncondEmbeddingShape)
            try copyBytes(from: uncondEmbeddingData, offset: 0, count: uncondEmbeddingData.count, to: array)
        return array
    }

    private func generateLCM(
        unetURLs: [URL],
        decoderURL: URL,
        unetLabel: String,
        vaeLabel: String,
        latentShape: [Int],
        decodedShape: [Int],
        promptInput: String,
        presetIndex: Int,
        preset: PipelinePromptPreset,
        seedOption: PipelineSeedOption,
        guidanceScale: Float,
        totalStart: Date
    ) async throws {
        guard let lcmPromptFile, let lcmScheduler else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        guard lcmScheduler.latentShape == latentShape, lcmScheduler.decodedShape == decodedShape else {
            log(
                "lcm scheduler shape mismatch: " +
                "schedulerLatent=\(shapeDescription(lcmScheduler.latentShape)) expectedLatent=\(shapeDescription(latentShape)) " +
                "schedulerDecoded=\(shapeDescription(lcmScheduler.decodedShape)) expectedDecoded=\(shapeDescription(decodedShape))"
            )
            throw PipelineSmokeError.invalidScheduler
        }

        let conditioning = try await makeLCMConditioningEmbedding(
            promptInput: promptInput,
            promptFile: lcmPromptFile,
            presetIndex: presetIndex,
            preset: preset
        )
        let timestepCond = try makeLCMTimestepCond(
            scheduler: lcmScheduler,
            guidanceScale: guidanceScale
        )
        let seedSelection = resolvedLCMSeed(for: preset, seedOption: seedOption)
        var rng = PipelineSeededRandom(seed: seedSelection.seed ^ fnv1a(conditioning.seedKey))
        var latents = makeRandomLatents(shape: latentShape, rng: &rng)
        let chunkSizeSummary = packageSizeSummary(urls: unetURLs)
        log(
            "lcm: guidance=\(format(float: guidanceScale)) " +
            "seed=\(seedSelection.label) latentShape=\(latentShape.map(String.init).joined(separator: "x")) " +
            "initial \(stats(latents).summary)"
        )
        log(
            "lcm chunks: count=\(chunkSizeSummary.count) " +
            "max=\(format(megabytesFromBytes: chunkSizeSummary.maxBytes))@\(chunkSizeSummary.maxIndex) " +
            "total=\(format(megabytesFromBytes: chunkSizeSummary.totalBytes))"
        )

        var stepMetrics: [PipelineMetric] = []
        var allChunkTimings: [PipelineChunkTiming] = []
        for stepIndex in lcmScheduler.steps.indices {
            let step = lcmScheduler.steps[stepIndex]
            let stepNumber = stepIndex + 1
            let stepStart = Date()
            let noiseResult = try await predictLCMNoiseValues(
                unetURLs: unetURLs,
                latents: latents,
                step: step,
                stepNumber: stepNumber,
                embedding: conditioning.embedding,
                timestepCond: timestepCond,
                scheduler: lcmScheduler
            )
            allChunkTimings.append(contentsOf: noiseResult.chunkTimings)
            let noiseStats = try updateLCMLatents(
                latents: &latents,
                noise: noiseResult.values,
                scheduler: lcmScheduler,
                stepIndex: stepIndex,
                rng: &rng
            )
            let elapsed = Date().timeIntervalSince(stepStart)
            let chunkLoadTotal = noiseResult.chunkTimings.reduce(0) { $0 + $1.loadSeconds }
            let chunkPredictTotal = noiseResult.chunkTimings.reduce(0) { $0 + $1.predictSeconds }
            stepMetrics.append(PipelineMetric(label: "LCM \(stepNumber)", value: format(seconds: elapsed)))
            log(
                "lcm step: \(stepNumber)/\(lcmScheduler.steps.count) t=\(step.timestep) " +
                "unet=\(format(seconds: noiseResult.elapsed)) " +
                "chunkLoad=\(format(seconds: chunkLoadTotal)) chunkPredict=\(format(seconds: chunkPredictTotal)) " +
                "noise \(noiseStats.summary) latents \(stats(latents).summary)"
            )
            await Task.yield()
        }

        log("lcm final: latents \(stats(latents).summary)")
        let decodeInput = PipelineDictionaryFeatureProvider(values: [
            "latents": MLFeatureValue(multiArray: try makeLatentArray(values: latents, shape: latentShape))
        ])
        let decoderStart = Date()
        let decodeResult = try await predictWithTemporaryModel(
            url: decoderURL,
            label: "lcm decoder 4-bit",
            values: decodeInput.values
        )
        let decodeOutput = decodeResult.provider
        let decoderElapsed = Date().timeIntervalSince(decoderStart)
        let decoded = try decodeOutput.multiArray(named: "decoded")
        log("lcm decoder: output \(multiArrayDescription(decoded))")
        log(
            "lcm decoder timing: load=\(format(seconds: decodeResult.loadSeconds)) " +
            "predict=\(format(seconds: decodeResult.predictSeconds)) total=\(format(seconds: decoderElapsed))"
        )
        let imageResult = try makeImage(fromDecodedArray: decoded)
        generatedImage = imageResult.image

        let totalElapsed = Date().timeIntervalSince(totalStart)
        let chunkMetrics = chunkTimingMetrics(
            timings: allChunkTimings,
            packageSizeSummary: chunkSizeSummary
        )
        logChunkTimingSummary(allChunkTimings)
        let runID = makeRunID(
            key: conditioning.runKey,
            seedToken: String(seedSelection.seed),
            guidanceScale: guidanceScale
        )
        lastGeneratedPrompt = promptInput
        lastRunID = runID
        lastResolvedSeed = seedSelection.seed
        metrics = stepMetrics + chunkMetrics + [
            PipelineMetric(label: "Pipeline", value: selectedFamily.title),
            PipelineMetric(label: "Run ID", value: runID),
            PipelineMetric(label: "Cond", value: conditioning.source),
            PipelineMetric(label: "Prompt Key", value: conditioning.runKey),
            PipelineMetric(label: "Preset", value: preset.title),
            PipelineMetric(label: "UNet", value: unetLabel),
            PipelineMetric(label: "VAE", value: vaeLabel),
            PipelineMetric(label: "Seed", value: seedSelection.label),
            PipelineMetric(label: "Guidance", value: format(float: guidanceScale)),
            PipelineMetric(label: "Preview", value: selectedPreviewMode.title),
            PipelineMetric(label: "Decoder", value: format(seconds: decoderElapsed)),
            PipelineMetric(label: "Total", value: format(seconds: totalElapsed)),
            PipelineMetric(label: "Final RMS", value: format(float: stats(latents).rms)),
            PipelineMetric(label: "Decoded RMS", value: format(float: imageResult.decodedStats.rms)),
            PipelineMetric(label: "Clipped", value: "\(imageResult.clippedChannels)/\(imageResult.totalChannels)")
        ]
        status = "Done"
        log("seed: mode=\(seedOption.title) resolved=\(seedSelection.seed)")
        log("lcm decode: \(format(seconds: decoderElapsed))")
        log("run: \(runID)")
        log("done: total=\(format(seconds: totalElapsed))")
    }

    private func packageSizeSummary(urls: [URL]) -> PipelinePackageSizeSummary {
        var totalBytes: Int64 = 0
        var maxBytes: Int64 = 0
        var maxIndex = 0

        for (index, url) in urls.enumerated() {
            let bytes = packageSizeBytes(at: url)
            totalBytes += bytes
            if bytes > maxBytes {
                maxBytes = bytes
                maxIndex = index + 1
            }
        }

        return PipelinePackageSizeSummary(
            count: urls.count,
            totalBytes: totalBytes,
            maxBytes: maxBytes,
            maxIndex: maxIndex
        )
    }

    private func packageSizeBytes(at url: URL) -> Int64 {
        let sizeKeys: Set<URLResourceKey> = [.fileSizeKey, .isRegularFileKey]
        if
            let values = try? url.resourceValues(forKeys: sizeKeys),
            values.isRegularFile == true
        {
            return Int64(values.fileSize ?? 0)
        }

        guard
            let enumerator = FileManager.default.enumerator(
                at: url,
                includingPropertiesForKeys: Array(sizeKeys),
                options: [.skipsHiddenFiles]
            )
        else {
            return 0
        }

        var totalBytes: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard
                let values = try? fileURL.resourceValues(forKeys: sizeKeys),
                values.isRegularFile == true
            else {
                continue
            }
            totalBytes += Int64(values.fileSize ?? 0)
        }
        return totalBytes
    }

    private func chunkTimingMetrics(
        timings: [PipelineChunkTiming],
        packageSizeSummary: PipelinePackageSizeSummary
    ) -> [PipelineMetric] {
        var output = [
            PipelineMetric(label: "Chunks", value: "\(packageSizeSummary.count)"),
            PipelineMetric(
                label: "Chunk Max",
                value: "\(format(megabytesFromBytes: packageSizeSummary.maxBytes)) c\(packageSizeSummary.maxIndex)"
            ),
            PipelineMetric(label: "Chunk Total", value: format(megabytesFromBytes: packageSizeSummary.totalBytes))
        ]
        guard !timings.isEmpty else {
            return output
        }

        let loadTotal = timings.reduce(0) { $0 + $1.loadSeconds }
        let predictTotal = timings.reduce(0) { $0 + $1.predictSeconds }
        if let loadMax = timings.max(by: { $0.loadSeconds < $1.loadSeconds }) {
            output.append(
                PipelineMetric(
                    label: "Load Max",
                    value: "\(format(seconds: loadMax.loadSeconds)) s\(loadMax.step)c\(loadMax.chunk)"
                )
            )
        }
        output.append(PipelineMetric(label: "Load Total", value: format(seconds: loadTotal)))

        if let predictMax = timings.max(by: { $0.predictSeconds < $1.predictSeconds }) {
            output.append(
                PipelineMetric(
                    label: "Pred Max",
                    value: "\(format(seconds: predictMax.predictSeconds)) s\(predictMax.step)c\(predictMax.chunk)"
                )
            )
        }
        output.append(PipelineMetric(label: "Pred Total", value: format(seconds: predictTotal)))
        return output
    }

    private func logChunkTimingSummary(_ timings: [PipelineChunkTiming]) {
        guard !timings.isEmpty else {
            log("lcm chunks summary: no timings")
            return
        }

        let loadTotal = timings.reduce(0) { $0 + $1.loadSeconds }
        let predictTotal = timings.reduce(0) { $0 + $1.predictSeconds }
        let loadMax = timings.max(by: { $0.loadSeconds < $1.loadSeconds })
        let predictMax = timings.max(by: { $0.predictSeconds < $1.predictSeconds })
        let loadMaxLabel = loadMax.map {
            "\(format(seconds: $0.loadSeconds))@s\($0.step)c\($0.chunk)"
        } ?? "n/a"
        let predictMaxLabel = predictMax.map {
            "\(format(seconds: $0.predictSeconds))@s\($0.step)c\($0.chunk)"
        } ?? "n/a"
        log(
            "lcm chunks summary: loadTotal=\(format(seconds: loadTotal)) loadMax=\(loadMaxLabel) " +
            "predictTotal=\(format(seconds: predictTotal)) predictMax=\(predictMaxLabel)"
        )
    }

    private func resolvedLCMSeed(
        for preset: PipelinePromptPreset,
        seedOption: PipelineSeedOption
    ) -> (seed: UInt64, label: String) {
        if seedOption.isRandom {
            let seed = UInt64.random(in: 1...UInt64.max)
            return (seed, "Random \(seed)")
        }
        if let seed = seedOption.seed {
            return (seed, seedOption.title)
        }

        let seed = curatedLCMSeed(for: preset.key)
        return (seed, "Curated \(seed)")
    }

    private func curatedLCMSeed(for presetKey: String) -> UInt64 {
        switch presetKey {
        case "cat_mascot":
            return 1
        case "lucky_cat":
            return 1
        case "cat_sticker":
            return 16
        case "round_cat":
            return 26
        case "tabby_icon":
            return 30
        case "white_mascot":
            return 14
        case "cat_plush":
            return 1
        case "cat_badge":
            return 0
        case "white_cat":
            return 10
        case "orange_cat":
            return 30
        case "cat", "cat_face", "sitting_cat":
            return 24
        case "cat_logo":
            return 3
        case "cat_simple", "black_cat":
            return 22
        default:
            return 7
        }
    }

    private func makeLCMPromptEmbedding(
        promptFile: PipelineLCMPresetFile,
        presetIndex: Int
    ) throws -> MLMultiArray {
        guard promptFile.embeddingDtype == "float16", promptFile.embeddingShape.count == 3 else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        let presetCount = promptFile.embeddingShape[0]
        let tokenCount = promptFile.embeddingShape[1]
        let hiddenSize = promptFile.embeddingShape[2]
        guard presetIndex >= 0 && presetIndex < presetCount else {
            throw PipelineSmokeError.invalidPromptAsset
        }

        let singleByteCount = tokenCount * hiddenSize * MemoryLayout<UInt16>.size
        let start = presetIndex * singleByteCount
        let end = start + singleByteCount
        guard lcmPromptEmbeddingData.count >= end else {
            throw PipelineSmokeError.invalidPromptAsset
        }

        let array = try makeEmptyArray(shape: [1, tokenCount, hiddenSize])
        try copyBytes(from: lcmPromptEmbeddingData, offset: start, count: singleByteCount, to: array)
        return array
    }

    private func makeLCMConditioningEmbedding(
        promptInput: String,
        promptFile: PipelineLCMPresetFile,
        presetIndex: Int,
        preset: PipelinePromptPreset
    ) async throws -> PipelineConditioningEmbedding {
        do {
            let conditioningPrompt = expandedTextConditioningPrompt(for: promptInput)
            let result = try await makeTextEncoderPromptEmbedding(prompt: conditioningPrompt)
            purgeCoreMLCache(context: "after text encoder release")
            let runKey = promptRunKey(promptInput)
            log(
                "conditioning: text_encoder prompt=\"\(promptInput)\" " +
                "conditioningPrompt=\"\(conditioningPrompt)\" tokens=\(result.tokenCount) " +
                "load=\(format(seconds: result.loadSeconds)) predict=\(format(seconds: result.predictSeconds))"
            )
            return PipelineConditioningEmbedding(
                embedding: result.embedding,
                source: "Text Encoder",
                runKey: runKey,
                seedKey: runKey
            )
        } catch {
            log(
                "conditioning: text_encoder unavailable \(error.localizedDescription); " +
                "fallback preset=\(preset.key)"
            )
            let embedding = try makeLCMPromptEmbedding(promptFile: promptFile, presetIndex: presetIndex)
            log("conditioning: preset prompt=\"\(preset.prompt)\"")
            return PipelineConditioningEmbedding(
                embedding: embedding,
                source: "Preset",
                runKey: preset.key,
                seedKey: preset.key
            )
        }
    }

    private func makeTextEncoderPromptEmbedding(prompt: String) async throws -> PipelineTextEncoderResult {
        var tokenizer = try loadCLIPTokenizer()
        let tokenization = try tokenizer.tokenize(prompt, maxLength: 77)
        clipTokenizer = tokenizer

        let inputIDs = try makeInputIDsArray(tokenization.ids)
        let input = PipelineDictionaryFeatureProvider(values: [
            "input_ids": MLFeatureValue(multiArray: inputIDs)
        ])
        let url = try bundledURL(
            named: "clip_text_encoder_77",
            extension: "mlmodelc",
            subdirectory: "TextEncoderAssets"
        )
        let loadStart = Date()
        let model = try loadModel(url: url, label: "text encoder")
        let loadElapsed = Date().timeIntervalSince(loadStart)
        let predictStart = Date()
        let output = try await model.prediction(from: input)
        let predictElapsed = Date().timeIntervalSince(predictStart)
        let hiddenStates = try output.multiArray(named: "hidden_states")
        guard hiddenStates.shape.map(\.intValue) == [1, 77, 768] else {
            throw PipelineSmokeError.invalidTextEncoderOutput
        }
        log("text encoder: output \(multiArrayDescription(hiddenStates))")
        let embedding = try cloneTextEncoderEmbedding(hiddenStates)
        return PipelineTextEncoderResult(
            embedding: embedding,
            tokenCount: tokenization.tokenCount,
            loadSeconds: loadElapsed,
            predictSeconds: predictElapsed
        )
    }

    private func loadCLIPTokenizer() throws -> PipelineCLIPTokenizer {
        if let clipTokenizer {
            return clipTokenizer
        }

        let vocabURL = try bundledURL(
            named: "clip_vocab",
            extension: "json",
            subdirectory: "TextEncoderAssets"
        )
        let mergesURL = try bundledURL(
            named: "clip_merges",
            extension: "txt",
            subdirectory: "TextEncoderAssets"
        )
        let tokenizer = try PipelineCLIPTokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
        log("text encoder tokenizer: vocab=\(tokenizer.vocabCount) merges=\(tokenizer.mergeCount)")
        clipTokenizer = tokenizer
        return tokenizer
    }

    private func makeInputIDsArray(_ ids: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        let destination = array.dataPointer.assumingMemoryBound(to: Int32.self)
        for (index, id) in ids.enumerated() {
            destination[index] = id
        }
        return array
    }

    private func cloneTextEncoderEmbedding(_ array: MLMultiArray) throws -> MLMultiArray {
        let values = try floats(from: array)
        let copy = try makeEmptyArray(shape: [1, 77, 768])
        try copyFloatsAsF16(values, to: copy)
        return copy
    }

    private func makeLCMTimestepCond(
        scheduler: PipelineLCMScheduler,
        guidanceScale: Float
    ) throws -> MLMultiArray {
        guard scheduler.timestepCondShape.count == 2, scheduler.timestepCondShape[0] == 1 else {
            throw PipelineSmokeError.invalidPromptAsset
        }
        let embeddingDim = scheduler.timestepCondShape[1]
        let halfDim = embeddingDim / 2
        guard halfDim > 1 else {
            throw PipelineSmokeError.invalidPromptAsset
        }

        let scale = (guidanceScale - 1.0) * 1000.0
        let exponentBase = Float(Darwin.log(10000.0) / Double(halfDim - 1))
        var values: [Float] = []
        values.reserveCapacity(embeddingDim)

        for index in 0..<halfDim {
            let frequency = Float(Darwin.exp(Double(Float(index) * -exponentBase)))
            values.append(Float(Darwin.sin(Double(scale * frequency))))
        }
        for index in 0..<halfDim {
            let frequency = Float(Darwin.exp(Double(Float(index) * -exponentBase)))
            values.append(Float(Darwin.cos(Double(scale * frequency))))
        }
        if embeddingDim % 2 == 1 {
            values.append(0)
        }

        let array = try makeEmptyArray(shape: scheduler.timestepCondShape)
        try copyFloatsAsF16(values, to: array)
        return array
    }

    private func makeDenoiseSchedule(
        scheduler: PipelineScheduler,
        mode: PipelineStepMode
    ) throws -> PipelineDenoiseSchedule {
        let availableStepCount = scheduler.timesteps.count
        guard availableStepCount > 0, scheduler.sigmas.count >= availableStepCount + 1 else {
            throw PipelineSmokeError.invalidScheduler
        }

        let targetStepCount = min(mode.stepCount, availableStepCount)
        if targetStepCount == availableStepCount {
            return PipelineDenoiseSchedule(timesteps: scheduler.timesteps, sigmas: scheduler.sigmas)
        }

        var selectedIndices: [Int] = []
        let denominator = max(targetStepCount - 1, 1)
        for step in 0..<targetStepCount {
            let position = Float(step) * Float(availableStepCount - 1) / Float(denominator)
            let index = Int(position.rounded())
            if selectedIndices.last != index {
                selectedIndices.append(index)
            }
        }
        while selectedIndices.count < targetStepCount {
            let next = min((selectedIndices.last ?? -1) + 1, availableStepCount - 1)
            if selectedIndices.last == next {
                break
            }
            selectedIndices.append(next)
        }

        let timesteps = selectedIndices.map { scheduler.timesteps[$0] }
        var sigmas = selectedIndices.map { scheduler.sigmas[$0] }
        sigmas.append(scheduler.sigmas[availableStepCount])
        return PipelineDenoiseSchedule(timesteps: timesteps, sigmas: sigmas)
    }

    private func makeInitialLatents(
        scheduler: PipelineScheduler,
        presetKey: String,
        seedOption: PipelineSeedOption
    ) throws -> [Float] {
        if let seed = seedOption.seed {
            return makeSeededLatents(scheduler: scheduler, seed: seed, presetKey: presetKey)
        }
        return try loadAssetLatents(scheduler: scheduler)
    }

    private func loadAssetLatents(scheduler: PipelineScheduler) throws -> [Float] {
        let expectedCount = scheduler.latentShape.reduce(1, *)
        let expectedBytes = expectedCount * MemoryLayout<UInt16>.size
        guard latentSeedData.count == expectedBytes else {
            throw PipelineSmokeError.invalidLatentSeed
        }

        return floats(fromF16Data: latentSeedData, count: expectedCount).map { value in
            clamp(value, min: -6, max: 6)
        }
    }

    private func makeSeededLatents(
        scheduler: PipelineScheduler,
        seed: UInt64,
        presetKey: String
    ) -> [Float] {
        var rng = PipelineSeededRandom(seed: seed ^ fnv1a(presetKey))
        return makeRandomLatents(shape: scheduler.latentShape, rng: &rng)
    }

    private func makeRandomLatents(shape: [Int], rng: inout PipelineSeededRandom) -> [Float] {
        let expectedCount = shape.reduce(1, *)
        return (0..<expectedCount).map { _ in clamp(rng.normal(), min: -6, max: 6) }
    }

    private func predictNoiseValues(
        unetModel: MLModel,
        latents: [Float],
        timestep: Int,
        embedding: MLMultiArray,
        scheduler: PipelineScheduler
    ) async throws -> (values: [Float], elapsed: TimeInterval, description: String) {
        let sample = try makeLatentArray(values: latents, shape: scheduler.latentShape)
        let timestepArray = try makeScalarArray(Float(timestep))
        let input = PipelineDictionaryFeatureProvider(values: [
            "encoder_hidden_states": MLFeatureValue(multiArray: embedding),
            "sample": MLFeatureValue(multiArray: sample),
            "timestep": MLFeatureValue(multiArray: timestepArray)
        ])

        let start = Date()
        let output = try await unetModel.prediction(from: input)
        let elapsed = Date().timeIntervalSince(start)
        let noiseArray = try output.multiArray(named: "noise_pred")
        return (try floats(from: noiseArray), elapsed, multiArrayDescription(noiseArray))
    }

    private func predictLCMNoiseValues(
        unetURLs: [URL],
        latents: [Float],
        step: PipelineLCMStep,
        stepNumber: Int,
        embedding: MLMultiArray,
        timestepCond: MLMultiArray,
        scheduler: PipelineLCMScheduler
    ) async throws -> (values: [Float], elapsed: TimeInterval, description: String, chunkTimings: [PipelineChunkTiming]) {
        let sample = try makeLatentArray(values: latents, shape: scheduler.latentShape)
        let timestepArray = try makeScalarArray(Float(step.timestep))
        let input = PipelineDictionaryFeatureProvider(values: [
            "encoder_hidden_states": MLFeatureValue(multiArray: embedding),
            "sample": MLFeatureValue(multiArray: sample),
            "timestep": MLFeatureValue(multiArray: timestepArray),
            "timestep_cond": MLFeatureValue(multiArray: timestepCond)
        ])

        let start = Date()
        let result = try await predictThroughChunkURLs(
            urls: unetURLs,
            input: input,
            stepNumber: stepNumber
        )
        let elapsed = Date().timeIntervalSince(start)
        let noiseArray = try result.provider.multiArray(named: "noise_pred")
        return (
            try floats(from: noiseArray),
            elapsed,
            multiArrayDescription(noiseArray),
            result.timings
        )
    }

    private func predictThroughChunkURLs(
        urls: [URL],
        input: PipelineDictionaryFeatureProvider,
        stepNumber: Int
    ) async throws -> (provider: MLFeatureProvider, timings: [PipelineChunkTiming]) {
        guard !urls.isEmpty else {
            throw PipelineSmokeError.missingOutput("noise_pred")
        }

        var inputValues = input.values
        var outputValues: [String: MLFeatureValue] = [:]
        var timings: [PipelineChunkTiming] = []

        for (index, url) in urls.enumerated() {
            let result = try await predictWithTemporaryModel(
                url: url,
                label: "lcm unet chunk \(index + 1)",
                values: inputValues,
                logsModelLifecycle: logsDetailedLCMChunks
            )
            let timing = PipelineChunkTiming(
                step: stepNumber,
                chunk: index + 1,
                loadSeconds: result.loadSeconds,
                predictSeconds: result.predictSeconds
            )
            timings.append(timing)
            if logsDetailedLCMChunks {
                log(
                    "lcm chunk: step=\(stepNumber) chunk=\(index + 1)/\(urls.count) " +
                    "load=\(format(seconds: timing.loadSeconds)) predict=\(format(seconds: timing.predictSeconds))"
                )
            }
            outputValues = result.provider.values
            for (name, value) in outputValues {
                inputValues[name] = value
            }
            await Task.yield()
        }

        return (PipelineDictionaryFeatureProvider(values: outputValues), timings)
    }

    private func predictWithTemporaryModel(
        url: URL,
        label: String,
        values: [String: MLFeatureValue],
        logsModelLifecycle: Bool = true
    ) async throws -> PipelineTemporaryPredictionResult {
        let loadStart = Date()
        let model = try loadModel(
            url: url,
            label: label,
            logsLifecycle: logsModelLifecycle,
            logsDescription: logsModelLifecycle
        )
        let loadElapsed = Date().timeIntervalSince(loadStart)
        let modelInputNames = Set(model.modelDescription.inputDescriptionsByName.keys)
        let filteredValues = values.filter { modelInputNames.contains($0.key) }
        let predictStart = Date()
        let output = try await model.prediction(from: PipelineDictionaryFeatureProvider(values: filteredValues))
        let predictElapsed = Date().timeIntervalSince(predictStart)
        return PipelineTemporaryPredictionResult(
            provider: PipelineDictionaryFeatureProvider(values: output.featureValueDictionary),
            loadSeconds: loadElapsed,
            predictSeconds: predictElapsed
        )
    }

    private func updateLatents(
        latents: inout [Float],
        previousModelOutput: inout [Float]?,
        uncondNoise: [Float],
        condNoise: [Float],
        guidanceScale: Float,
        schedule: PipelineDenoiseSchedule,
        stepIndex: Int
    ) throws -> PipelineTensorStats {
        guard latents.count == uncondNoise.count, latents.count == condNoise.count else {
            throw PipelineSmokeError.invalidArrayShape
        }

        var guidedNoise = [Float]()
        guidedNoise.reserveCapacity(latents.count)
        var modelOutput = [Float]()
        modelOutput.reserveCapacity(latents.count)

        let sigmaSRaw = schedule.sigmas[stepIndex]
        let sigmaTRaw = schedule.sigmas[stepIndex + 1]
        let (alphaS, sigmaS) = alphaSigma(from: sigmaSRaw)
        let (alphaT, sigmaT) = alphaSigma(from: sigmaTRaw)
        let lambdaS = lambda(alpha: alphaS, sigma: sigmaS)
        let lambdaT = lambda(alpha: alphaT, sigma: sigmaT)
        let h = lambdaT - lambdaS
        let expNegH = expNeg(h)

        for index in latents.indices {
            let noise = uncondNoise[index] + guidanceScale * (condNoise[index] - uncondNoise[index])
            guidedNoise.append(noise)
            modelOutput.append((latents[index] - sigmaS * noise) / max(alphaS, 0.000001))
        }

        let isFinalZero = stepIndex == schedule.stepCount - 1 && sigmaTRaw == 0
        let useFirstOrder = previousModelOutput == nil || isFinalZero
        if useFirstOrder {
            for index in latents.indices {
                latents[index] = (sigmaT / sigmaS) * latents[index] - alphaT * (expNegH - 1.0) * modelOutput[index]
            }
        } else if let previous = previousModelOutput {
            let sigmaS1Raw = schedule.sigmas[stepIndex - 1]
            let (alphaS1, sigmaS1) = alphaSigma(from: sigmaS1Raw)
            let lambdaS1 = lambda(alpha: alphaS1, sigma: sigmaS1)
            let h0 = lambdaS - lambdaS1
            let r0 = h0 / h
            for index in latents.indices {
                let d1 = (modelOutput[index] - previous[index]) / r0
                let coeff = alphaT * (expNegH - 1.0)
                latents[index] = (sigmaT / sigmaS) * latents[index] - coeff * modelOutput[index] - 0.5 * coeff * d1
            }
        }

        previousModelOutput = modelOutput
        return stats(guidedNoise)
    }

    private func updateLCMLatents(
        latents: inout [Float],
        noise: [Float],
        scheduler: PipelineLCMScheduler,
        stepIndex: Int,
        rng: inout PipelineSeededRandom
    ) throws -> PipelineTensorStats {
        guard latents.count == noise.count, scheduler.steps.indices.contains(stepIndex) else {
            throw PipelineSmokeError.invalidArrayShape
        }
        guard scheduler.predictionType == "epsilon" else {
            throw PipelineSmokeError.invalidScheduler
        }

        let step = scheduler.steps[stepIndex]
        let isFinalStep = stepIndex == scheduler.steps.count - 1
        for index in latents.indices {
            let predictedOriginal = (latents[index] - step.sqrtBeta * noise[index]) / max(step.sqrtAlpha, 0.000001)
            let denoised = step.cOut * predictedOriginal + step.cSkip * latents[index]
            if isFinalStep {
                latents[index] = denoised
            } else {
                latents[index] = step.sqrtAlphaPrev * denoised + step.sqrtBetaPrev * clamp(rng.normal(), min: -6, max: 6)
            }
        }
        return stats(noise)
    }

    private func makeLatentArray(values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try makeEmptyArray(shape: shape)
        try copyFloatsAsF16(values, to: array)
        return array
    }

    private func makeScalarArray(_ value: Float) throws -> MLMultiArray {
        let array = try makeEmptyArray(shape: [1])
        try copyFloatsAsF16([value], to: array)
        return array
    }

    private func makeEmptyArray(shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    }

    private func copyBytes(from data: Data, offset: Int, count: Int, to array: MLMultiArray) throws {
        guard array.dataType == .float16, array.count * MemoryLayout<UInt16>.size == count else {
            throw PipelineSmokeError.invalidArrayShape
        }
        try data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else {
                throw PipelineSmokeError.invalidPromptAsset
            }
            memcpy(array.dataPointer, base.advanced(by: offset), count)
        }
    }

    private func copyFloatsAsF16(_ values: [Float], to array: MLMultiArray) throws {
        guard array.dataType == .float16, array.count == values.count else {
            throw PipelineSmokeError.invalidArrayShape
        }

        let destination = array.dataPointer.assumingMemoryBound(to: UInt16.self)
        for (index, value) in values.enumerated() {
            destination[index] = Float16(clamp(value, min: -65504, max: 65504)).bitPattern
        }
    }

    private func floats(from array: MLMultiArray) throws -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var output = [Float]()
        output.reserveCapacity(array.count)

        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                Float(Float16(bitPattern: pointer[$0]))
            }
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                pointer[$0]
            }
        case .double:
            let pointer = array.dataPointer.assumingMemoryBound(to: Double.self)
            appendLogicalValues(shape: shape, strides: strides, dimension: 0, offset: 0, into: &output) {
                Float(pointer[$0])
            }
        default:
            throw PipelineSmokeError.unsupportedArrayType
        }

        return output
    }

    private func floats(fromF16Data data: Data, count: Int) -> [Float] {
        data.withUnsafeBytes { rawBuffer in
            (0..<count).map { index in
                let low = UInt16(rawBuffer[index * 2])
                let high = UInt16(rawBuffer[index * 2 + 1]) << 8
                return Float(Float16(bitPattern: low | high))
            }
        }
    }

    private func makeImage(fromDecodedArray array: MLMultiArray) throws -> PipelineImageResult {
        let values = try floats(from: array)
        let shape = array.shape.map(\.intValue)
        guard shape.count == 4, shape[0] == 1, shape[1] >= 3 else {
            throw PipelineSmokeError.invalidDecodedShape
        }

        let height = shape[2]
        let width = shape[3]
        let totalChannels = width * height * 3
        guard values.count >= totalChannels else {
            throw PipelineSmokeError.invalidDecodedShape
        }

        let decodedStats = stats(Array(values.prefix(totalChannels)))
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        var clippedChannels = 0

        for y in 0..<height {
            for x in 0..<width {
                let pixel = (y * width + x) * 4
                let red = value(atChannel: 0, x: x, y: y, values: values, width: width, height: height)
                let green = value(atChannel: 1, x: x, y: y, values: values, width: width, height: height)
                let blue = value(atChannel: 2, x: x, y: y, values: values, width: width, height: height)
                clippedChannels += isOutsideImageRange(red) ? 1 : 0
                clippedChannels += isOutsideImageRange(green) ? 1 : 0
                clippedChannels += isOutsideImageRange(blue) ? 1 : 0
                rgba[pixel + 0] = colorByte(red)
                rgba[pixel + 1] = colorByte(green)
                rgba[pixel + 2] = colorByte(blue)
                rgba[pixel + 3] = 255
            }
        }
        log("decoded: \(decodedStats.summary) clipped=\(clippedChannels)/\(totalChannels)")

        let preview = makePreviewRGBA(
            from: rgba,
            width: width,
            height: height,
            mode: selectedPreviewMode
        )
        if selectedPreviewMode.usesImagePostprocess {
            log("preview: \(selectedPreviewMode.title) \(width)x\(height)->\(preview.width)x\(preview.height)")
        }

        let data = Data(preview.rgba) as CFData
        guard
            let provider = CGDataProvider(data: data),
            let image = CGImage(
                width: preview.width,
                height: preview.height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: preview.width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: selectedPreviewMode.shouldInterpolateCGImage,
                intent: .defaultIntent
            )
        else {
            throw PipelineSmokeError.imageCreationFailed
        }

        return PipelineImageResult(
            image: image,
            decodedStats: decodedStats,
            clippedChannels: clippedChannels,
            totalChannels: totalChannels
        )
    }

    private func makePreviewRGBA(
        from rgba: [UInt8],
        width: Int,
        height: Int,
        mode: PipelinePreviewMode
    ) -> (rgba: [UInt8], width: Int, height: Int) {
        switch mode {
        case .smooth, .crisp:
            return (rgba, width, height)
        case .sharp2x:
            let upscaled = upscaleRGBA2xBicubic(rgba, width: width, height: height)
            let sharpened = unsharpRGBA(upscaled.rgba, width: upscaled.width, height: upscaled.height, amount: 0.45)
            return (sharpened, upscaled.width, upscaled.height)
        }
    }

    private func upscaleRGBA2xBicubic(
        _ rgba: [UInt8],
        width: Int,
        height: Int
    ) -> (rgba: [UInt8], width: Int, height: Int) {
        let outputWidth = width * 2
        let outputHeight = height * 2
        var output = [UInt8](repeating: 255, count: outputWidth * outputHeight * 4)

        for y in 0..<outputHeight {
            let sourceY = Float(y) / 2.0
            for x in 0..<outputWidth {
                let sourceX = Float(x) / 2.0
                let outputPixel = (y * outputWidth + x) * 4

                for channel in 0..<3 {
                    output[outputPixel + channel] = byte(
                        bicubicSampleRGBA(
                            rgba,
                            width: width,
                            height: height,
                            x: sourceX,
                            y: sourceY,
                            channel: channel
                        )
                    )
                }
                output[outputPixel + 3] = 255
            }
        }

        return (output, outputWidth, outputHeight)
    }

    private func bicubicSampleRGBA(
        _ rgba: [UInt8],
        width: Int,
        height: Int,
        x: Float,
        y: Float,
        channel: Int
    ) -> Float {
        let baseX = Int(Darwin.floor(Double(x)))
        let baseY = Int(Darwin.floor(Double(y)))
        let fractionX = x - Float(baseX)
        let fractionY = y - Float(baseY)
        var rows: [Float] = []
        rows.reserveCapacity(4)

        for yy in -1...2 {
            var samples: [Float] = []
            samples.reserveCapacity(4)
            let sourceY = clampInt(baseY + yy, min: 0, max: height - 1)
            for xx in -1...2 {
                let sourceX = clampInt(baseX + xx, min: 0, max: width - 1)
                samples.append(Float(rgba[(sourceY * width + sourceX) * 4 + channel]))
            }
            rows.append(catmullRom(samples[0], samples[1], samples[2], samples[3], fractionX))
        }

        return catmullRom(rows[0], rows[1], rows[2], rows[3], fractionY)
    }

    private func catmullRom(_ a: Float, _ b: Float, _ c: Float, _ d: Float, _ t: Float) -> Float {
        let t2 = t * t
        let t3 = t2 * t
        return 0.5 * (
            2.0 * b +
            (-a + c) * t +
            (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 +
            (-a + 3.0 * b - 3.0 * c + d) * t3
        )
    }

    private func unsharpRGBA(
        _ rgba: [UInt8],
        width: Int,
        height: Int,
        amount: Float
    ) -> [UInt8] {
        var output = rgba
        for y in 0..<height {
            for x in 0..<width {
                let pixel = (y * width + x) * 4
                for channel in 0..<3 {
                    var sum: Int = 0
                    var count: Int = 0
                    for dy in -1...1 {
                        let yy = y + dy
                        guard yy >= 0, yy < height else { continue }
                        for dx in -1...1 {
                            let xx = x + dx
                            guard xx >= 0, xx < width else { continue }
                            sum += Int(rgba[(yy * width + xx) * 4 + channel])
                            count += 1
                        }
                    }
                    let original = Float(rgba[pixel + channel])
                    let blurred = Float(sum) / Float(max(1, count))
                    output[pixel + channel] = byte(original + (original - blurred) * amount)
                }
                output[pixel + 3] = 255
            }
        }
        return output
    }

    private func value(
        atChannel channel: Int,
        x: Int,
        y: Int,
        values: [Float],
        width: Int,
        height: Int
    ) -> Float {
        let index = channel * width * height + y * width + x
        guard values.indices.contains(index) else { return 0 }
        return values[index]
    }

    private func isOutsideImageRange(_ value: Float) -> Bool {
        value < -1.0 || value > 1.0
    }

    private func colorByte(_ value: Float) -> UInt8 {
        let normalized = clamp((value / 2 + 0.5) * 255, min: 0, max: 255)
        return UInt8(normalized.rounded())
    }

    private func byte(_ value: Float) -> UInt8 {
        UInt8(clamp(value, min: 0, max: 255).rounded())
    }

    private func clampInt(_ value: Int, min lowerBound: Int, max upperBound: Int) -> Int {
        Swift.max(lowerBound, Swift.min(upperBound, value))
    }

    private func alphaSigma(from rawSigma: Float) -> (alpha: Float, sigma: Float) {
        let alpha = 1.0 / sqrt(rawSigma * rawSigma + 1.0)
        return (alpha, rawSigma * alpha)
    }

    private func lambda(alpha: Float, sigma: Float) -> Float {
        if sigma == 0 {
            return .infinity
        }
        return Float(Darwin.log(Double(alpha)) - Darwin.log(Double(sigma)))
    }

    private func expNeg(_ value: Float) -> Float {
        value.isInfinite ? 0.0 : exp(-value)
    }

    private func stats(_ values: [Float]) -> PipelineTensorStats {
        var minValue = Float.infinity
        var maxValue = -Float.infinity
        var sum: Float = 0
        var sumSquares: Float = 0
        var count: Float = 0

        for value in values where value.isFinite {
            minValue = Swift.min(minValue, value)
            maxValue = Swift.max(maxValue, value)
            sum += value
            sumSquares += value * value
            count += 1
        }

        guard count > 0 else {
            return PipelineTensorStats(min: .nan, max: .nan, mean: .nan, rms: .nan)
        }
        return PipelineTensorStats(
            min: minValue,
            max: maxValue,
            mean: sum / count,
            rms: sqrt(sumSquares / count)
        )
    }

    private func appendLogicalValues(
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

    private func scheduleSummary(_ timesteps: [Int]) -> String {
        if timesteps.count <= 8 {
            return timesteps.map(String.init).joined(separator: ",")
        }
        let head = timesteps.prefix(4).map(String.init).joined(separator: ",")
        let tail = timesteps.suffix(3).map(String.init).joined(separator: ",")
        return "\(head),...,\(tail)"
    }

    private func shapeDescription(_ shape: [Int]) -> String {
        shape.map(String.init).joined(separator: "x")
    }

    private func bundledURL(named name: String, extension pathExtension: String, subdirectory: String? = nil) throws -> URL {
        if let url = Bundle.main.url(forResource: name, withExtension: pathExtension, subdirectory: subdirectory) {
            return url
        }

        guard let resourceURL = Bundle.main.resourceURL else {
            throw PipelineSmokeError.missingResource("\(name).\(pathExtension)")
        }

        let targetFileName = "\(name).\(pathExtension)"
        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL,
            includingPropertiesForKeys: [.nameKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw PipelineSmokeError.missingResource(targetFileName)
        }

        for case let url as URL in enumerator where url.lastPathComponent == targetFileName {
            return url
        }

        throw PipelineSmokeError.missingResource(targetFileName)
    }

    private func log(_ message: String) {
        let line = "\(timestamp()) \(message)"
        print("[WatchPipeline] \(line)")
        logLines.insert(PipelineLogLine(text: line), at: 0)
        if logLines.count > 240 {
            logLines.removeLast(logLines.count - 240)
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

    private func format(megabytesFromBytes bytes: Int64) -> String {
        String(format: "%.1fMB", Double(bytes) / 1_048_576.0)
    }

    private func format(float: Float) -> String {
        String(format: "%.3f", float)
    }

    private func makeRunID(
        preset: PipelinePromptPreset,
        seedToken: String,
        guidanceScale: Float
    ) -> String {
        makeRunID(
            key: preset.key,
            seedToken: seedToken,
            guidanceScale: guidanceScale
        )
    }

    private func makeRunID(
        key: String,
        seedToken: String,
        guidanceScale: Float
    ) -> String {
        "\(key)-s\(seedToken)-g\(compact(float: guidanceScale))-\(selectedPreviewMode.rawValue)"
    }

    private func compact(float: Float) -> String {
        if abs(float.rounded() - float) < 0.0001 {
            return String(Int(float.rounded()))
        }
        return String(format: "%.1f", float).replacingOccurrences(of: ".", with: "p")
    }

    private func fnv1a(_ text: String) -> UInt64 {
        var value: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            value ^= UInt64(byte)
            value &*= 0x100000001b3
        }
        return value
    }

    private func multiArrayDescription(_ array: MLMultiArray) -> String {
        "dtype=\(array.dataType.rawValue) shape=\(array.shape.map(\.intValue)) strides=\(array.strides.map(\.intValue))"
    }

    private func clamp(_ value: Float, min lowerBound: Float, max upperBound: Float) -> Float {
        Swift.max(lowerBound, Swift.min(upperBound, value.isFinite ? value : 0))
    }
}

struct PipelineSmokeView: View {
    @StateObject private var viewModel = PipelineSmokeViewModel()

    var body: some View {
        NavigationStack {
            List {
                Section {
                    TextField("Prompt", text: $viewModel.promptText)
                        .disabled(viewModel.isGenerating)

                    Button(viewModel.generateButtonTitle) {
                        Task {
                            await viewModel.generate()
                        }
                    }
                    .disabled(!viewModel.canGenerate)
                }

                if let image = viewModel.generatedImage {
                    Section {
                        Image(decorative: image, scale: 1, orientation: .up)
                            .resizable()
                            .interpolation(viewModel.selectedPreviewMode.interpolation)
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                    }
                }
            }
            .navigationTitle(viewModel.status)
            .onAppear {
                viewModel.prepare()
            }
        }
    }
}

private struct PipelineSeededRandom {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func normal() -> Float {
        let u1 = max(uniform(), 0.000001)
        let u2 = uniform()
        let radius = sqrt(-2.0 * Darwin.log(Double(u1)))
        let angle = 2.0 * Double.pi * Double(u2)
        return Float(radius * cos(angle))
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

private struct PipelineBPEPair: Hashable {
    let first: String
    let second: String
}

private struct PipelineCLIPTokenizer {
    private let vocab: [String: Int]
    private let bpeRanks: [PipelineBPEPair: Int]
    private let startTokenID: Int32
    private let endTokenID: Int32
    private let byteEncoder: [String]
    private var cache: [String: String] = [:]

    var vocabCount: Int { vocab.count }
    var mergeCount: Int { bpeRanks.count }

    init(vocabURL: URL, mergesURL: URL) throws {
        vocab = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: vocabURL))
        guard
            let startID = vocab["<|startoftext|>"],
            let endID = vocab["<|endoftext|>"]
        else {
            throw PipelineSmokeError.invalidTokenizerAsset
        }
        startTokenID = Int32(startID)
        endTokenID = Int32(endID)
        byteEncoder = PipelineCLIPTokenizer.makeByteEncoder()

        var ranks: [PipelineBPEPair: Int] = [:]
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var rank = 0
        for rawLine in mergesText.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            ranks[PipelineBPEPair(first: String(parts[0]), second: String(parts[1]))] = rank
            rank += 1
        }
        guard !ranks.isEmpty else {
            throw PipelineSmokeError.invalidTokenizerAsset
        }
        bpeRanks = ranks
    }

    mutating func tokenize(_ text: String, maxLength: Int) throws -> PipelineTokenization {
        let pieces = try tokenizeToPieces(text)
        var ids = [startTokenID]
        ids.reserveCapacity(maxLength)
        for piece in pieces {
            guard let id = vocab[piece] else {
                throw PipelineSmokeError.invalidTokenizerAsset
            }
            ids.append(Int32(id))
        }
        ids.append(endTokenID)

        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
            ids[maxLength - 1] = endTokenID
        }
        let tokenCount = ids.count
        while ids.count < maxLength {
            ids.append(endTokenID)
        }

        return PipelineTokenization(ids: ids, tokenCount: tokenCount)
    }

    private mutating func tokenizeToPieces(_ text: String) throws -> [String] {
        var pieces: [String] = []
        for token in splitForCLIP(text) {
            let encoded = encodeBytes(token)
            let bpeText = bpe(encoded)
            pieces.append(contentsOf: bpeText.split(separator: " ").map(String.init))
        }
        return pieces
    }

    private func splitForCLIP(_ text: String) -> [String] {
        enum TokenKind: Equatable {
            case letters
            case digits
            case symbols
        }

        let normalized = normalize(text)
        var tokens: [String] = []
        var scalars: [UnicodeScalar] = []
        var currentKind: TokenKind?

        func flush() {
            guard !scalars.isEmpty else { return }
            tokens.append(String(String.UnicodeScalarView(scalars)))
            scalars.removeAll(keepingCapacity: true)
            currentKind = nil
        }

        for scalar in normalized.unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                flush()
                continue
            }

            let nextKind: TokenKind
            if CharacterSet.letters.contains(scalar) {
                nextKind = .letters
            } else if CharacterSet.decimalDigits.contains(scalar) {
                nextKind = .digits
            } else {
                nextKind = .symbols
            }

            if let currentKind, currentKind != nextKind {
                flush()
            }
            currentKind = nextKind
            scalars.append(scalar)
        }
        flush()
        return tokens
    }

    private func normalize(_ text: String) -> String {
        var scalars: [UnicodeScalar] = []
        var previousWasSpace = true

        for scalar in text.lowercased().unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                if !previousWasSpace {
                    scalars.append(" ")
                    previousWasSpace = true
                }
            } else {
                scalars.append(scalar)
                previousWasSpace = false
            }
        }

        return String(String.UnicodeScalarView(scalars)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func encodeBytes(_ token: String) -> String {
        var output = ""
        for byte in token.utf8 {
            output += byteEncoder[Int(byte)]
        }
        return output
    }

    private mutating func bpe(_ token: String) -> String {
        if let cached = cache[token] {
            return cached
        }

        var word = token.map { String($0) }
        guard !word.isEmpty else {
            cache[token] = ""
            return ""
        }
        word[word.count - 1] += "</w>"

        while word.count > 1 {
            let pairs = getPairs(word)
            guard
                let best = pairs.min(by: { rank($0) < rank($1) }),
                bpeRanks[best] != nil
            else {
                break
            }

            var merged: [String] = []
            var index = 0
            while index < word.count {
                if
                    index < word.count - 1,
                    word[index] == best.first,
                    word[index + 1] == best.second
                {
                    merged.append(word[index] + word[index + 1])
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }
            word = merged
        }

        let result = word.joined(separator: " ")
        cache[token] = result
        return result
    }

    private func getPairs(_ word: [String]) -> Set<PipelineBPEPair> {
        guard word.count > 1 else { return [] }
        var pairs = Set<PipelineBPEPair>()
        for index in 0..<(word.count - 1) {
            pairs.insert(PipelineBPEPair(first: word[index], second: word[index + 1]))
        }
        return pairs
    }

    private func rank(_ pair: PipelineBPEPair) -> Int {
        bpeRanks[pair] ?? Int.max
    }

    private static func makeByteEncoder() -> [String] {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var codePoints = bytes
        var next = 0

        for byte in 0...255 where !bytes.contains(byte) {
            bytes.append(byte)
            codePoints.append(256 + next)
            next += 1
        }

        var output = Array(repeating: "", count: 256)
        for (byte, codePoint) in zip(bytes, codePoints) {
            if let scalar = UnicodeScalar(codePoint) {
                output[byte] = String(scalar)
            }
        }
        return output
    }
}

private final class PipelineDictionaryFeatureProvider: MLFeatureProvider {
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

private extension MLFeatureProvider {
    var featureValueDictionary: [String: MLFeatureValue] {
        featureNames.reduce(into: [String: MLFeatureValue]()) { values, name in
            values[name] = featureValue(for: name)
        }
    }

    func multiArray(named preferredName: String) throws -> MLMultiArray {
        if let value = featureValue(for: preferredName)?.multiArrayValue {
            return value
        }

        for name in featureNames.sorted() {
            if let value = featureValue(for: name)?.multiArrayValue {
                return value
            }
        }

        throw PipelineSmokeError.missingOutput(preferredName)
    }
}

private extension MLModel {
    var inputNames: String {
        modelDescription.inputDescriptionsByName.keys.sorted().joined(separator: ",")
    }

    var outputNames: String {
        modelDescription.outputDescriptionsByName.keys.sorted().joined(separator: ",")
    }
}

private enum PipelineSmokeError: LocalizedError {
    case imageCreationFailed
    case invalidArrayShape
    case invalidDecodedShape
    case invalidLatentSeed
    case invalidPromptAsset
    case invalidScheduler
    case invalidTextEncoderOutput
    case invalidTokenizerAsset
    case missingOutput(String)
    case missingResource(String)
    case unsupportedArrayType

    var errorDescription: String? {
        switch self {
        case .imageCreationFailed:
            return "failed to create CGImage"
        case .invalidArrayShape:
            return "invalid multi-array shape"
        case .invalidDecodedShape:
            return "invalid decoded image shape"
        case .invalidLatentSeed:
            return "invalid latent seed asset"
        case .invalidPromptAsset:
            return "invalid prompt asset"
        case .invalidScheduler:
            return "invalid scheduler asset"
        case .invalidTextEncoderOutput:
            return "invalid text encoder output"
        case .invalidTokenizerAsset:
            return "invalid tokenizer asset"
        case .missingOutput(let name):
            return "missing model output: \(name)"
        case .missingResource(let name):
            return "missing bundled resource: \(name)"
        case .unsupportedArrayType:
            return "unsupported MLMultiArray data type"
        }
    }
}
