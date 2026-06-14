import CoreML
import Foundation
import StableDiffusion
import SwiftUI
import UIKit

enum ResolutionPreset: Int, CaseIterable {
    case p256 = 256
    case p512 = 512

    var pixelSize: Int { rawValue }
    var label: String { "\(rawValue)x\(rawValue)" }
    var bundledFolderName: String { "\(rawValue)" }
    var displayName: String {
        switch self {
        case .p256:
            return "高速"
        case .p512:
            return "標準"
        }
    }
}

enum SchedulerOption: String, CaseIterable, Identifiable {
    case pndm
    case dpmSolverMultistep

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .pndm:
            return "安定重視"
        case .dpmSolverMultistep:
            return "高速・高品質"
        }
    }

    var detailText: String {
        switch self {
        case .pndm:
            return "少し時間はかかりますが、変化が穏やかです。"
        case .dpmSolverMultistep:
            return "通常はこちらで問題ありません。速度と品質のバランスが良い方式です。"
        }
    }

    var stableDiffusionScheduler: StableDiffusionScheduler {
        switch self {
        case .pndm:
            return .pndmScheduler
        case .dpmSolverMultistep:
            return .dpmSolverMultistepScheduler
        }
    }
}

enum SeedMode: String, CaseIterable, Identifiable {
    case random
    case manual

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .random:
            return "毎回ランダム"
        case .manual:
            return "固定値を使う"
        }
    }

    var detailText: String {
        switch self {
        case .random:
            return "生成するたびに別のシード値を自動で選びます。"
        case .manual:
            return "同じ条件で似た結果を再現したいときに使います。"
        }
    }
}

enum NegativePromptPreset: String, CaseIterable, Identifiable {
    case standard
    case photo
    case illustration
    case textBlock

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .standard:
            return "標準"
        case .photo:
            return "写真向け"
        case .illustration:
            return "イラスト向け"
        case .textBlock:
            return "文字・透かし抑制"
        }
    }

    var detailText: String {
        switch self {
        case .standard:
            return "迷ったらこれで十分です。"
        case .photo:
            return "写真風で崩れや眠さを抑えたいとき向けです。"
        case .illustration:
            return "輪郭や手の破綻を少し抑えたいとき向けです。"
        case .textBlock:
            return "文字、ロゴ、透かしが出やすいとき向けです。"
        }
    }

    var promptText: String {
        switch self {
        case .standard:
            return "low quality, blurry, distorted, deformed, bad anatomy"
        case .photo:
            return "low quality, blurry, overexposed, underexposed, distorted, deformed, bad anatomy, plastic skin"
        case .illustration:
            return "low quality, blurry, bad anatomy, extra fingers, extra limbs, deformed hands, messy composition"
        case .textBlock:
            return "text, watermark, logo, signature, caption, low quality, blurry"
        }
    }
}

private enum SD15RuntimeResources {
    struct CompiledModel: Sendable {
        let compiledName: String
        let displayName: String
    }

    static let modelFamilyFolder = "sd15"
    static let compiledModels: [CompiledModel] = [
        .init(compiledName: "TextEncoder.mlmodelc", displayName: "Text Encoder"),
        .init(compiledName: "Unet.mlmodelc", displayName: "UNet"),
        .init(compiledName: "VAEDecoder.mlmodelc", displayName: "VAE Decoder"),
    ]
    static let tokenizerFiles = ["vocab.json", "merges.txt"]
}

private final class RuntimeResourceLocator: @unchecked Sendable {
    private let fileManager = FileManager.default

    func availableResolutions() throws -> [ResolutionPreset] {
        try ResolutionPreset.allCases.filter { resolution in
            try bundledCompiledResourcesDirectory(for: resolution) != nil
        }
    }

    func resolveBestResources(
        preferred preferredResolution: ResolutionPreset,
        publish: @escaping @Sendable (String) -> Void
    ) throws -> (resolution: ResolutionPreset, url: URL) {
        let resolutionOrder = [preferredResolution] + ResolutionPreset.allCases.filter { $0 != preferredResolution }

        for resolution in resolutionOrder {
            if let url = try bundledCompiledResourcesDirectory(for: resolution) {
                publish("同梱済みモデルを確認しました。")
                return (resolution, url)
            }
        }

        throw SDXLGeneratorViewModel.GeneratorError.missingResources("BundledResources/\(SD15RuntimeResources.modelFamilyFolder)")
    }

    private func bundledCompiledResourcesDirectory(for resolution: ResolutionPreset) throws -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw SDXLGeneratorViewModel.GeneratorError.missingResources("Bundle.main.resourceURL")
        }

        let url = resourceURL
            .appending(path: "BundledResources", directoryHint: .isDirectory)
            .appending(path: SD15RuntimeResources.modelFamilyFolder, directoryHint: .isDirectory)
            .appending(path: resolution.bundledFolderName, directoryHint: .isDirectory)
            .appending(path: "Resources", directoryHint: .isDirectory)

        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }

        for model in SD15RuntimeResources.compiledModels {
            let modelURL = url.appending(path: model.compiledName, directoryHint: .isDirectory)
            let weightsURL = modelURL
                .appending(path: "weights", directoryHint: .isDirectory)
                .appending(path: "weight.bin")
            guard fileManager.fileExists(atPath: modelURL.path),
                  fileManager.fileExists(atPath: weightsURL.path) else {
                return nil
            }
        }

        for tokenFile in SD15RuntimeResources.tokenizerFiles {
            guard fileManager.fileExists(atPath: url.appending(path: tokenFile).path) else {
                return nil
            }
        }

        return url
    }
}

final class SDXLPipelineCache: @unchecked Sendable {
    private let queue = DispatchQueue(label: "SDXLCoreMLTest.pipeline-cache")
    private var cachedPipeline: StableDiffusionPipeline?
    private var cachedResourcesPath: String?
    private var cachedReduceMemory: Bool?

    func preloadPipeline(
        resourcesURL: URL,
        reduceMemory: Bool,
        publish: @escaping @Sendable (String) -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    if self.cachedResourcesPath == resourcesURL.path,
                       self.cachedReduceMemory == reduceMemory,
                       self.cachedPipeline != nil {
                        publish("モデルはすでに読み込み済みです。")
                        continuation.resume(returning: ())
                        return
                    }

                    let startTime = Date()
                    self.cachedPipeline?.unloadResources()
                    self.cachedPipeline = nil
                    self.cachedResourcesPath = nil
                    self.cachedReduceMemory = nil

                    func elapsedString(since start: Date) -> String {
                        String(format: "%.1f秒", Date().timeIntervalSince(start))
                    }

                    func update(_ message: String) {
                        print("[SD15] \(message)")
                        publish(message)
                    }

                    update("SD 1.5 モデルを準備しています...")
                    let pipeline = try self.buildPipeline(resourcesURL: resourcesURL, reduceMemory: reduceMemory)
                    update("パイプラインを作成しました（\(elapsedString(since: startTime))）。")

                    update("モデルを読み込んでいます...")
                    try pipeline.loadResources()
                    update("モデルの読み込みが完了しました（\(elapsedString(since: startTime))）。")

                    self.cachedPipeline = pipeline
                    self.cachedResourcesPath = resourcesURL.path
                    self.cachedReduceMemory = reduceMemory
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func unloadCachedPipeline(publish: (@Sendable (String) -> Void)? = nil) {
        queue.async { [self] in
            self.cachedPipeline?.unloadResources()
            self.cachedPipeline = nil
            self.cachedResourcesPath = nil
            self.cachedReduceMemory = nil
            let message = "読み込み済みモデルを解放しました。"
            print("[SD15] \(message)")
            publish?(message)
        }
    }

    func generateImage(
        resolution: ResolutionPreset,
        resourcesURL: URL,
        prompt: String,
        negativePrompt: String,
        stepCount: Int,
        seed: UInt32,
        guidanceScale: Float,
        schedulerOption: SchedulerOption,
        disableSafety: Bool,
        reduceMemory: Bool,
        publish: @escaping @Sendable (String) -> Void
    ) async throws -> UIImage {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    let startTime = Date()

                    func elapsedString(since start: Date) -> String {
                        String(format: "%.1f秒", Date().timeIntervalSince(start))
                    }

                    func update(_ message: String) {
                        print("[SD15] \(message)")
                        publish(message)
                    }

                    update("生成を開始します。")

                    let pipeline: StableDiffusionPipeline
                    if let cachedPipeline = self.cachedPipeline,
                       self.cachedResourcesPath == resourcesURL.path,
                       self.cachedReduceMemory == reduceMemory {
                        pipeline = cachedPipeline
                        update("読み込み済みモデルを再利用します（\(elapsedString(since: startTime))）。")
                    } else {
                        self.cachedPipeline?.unloadResources()
                        self.cachedPipeline = nil
                        self.cachedResourcesPath = nil
                        self.cachedReduceMemory = nil

                        update("パイプラインを準備しています...")
                        let newPipeline = try self.buildPipeline(resourcesURL: resourcesURL, reduceMemory: reduceMemory)
                        update("パイプライン作成完了（\(elapsedString(since: startTime))）。")

                        update("モデルを読み込んでいます...")
                        try newPipeline.loadResources()
                        update("モデル読み込み完了（\(elapsedString(since: startTime))）。")

                        self.cachedPipeline = newPipeline
                        self.cachedResourcesPath = resourcesURL.path
                        self.cachedReduceMemory = reduceMemory
                        pipeline = newPipeline
                    }

                    var generationConfig = PipelineConfiguration(prompt: prompt)
                    generationConfig.negativePrompt = negativePrompt
                    generationConfig.imageCount = 1
                    generationConfig.stepCount = stepCount
                    generationConfig.seed = seed
                    generationConfig.guidanceScale = guidanceScale
                    generationConfig.schedulerType = schedulerOption.stableDiffusionScheduler
                    generationConfig.disableSafety = disableSafety

                    update("画像を生成しています...")
                    var lastStepTimestamp = Date()
                    let images = try pipeline.generateImages(configuration: generationConfig) { progress in
                        let now = Date()
                        let delta = now.timeIntervalSince(lastStepTimestamp)
                        lastStepTimestamp = now
                        let stepNumber = progress.step + 1
                        let total = progress.stepCount
                        let percent = (Double(stepNumber) / Double(total)) * 100.0
                        update(
                            "ステップ \(stepNumber)/\(total) " +
                            "(\(String(format: "%.0f", percent))%) " +
                            "経過 \(elapsedString(since: startTime)) " +
                            "[前回から \(String(format: "%.2f", delta))秒]"
                        )
                        return true
                    }

                    update("生成とデコードが完了しました（\(elapsedString(since: startTime))）。")

                    guard let cgImage = images.first ?? nil else {
                        throw SDXLGeneratorViewModel.GeneratorError.noImageReturned
                    }
                    continuation.resume(returning: UIImage(cgImage: cgImage))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func buildPipeline(resourcesURL: URL, reduceMemory: Bool) throws -> StableDiffusionPipeline {
        let config = MLModelConfiguration()
        #if targetEnvironment(macCatalyst)
        config.computeUnits = .all
        #else
        config.computeUnits = .cpuAndNeuralEngine
        #endif

        return try StableDiffusionPipeline(
            resourcesAt: resourcesURL,
            controlNet: [],
            configuration: config,
            disableSafety: false,
            reduceMemory: reduceMemory
        )
    }
}

@MainActor
final class SDXLGeneratorViewModel: ObservableObject {
    static let examplePrompt = "small red robot, clean background, soft light"
    private static let reduceMemoryDefaultsKey = "reduceMemoryEnabled"
    private static let backgroundResourceRetentionNanoseconds: UInt64 = 10 * 60 * 1_000_000_000

    @Published var prompt = SDXLGeneratorViewModel.examplePrompt
    @Published var negativePrompt = NegativePromptPreset.standard.promptText
    @Published var selectedNegativePromptPreset: NegativePromptPreset = .standard
    @Published var stepCount = 12
    @Published var guidanceScale: Float = 6.0
    @Published var schedulerOption: SchedulerOption = .dpmSolverMultistep
    @Published var disableSafety = false
    @Published var seedMode: SeedMode = .random
    @Published var manualSeedText = ""
    @Published private(set) var reduceMemoryEnabled: Bool
    @Published var isGenerating = false
    @Published var isPreparingModel = false
    @Published var status = "待機中"
    @Published var errorMessage: String?
    @Published var image: UIImage?
    @Published var latestConsoleProgress = "まだ生成ログはありません。"
    @Published var hasPreloadedResources = false
    @Published private(set) var lastUsedSeed: UInt32?
    @Published private(set) var progressTitle = "待機中"
    @Published private(set) var progressDetail = "生成ボタンを押すとモデルを読み込んで開始します。"
    @Published private(set) var currentStep = 0
    @Published private(set) var totalSteps = 0
    @Published private(set) var progressUnitLabel = "ステップ"
    @Published private(set) var selectedResolution: ResolutionPreset = .p512
    @Published private(set) var availableResolutions: [ResolutionPreset] = []

    private let pipelineCache = SDXLPipelineCache()
    private let resourceLocator = RuntimeResourceLocator()
    private let stepRegex = try? NSRegularExpression(pattern: #"ステップ (\d+)/(\d+)"#)
    private var compiledResourcesPath: String?
    private var scheduledResourceReleaseTask: Task<Void, Never>?
    private let preferredResolution: ResolutionPreset = .p512

    init() {
        UserDefaults.standard.register(defaults: [Self.reduceMemoryDefaultsKey: true])
        reduceMemoryEnabled = UserDefaults.standard.bool(forKey: Self.reduceMemoryDefaultsKey)
        refreshAvailableResolutions()
    }

    var modelSummaryText: String { "SD 1.5 / \(selectedResolution.label)" }
    var isShowingExamplePrompt: Bool { prompt == Self.examplePrompt }
    var promptHintText: String { "例: tiny astronaut, blue sky" }
    var promptGuidanceText: String { "短い英語プロンプトに寄せた構成です。2〜6語くらいが扱いやすいです。" }
    var progressStepText: String { totalSteps > 0 ? "\(currentStep) / \(totalSteps) \(progressUnitLabel)" : " " }
    var progressValue: Double? {
        guard totalSteps > 0 else { return nil }
        return Double(currentStep) / Double(totalSteps)
    }
    var lastUsedSeedText: String {
        lastUsedSeed.map(String.init) ?? "まだ生成していません"
    }
    var resolutionSummaryText: String {
        let names = availableResolutions.map(\.label).joined(separator: " / ")
        return names.isEmpty ? "未検出" : names
    }

    func applyNegativePromptPreset(_ preset: NegativePromptPreset) {
        selectedNegativePromptPreset = preset
        negativePrompt = preset.promptText
    }

    func setReduceMemoryEnabled(_ enabled: Bool) {
        guard reduceMemoryEnabled != enabled else { return }
        reduceMemoryEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: Self.reduceMemoryDefaultsKey)
        hasPreloadedResources = false
        releaseCachedResources()
    }

    func preloadResourcesIfNeeded() {
        guard !isGenerating, !isPreparingModel, !hasPreloadedResources else { return }

        isPreparingModel = true
        errorMessage = nil
        status = "モデル準備中..."
        progressTitle = "モデル準備中"
        progressDetail = "ローカルモデルを確認しています。"
        resetProgress(unitLabel: "項目")
        latestConsoleProgress = "SD 1.5 モデルの事前準備を開始します。"

        Task {
            defer { isPreparingModel = false }

            do {
                let prepared = try await prepareResources()
                if compiledResourcesPath == prepared.url.path, hasPreloadedResources {
                    return
                }

                try await pipelineCache.preloadPipeline(
                    resourcesURL: prepared.url,
                    reduceMemory: reduceMemoryEnabled,
                    publish: { [weak self] message in
                        Task { @MainActor in
                            self?.handleProgressMessage(message)
                        }
                    }
                )
                compiledResourcesPath = prepared.url.path
                selectedResolution = prepared.resolution
                hasPreloadedResources = true
                latestConsoleProgress = "SD 1.5 モデルの事前読み込みが完了しました。"
                progressTitle = "準備完了"
                progressDetail = "\(prepared.resolution.label) モデルを使う準備ができました。"
                resetProgress()
                status = "待機中"
            } catch {
                errorMessage = error.localizedDescription
                latestConsoleProgress = error.localizedDescription
                progressTitle = "事前準備失敗"
                progressDetail = error.localizedDescription
                resetProgress()
                status = "事前準備失敗"
            }
        }
    }

    func releaseCachedResources() {
        cancelScheduledResourceRelease()
        pipelineCache.unloadCachedPipeline { [weak self] message in
            Task { @MainActor in
                self?.hasPreloadedResources = false
                self?.latestConsoleProgress = message
                if self?.isGenerating == false {
                    self?.status = "待機中"
                    self?.progressTitle = "待機中"
                    self?.progressDetail = "必要なときにモデルを再読み込みします。"
                    self?.resetProgress()
                }
            }
        }
    }

    func scheduleCachedResourceRelease() {
        guard !isGenerating, !isPreparingModel else { return }
        cancelScheduledResourceRelease()
        scheduledResourceReleaseTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: Self.backgroundResourceRetentionNanoseconds)
            } catch {
                return
            }

            guard !Task.isCancelled else { return }
            self?.releaseCachedResources()
        }
    }

    func cancelScheduledResourceRelease() {
        scheduledResourceReleaseTask?.cancel()
        scheduledResourceReleaseTask = nil
    }

    func generate() {
        cancelScheduledResourceRelease()
        guard !isGenerating, !isPreparingModel else { return }

        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            errorMessage = "プロンプトを入力してください。"
            status = "入力待ち"
            progressTitle = "入力待ち"
            progressDetail = "生成したい内容を書いてください。"
            return
        }

        let seed: UInt32
        do {
            seed = try resolveSeed()
        } catch {
            errorMessage = error.localizedDescription
            status = "設定を確認してください"
            progressTitle = "設定エラー"
            progressDetail = error.localizedDescription
            return
        }

        image = nil
        errorMessage = nil
        isGenerating = true
        lastUsedSeed = seed
        status = "生成準備中..."
        progressTitle = "生成準備中"
        progressDetail = "モデルの状態を確認しています。"
        resetProgress()
        latestConsoleProgress = "生成を開始します。"

        let prompt = self.prompt
        let negativePrompt = self.negativePrompt
        let stepCount = self.stepCount
        let guidanceScale = self.guidanceScale
        let schedulerOption = self.schedulerOption
        let disableSafety = self.disableSafety
        let reduceMemory = self.reduceMemoryEnabled

        Task {
            do {
                let prepared = try await prepareResources()
                let generatedImage = try await runGeneration(
                    resolution: prepared.resolution,
                    resourcesURL: prepared.url,
                    prompt: prompt,
                    negativePrompt: negativePrompt,
                    stepCount: stepCount,
                    seed: seed,
                    guidanceScale: guidanceScale,
                    schedulerOption: schedulerOption,
                    disableSafety: disableSafety,
                    reduceMemory: reduceMemory
                )
                image = generatedImage
                selectedResolution = prepared.resolution
                status = "生成完了"
                progressTitle = "生成完了"
                progressDetail = "画像ができました。保存や共有ができます。"
                resetProgress()
                latestConsoleProgress = "画像生成が完了しました。"
                isGenerating = false
                hasPreloadedResources = true
                compiledResourcesPath = prepared.url.path
            } catch {
                errorMessage = error.localizedDescription
                latestConsoleProgress = error.localizedDescription
                status = "生成失敗"
                progressTitle = "生成失敗"
                progressDetail = error.localizedDescription
                resetProgress()
                isGenerating = false
            }
        }
    }

    private func prepareResources() async throws -> (resolution: ResolutionPreset, url: URL) {
        let preferredResolution = self.preferredResolution
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [resourceLocator] in
                do {
                    let prepared = try resourceLocator.resolveBestResources(preferred: preferredResolution) { message in
                        print("[SD15] \(message)")
                    }
                    continuation.resume(returning: prepared)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func refreshAvailableResolutions() {
        do {
            availableResolutions = try resourceLocator.availableResolutions()
            if let best = availableResolutions.first(where: { $0 == preferredResolution }) ?? availableResolutions.first {
                selectedResolution = best
            }
        } catch {
            availableResolutions = []
        }
    }

    private func resolveSeed() throws -> UInt32 {
        switch seedMode {
        case .random:
            return UInt32.random(in: UInt32.min...UInt32.max)
        case .manual:
            let trimmed = manualSeedText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw GeneratorError.invalidSeed("固定シード値が空です。0 から 4294967295 の整数を入力してください。")
            }
            guard let value = UInt32(trimmed) else {
                throw GeneratorError.invalidSeed("固定シード値の形式が正しくありません。0 から 4294967295 の整数を入力してください。")
            }
            return value
        }
    }

    private func runGeneration(
        resolution: ResolutionPreset,
        resourcesURL: URL,
        prompt: String,
        negativePrompt: String,
        stepCount: Int,
        seed: UInt32,
        guidanceScale: Float,
        schedulerOption: SchedulerOption,
        disableSafety: Bool,
        reduceMemory: Bool
    ) async throws -> UIImage {
        guard #available(iOS 18.0, macCatalyst 18.0, *) else {
            throw GeneratorError.unsupportedOS
        }

        return try await pipelineCache.generateImage(
            resolution: resolution,
            resourcesURL: resourcesURL,
            prompt: prompt,
            negativePrompt: negativePrompt,
            stepCount: stepCount,
            seed: seed,
            guidanceScale: guidanceScale,
            schedulerOption: schedulerOption,
            disableSafety: disableSafety,
            reduceMemory: reduceMemory,
            publish: { [weak self] message in
                Task { @MainActor in
                    self?.handleProgressMessage(message)
                }
            }
        )
    }

    private func handleProgressMessage(_ message: String) {
        latestConsoleProgress = message
        status = message

        if let match = stepMatch(in: message) {
            progressTitle = "画像を生成中"
            progressDetail = "今は \(match.current) / \(match.total) ステップ目です。"
            setProgress(current: match.current, total: match.total, unitLabel: "ステップ")
            return
        }

        resetProgress()

        if message.contains("モデルを読み込んでいます") {
            progressTitle = "モデルを読み込み中"
            progressDetail = "ローカルモデルを読み込んでいます。完了までしばらくかかる場合があります。"
        } else if message.contains("パイプライン") || message.contains("準備しています") {
            progressTitle = "生成準備中"
            progressDetail = "生成の前処理を進めています。"
        } else if message.contains("生成を開始します") || message.contains("画像を生成しています") {
            progressTitle = "生成開始"
            progressDetail = "まもなくステップ進行が始まります。"
        } else if message.contains("モデルの読み込みが完了") || message.contains("モデル読み込み完了") {
            progressTitle = "モデル準備完了"
            progressDetail = "これから画像生成に入ります。"
        } else if message.contains("生成とデコードが完了") {
            progressTitle = "仕上げ中"
            progressDetail = "画像の変換処理を終えています。"
        } else {
            progressTitle = isGenerating ? "処理中" : "待機中"
            progressDetail = message
        }
    }

    private func stepMatch(in message: String) -> (current: Int, total: Int)? {
        guard let stepRegex else { return nil }
        return progressMatch(in: message, regex: stepRegex)
    }

    private func progressMatch(
        in message: String,
        regex: NSRegularExpression
    ) -> (current: Int, total: Int)? {
        let range = NSRange(message.startIndex..<message.endIndex, in: message)
        guard let match = regex.firstMatch(in: message, options: [], range: range),
              let currentRange = Range(match.range(at: 1), in: message),
              let totalRange = Range(match.range(at: 2), in: message),
              let current = Int(message[currentRange]),
              let total = Int(message[totalRange]) else {
            return nil
        }
        return (current, total)
    }

    private func setProgress(current: Int, total: Int, unitLabel: String) {
        currentStep = current
        totalSteps = total
        progressUnitLabel = unitLabel
    }

    private func resetProgress(unitLabel: String = "ステップ") {
        currentStep = 0
        totalSteps = 0
        progressUnitLabel = unitLabel
    }
}

extension SDXLGeneratorViewModel {
    enum GeneratorError: LocalizedError {
        case unsupportedOS
        case missingResources(String)
        case noImageReturned
        case invalidSeed(String)

        var errorDescription: String? {
            switch self {
            case .unsupportedOS:
                return "このアプリは iOS 18 以降または Mac Catalyst が必要です。"
            case .missingResources(let path):
                return "必要なモデルファイルが見つかりません: \(path)"
            case .noImageReturned:
                return "生成処理は完了しましたが、画像を受け取れませんでした。"
            case .invalidSeed(let message):
                return message
            }
        }
    }
}
