import CoreML
import Foundation
import StableDiffusion
import SwiftUI
import UIKit

enum ResolutionPreset: Int {
    case p768 = 768

    var pixelSize: Int { rawValue }
    var label: String { "\(rawValue)x\(rawValue)" }
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
            return "手や輪郭の崩れを少し抑えたいとき向けです。"
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

private enum SDXLRuntimeResources {
    struct ModelPackage {
        let sourceName: String
        let compiledName: String
        let displayName: String
    }

    static let modelPackages: [ModelPackage] = [
        .init(sourceName: "TextEncoder.mlpackage", compiledName: "TextEncoder.mlmodelc", displayName: "Text Encoder"),
        .init(sourceName: "TextEncoder2.mlpackage", compiledName: "TextEncoder2.mlmodelc", displayName: "Text Encoder 2"),
        .init(sourceName: "UnetChunk1.mlpackage", compiledName: "UnetChunk1.mlmodelc", displayName: "UNet Chunk 1"),
        .init(sourceName: "UnetChunk2.mlpackage", compiledName: "UnetChunk2.mlmodelc", displayName: "UNet Chunk 2"),
        .init(sourceName: "VAEDecoder.mlpackage", compiledName: "VAEDecoder.mlmodelc", displayName: "VAE Decoder"),
    ]

    static let tokenizerFiles = ["vocab.json", "merges.txt"]
}

private final class RuntimeResourceCompiler: @unchecked Sendable {
    private let fileManager = FileManager.default

    func prepareCompiledResources(
        publish: @escaping @Sendable (String) -> Void
    ) throws -> URL {
        if let precompiledURL = try bundledCompiledResourcesDirectoryIfAvailable() {
            publish("同梱済みモデルを確認しました。")
            return precompiledURL
        }

        let bundledSourcesURL = try bundledSourcesDirectory()
        let compiledResourcesURL = try compiledResourcesDirectory()

        try fileManager.createDirectory(at: compiledResourcesURL, withIntermediateDirectories: true)

        for tokenFile in SDXLRuntimeResources.tokenizerFiles {
            let sourceURL = bundledSourcesURL.appending(path: tokenFile)
            let destinationURL = compiledResourcesURL.appending(path: tokenFile)
            if !fileManager.fileExists(atPath: sourceURL.path) {
                throw SDXLGeneratorViewModel.GeneratorError.missingResources(sourceURL.path)
            }
            if !fileManager.fileExists(atPath: destinationURL.path) {
                try copyItemReplacingIfNeeded(from: sourceURL, to: destinationURL)
            }
        }

        for (index, package) in SDXLRuntimeResources.modelPackages.enumerated() {
            let compiledURL = compiledResourcesURL.appending(path: package.compiledName, directoryHint: .isDirectory)
            if fileManager.fileExists(atPath: compiledURL.path) {
                publish("端末向けモデル確認 \(index + 1)/\(SDXLRuntimeResources.modelPackages.count): \(package.displayName)")
                continue
            }

            let sourceURL = bundledSourcesURL.appending(path: package.sourceName, directoryHint: .isDirectory)
            if !fileManager.fileExists(atPath: sourceURL.path) {
                throw SDXLGeneratorViewModel.GeneratorError.missingResources(sourceURL.path)
            }

            publish("端末向けにモデルを最適化中 \(index + 1)/\(SDXLRuntimeResources.modelPackages.count): \(package.displayName)")
            let temporaryCompiledURL = try MLModel.compileModel(at: sourceURL)
            try copyItemReplacingIfNeeded(from: temporaryCompiledURL, to: compiledURL)
        }

        return compiledResourcesURL
    }

    private func bundledCompiledResourcesDirectoryIfAvailable() throws -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw SDXLGeneratorViewModel.GeneratorError.missingResources("Bundle.main.resourceURL")
        }

        let url = resourceURL
            .appending(path: "BundledResources", directoryHint: .isDirectory)
            .appending(path: "sdxl", directoryHint: .isDirectory)
            .appending(path: "768", directoryHint: .isDirectory)
            .appending(path: "Resources", directoryHint: .isDirectory)

        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }

        for package in SDXLRuntimeResources.modelPackages {
            let modelURL = url.appending(path: package.compiledName, directoryHint: .isDirectory)
            let weightsURL = modelURL
                .appending(path: "weights", directoryHint: .isDirectory)
                .appending(path: "weight.bin")
            guard fileManager.fileExists(atPath: modelURL.path),
                  fileManager.fileExists(atPath: weightsURL.path) else {
                return nil
            }
        }

        for tokenFile in SDXLRuntimeResources.tokenizerFiles {
            guard fileManager.fileExists(atPath: url.appending(path: tokenFile).path) else {
                return nil
            }
        }

        return url
    }

    private func bundledSourcesDirectory() throws -> URL {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw SDXLGeneratorViewModel.GeneratorError.missingResources("Bundle.main.resourceURL")
        }
        let url = resourceURL
            .appending(path: "BundledResources", directoryHint: .isDirectory)
            .appending(path: "sdxl", directoryHint: .isDirectory)
            .appending(path: "768", directoryHint: .isDirectory)
            .appending(path: "Sources", directoryHint: .isDirectory)

        guard fileManager.fileExists(atPath: url.path) else {
            throw SDXLGeneratorViewModel.GeneratorError.missingResources(url.path)
        }
        return url
    }

    private func compiledResourcesDirectory() throws -> URL {
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return appSupport
            .appending(path: "SDXLCompiledResources", directoryHint: .isDirectory)
            .appending(path: "sdxl", directoryHint: .isDirectory)
            .appending(path: "768", directoryHint: .isDirectory)
            .appending(path: "Resources", directoryHint: .isDirectory)
    }

    private func copyItemReplacingIfNeeded(from sourceURL: URL, to destinationURL: URL) throws {
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.copyItem(at: sourceURL, to: destinationURL)
    }
}

final class SDXLPipelineCache: @unchecked Sendable {
    private let queue = DispatchQueue(label: "SDXLCoreMLTest.pipeline-cache")
    private var cachedPipeline: StableDiffusionXLPipeline?
    private var cachedResourcesPath: String?

    func preloadPipeline(
        resourcesURL: URL,
        publish: @escaping @Sendable (String) -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    if self.cachedResourcesPath == resourcesURL.path, self.cachedPipeline != nil {
                        publish("モデルはすでに読み込み済みです。")
                        continuation.resume(returning: ())
                        return
                    }

                    let startTime = Date()
                    self.cachedPipeline?.unloadResources()
                    self.cachedPipeline = nil
                    self.cachedResourcesPath = nil

                    func elapsedString(since start: Date) -> String {
                        String(format: "%.1f秒", Date().timeIntervalSince(start))
                    }

                    func update(_ message: String) {
                        print("[SDXL] \(message)")
                        publish(message)
                    }

                    update("SDXL モデルを準備しています...")
                    let pipeline = try self.buildPipeline(resourcesURL: resourcesURL)
                    update("パイプラインを作成しました（\(elapsedString(since: startTime))）。")

                    update("モデルを読み込んでいます...")
                    try pipeline.loadResources()
                    update("モデルの読み込みが完了しました（\(elapsedString(since: startTime))）。")

                    self.cachedPipeline = pipeline
                    self.cachedResourcesPath = resourcesURL.path
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
            let message = "読み込み済みモデルを解放しました。"
            print("[SDXL] \(message)")
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
                        print("[SDXL] \(message)")
                        publish(message)
                    }

                    update("生成を開始します。")

                    let pipeline: StableDiffusionXLPipeline
                    if let cachedPipeline = self.cachedPipeline, self.cachedResourcesPath == resourcesURL.path {
                        pipeline = cachedPipeline
                        update("読み込み済みモデルを再利用します（\(elapsedString(since: startTime))）。")
                    } else {
                        self.cachedPipeline?.unloadResources()
                        self.cachedPipeline = nil
                        self.cachedResourcesPath = nil

                        update("パイプラインを準備しています...")
                        let newPipeline = try self.buildPipeline(resourcesURL: resourcesURL)
                        update("パイプライン作成完了（\(elapsedString(since: startTime))）。")

                        update("モデルを読み込んでいます...")
                        try newPipeline.loadResources()
                        update("モデル読み込み完了（\(elapsedString(since: startTime))）。")

                        self.cachedPipeline = newPipeline
                        self.cachedResourcesPath = resourcesURL.path
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
                    generationConfig.useDenoisedIntermediates = false
                    generationConfig.encoderScaleFactor = 0.13025
                    generationConfig.decoderScaleFactor = 0.13025
                    generationConfig.originalSize = Float32(resolution.pixelSize)
                    generationConfig.targetSize = Float32(resolution.pixelSize)
                    generationConfig.cropsCoordsTopLeft = 0

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

    private func buildPipeline(resourcesURL: URL) throws -> StableDiffusionXLPipeline {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return try StableDiffusionXLPipeline(
            resourcesAt: resourcesURL,
            configuration: config,
            reduceMemory: true
        )
    }
}

@MainActor
final class SDXLGeneratorViewModel: ObservableObject {
    @Published var prompt = "cinematic photo of a retro robot walking in Tokyo at night, neon reflections, 35mm film, detailed"
    @Published var negativePrompt = NegativePromptPreset.standard.promptText
    @Published var selectedNegativePromptPreset: NegativePromptPreset = .standard
    @Published var stepCount = 20
    @Published var guidanceScale: Float = 5.0
    @Published var schedulerOption: SchedulerOption = .dpmSolverMultistep
    @Published var disableSafety = false
    @Published var seedMode: SeedMode = .random
    @Published var manualSeedText = ""
    @Published var isGenerating = false
    @Published var status = "待機中"
    @Published var errorMessage: String?
    @Published var image: UIImage?
    @Published var latestConsoleProgress = "まだ生成ログはありません。"
    @Published var hasPreloadedResources = false
    @Published private(set) var lastUsedSeed: UInt32?
    @Published private(set) var progressTitle = "待機中"
    @Published private(set) var progressDetail = "生成前です。"
    @Published private(set) var currentStep = 0
    @Published private(set) var totalSteps = 0

    private let pipelineCache = SDXLPipelineCache()
    private let resourceCompiler = RuntimeResourceCompiler()
    private let stepRegex = try? NSRegularExpression(pattern: #"ステップ (\d+)/(\d+)"#)
    private var compiledResourcesPath: String?

    let selectedResolution: ResolutionPreset = .p768

    var modelSummaryText: String { "SDXL / \(selectedResolution.label)" }
    var promptHintText: String { "例: cinematic photo of a small cat in a Tokyo cafe, soft morning light" }
    var promptGuidanceText: String { "日本語でも入力できますが、SDXL 1.0 は英語プロンプトの方が安定しやすいです。" }
    var progressStepText: String { totalSteps > 0 ? "\(currentStep) / \(totalSteps) ステップ" : " " }
    var progressValue: Double? {
        guard totalSteps > 0 else { return nil }
        return Double(currentStep) / Double(totalSteps)
    }
    var lastUsedSeedText: String {
        lastUsedSeed.map(String.init) ?? "まだ生成していません"
    }

    func applyNegativePromptPreset(_ preset: NegativePromptPreset) {
        selectedNegativePromptPreset = preset
        negativePrompt = preset.promptText
    }

    func preloadResourcesIfNeeded() {
        Task {
            do {
                let compiledURL = try await prepareResources()
                if compiledResourcesPath == compiledURL.path, hasPreloadedResources {
                    return
                }

                try await pipelineCache.preloadPipeline(
                    resourcesURL: compiledURL,
                    publish: { [weak self] message in
                        Task { @MainActor in
                            self?.handleProgressMessage(message)
                        }
                    }
                )
                compiledResourcesPath = compiledURL.path
                hasPreloadedResources = true
                latestConsoleProgress = "SDXL モデルの事前読み込みが完了しました。"
                progressTitle = "準備完了"
                progressDetail = "すぐに生成を始められます。"
                status = "待機中"
            } catch {
                errorMessage = error.localizedDescription
                latestConsoleProgress = error.localizedDescription
                progressTitle = "事前準備失敗"
                progressDetail = error.localizedDescription
                status = "事前準備失敗"
            }
        }
    }

    func releaseCachedResources() {
        pipelineCache.unloadCachedPipeline { [weak self] message in
            Task { @MainActor in
                self?.hasPreloadedResources = false
                self?.latestConsoleProgress = message
                if self?.isGenerating == false {
                    self?.status = "待機中"
                    self?.progressTitle = "待機中"
                    self?.progressDetail = "必要なときにモデルを再読み込みします。"
                    self?.currentStep = 0
                    self?.totalSteps = 0
                }
            }
        }
    }

    func generate() {
        guard !isGenerating else { return }

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
        currentStep = 0
        totalSteps = 0
        latestConsoleProgress = "生成を開始します。"

        let prompt = self.prompt
        let negativePrompt = self.negativePrompt
        let stepCount = self.stepCount
        let guidanceScale = self.guidanceScale
        let schedulerOption = self.schedulerOption
        let disableSafety = self.disableSafety
        let resolution = self.selectedResolution

        Task {
            do {
                let compiledURL = try await prepareResources()
                let generatedImage = try await runGeneration(
                    resolution: resolution,
                    resourcesURL: compiledURL,
                    prompt: prompt,
                    negativePrompt: negativePrompt,
                    stepCount: stepCount,
                    seed: seed,
                    guidanceScale: guidanceScale,
                    schedulerOption: schedulerOption,
                    disableSafety: disableSafety
                )
                image = generatedImage
                status = "生成完了"
                progressTitle = "生成完了"
                progressDetail = "画像ができました。保存や共有ができます。"
                currentStep = 0
                totalSteps = 0
                latestConsoleProgress = "画像生成が完了しました。"
                isGenerating = false
                hasPreloadedResources = true
                compiledResourcesPath = compiledURL.path
            } catch {
                errorMessage = error.localizedDescription
                latestConsoleProgress = error.localizedDescription
                status = "生成失敗"
                progressTitle = "生成失敗"
                progressDetail = error.localizedDescription
                currentStep = 0
                totalSteps = 0
                isGenerating = false
            }
        }
    }

    private func prepareResources() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [resourceCompiler] in
                do {
                    let compiledURL = try resourceCompiler.prepareCompiledResources { message in
                        print("[SDXL] \(message)")
                        Task { @MainActor [weak self] in
                            self?.handleCompilationMessage(message)
                        }
                    }
                    continuation.resume(returning: compiledURL)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
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
        disableSafety: Bool
    ) async throws -> UIImage {
        guard #available(iOS 18.0, *) else {
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
            publish: { [weak self] message in
                Task { @MainActor in
                    self?.handleProgressMessage(message)
                }
            }
        )
    }

    private func handleCompilationMessage(_ message: String) {
        latestConsoleProgress = message
        progressTitle = "端末向け準備中"
        progressDetail = message
        status = message
        currentStep = 0
        totalSteps = 0
    }

    private func handleProgressMessage(_ message: String) {
        latestConsoleProgress = message
        status = message

        if let match = stepMatch(in: message) {
            progressTitle = "画像を生成中"
            progressDetail = "今は \(match.current) / \(match.total) ステップ目です。"
            currentStep = match.current
            totalSteps = match.total
            return
        }

        currentStep = 0
        totalSteps = 0

        if message.contains("モデルを読み込んでいます") {
            progressTitle = "モデルを読み込み中"
            progressDetail = "初回や再起動直後は少し時間がかかります。"
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
        let range = NSRange(message.startIndex..<message.endIndex, in: message)
        guard let match = stepRegex.firstMatch(in: message, options: [], range: range),
              let currentRange = Range(match.range(at: 1), in: message),
              let totalRange = Range(match.range(at: 2), in: message),
              let current = Int(message[currentRange]),
              let total = Int(message[totalRange]) else {
            return nil
        }
        return (current, total)
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
                return "このアプリは iOS 18 以降が必要です。"
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
