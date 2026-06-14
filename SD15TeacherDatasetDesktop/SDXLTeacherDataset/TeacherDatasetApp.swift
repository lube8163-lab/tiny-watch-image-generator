import CoreML
import Foundation
import StableDiffusion
import SwiftUI
import UIKit

@main
struct TeacherDatasetApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var viewModel = TeacherDatasetViewModel()

    var body: some Scene {
        WindowGroup {
            TeacherDatasetView(viewModel: viewModel)
                .preferredColorScheme(.light)
                .onChange(of: scenePhase) { _, phase in
                    viewModel.handleScenePhase(phase)
                }
        }
    }
}

private enum TeacherStyle {
    static let background = Color(red: 0.95, green: 0.96, blue: 0.94)
    static let darkBackground = Color(red: 0.04, green: 0.05, blue: 0.05)
    static let surface = Color.white
    static let darkSurface = Color(red: 0.08, green: 0.10, blue: 0.10)
    static let surfaceInset = Color(red: 0.92, green: 0.94, blue: 0.91)
    static let darkInset = Color(red: 0.13, green: 0.16, blue: 0.15)
    static let ink = Color(red: 0.16, green: 0.19, blue: 0.18)
    static let darkInk = Color(red: 0.86, green: 0.91, blue: 0.88)
    static let muted = Color(red: 0.43, green: 0.47, blue: 0.45)
    static let line = Color.black.opacity(0.08)
    static let accent = Color(red: 0.18, green: 0.46, blue: 0.36)
}

private struct TeacherDatasetView: View {
    @ObservedObject var viewModel: TeacherDatasetViewModel
    @State private var isShowingRunningOverlay = true

    var body: some View {
        NavigationStack {
            ZStack {
                if isBlackout {
                    blackoutView
                } else {
                    TeacherStyle.background
                        .ignoresSafeArea()
                    ScrollView {
                        VStack(spacing: 14) {
                            statusCard
                            controlsCard
                            logCard
                            if let image = viewModel.lastImage {
                                imageCard(image)
                            }
                        }
                        .padding(16)
                    }
                }
            }
            .navigationTitle("SD 1.5 Teacher Dataset")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task {
            viewModel.startAutomationIfNeeded()
        }
        .onChange(of: viewModel.isRunning) { _, isRunning in
            guard isRunning else {
                isShowingRunningOverlay = true
                return
            }
            revealRunningOverlay()
        }
    }

    private var isBlackout: Bool {
        viewModel.useDimProgressView && (viewModel.isRunning || viewModel.isPostRunBlackout)
    }
    private var isDim: Bool { isBlackout }
    private var fg: Color { isDim ? TeacherStyle.darkInk : TeacherStyle.ink }
    private var muted: Color { isDim ? TeacherStyle.darkInk.opacity(0.68) : TeacherStyle.muted }
    private var surface: Color { isDim ? TeacherStyle.darkSurface : TeacherStyle.surface }
    private var inset: Color { isDim ? TeacherStyle.darkInset : TeacherStyle.surfaceInset }

    private var blackoutView: some View {
        ZStack {
            Color.black
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture {
                    if viewModel.isPostRunBlackout {
                        viewModel.dismissPostRunBlackout()
                    } else {
                        revealRunningOverlay()
                    }
                }

            if isShowingRunningOverlay {
                VStack(spacing: 14) {
                    Label(viewModel.statusTitle, systemImage: viewModel.isRunning ? "cpu" : "checkmark.circle")
                        .font(.headline)
                        .foregroundStyle(TeacherStyle.darkInk.opacity(0.72))

                    if viewModel.isRunning {
                        ProgressView(value: viewModel.progressValue)
                            .tint(TeacherStyle.accent)
                            .frame(maxWidth: 260)
                    }

                    Text(viewModel.progressText)
                        .font(.system(.footnote, design: .monospaced))
                        .foregroundStyle(TeacherStyle.darkInk.opacity(0.66))

                    Text(viewModel.currentJobTitle)
                        .font(.caption2)
                        .foregroundStyle(TeacherStyle.darkInk.opacity(0.48))
                        .lineLimit(2)
                        .multilineTextAlignment(.center)

                    Text(viewModel.isRunning ? "Tap to show briefly" : "Tap to return")
                        .font(.caption2)
                        .foregroundStyle(TeacherStyle.darkInk.opacity(0.32))
                }
                .padding(18)
                .background(Color.black.opacity(0.001))
                .onTapGesture {
                    if viewModel.isPostRunBlackout {
                        viewModel.dismissPostRunBlackout()
                    } else {
                        revealRunningOverlay()
                    }
                }
            }
        }
    }

    private func revealRunningOverlay() {
        isShowingRunningOverlay = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
            if viewModel.isRunning && viewModel.useDimProgressView {
                isShowingRunningOverlay = false
            }
        }
    }

    private var statusCard: some View {
        card {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label(viewModel.statusTitle, systemImage: viewModel.isRunning ? "cpu" : "tray.and.arrow.down")
                        .font(.headline)
                        .foregroundStyle(fg)
                    Spacer()
                    Text(viewModel.progressText)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(fg)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(inset)
                        .clipShape(Capsule())
                }

                ProgressView(value: viewModel.progressValue)
                    .tint(TeacherStyle.accent)

                valueRow("Preset", value: viewModel.selectedPreset.displayName)
                valueRow("Current", value: viewModel.currentJobTitle)
                valueRow("Last elapsed", value: viewModel.lastElapsedText)
                valueRow("ETA", value: viewModel.estimatedRemainingText)
                valueRow("Failed", value: "\(viewModel.failedCount)")
                valueRow("Thermal / Battery", value: "\(viewModel.thermalStateText) / \(viewModel.batteryStateText)")
                if !viewModel.outputPath.isEmpty {
                    valueRow("Output", value: viewModel.outputPath)
                }
            }
        }
    }

    private var controlsCard: some View {
        card {
            VStack(alignment: .leading, spacing: 14) {
                Picker("Run preset", selection: $viewModel.selectedPreset) {
                    ForEach(TeacherRunPreset.allCases) { preset in
                        Text(preset.displayName).tag(preset)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(viewModel.isRunning)

                Text(viewModel.selectedPreset.detail)
                    .font(.footnote)
                    .foregroundStyle(muted)

                Stepper("Steps: \(viewModel.stepCount)", value: $viewModel.stepCount, in: 20...30)
                    .foregroundStyle(fg)
                    .disabled(viewModel.isRunning)

                HStack(spacing: 10) {
                    Button {
                        viewModel.startSelectedPreset()
                    } label: {
                        Label("Start", systemImage: "play.fill")
                    }
                    .buttonStyle(TeacherButtonStyle(isDim: isDim))
                    .disabled(viewModel.isRunning)

                    Button {
                        viewModel.pause()
                    } label: {
                        Label("Pause", systemImage: "pause.fill")
                    }
                    .buttonStyle(TeacherButtonStyle(isDim: isDim))
                    .disabled(!viewModel.isRunning || viewModel.isPaused)

                    Button {
                        viewModel.resume()
                    } label: {
                        Label("Resume", systemImage: "forward.fill")
                    }
                    .buttonStyle(TeacherButtonStyle(isDim: isDim))
                    .disabled(viewModel.isRunning || !viewModel.hasResumableRun)
                }

                HStack(spacing: 10) {
                    Button {
                        viewModel.stop()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(TeacherButtonStyle(isDim: isDim))
                    .disabled(!viewModel.isRunning && !viewModel.hasResumableRun)

                    Button {
                        viewModel.resumeLatestRun()
                    } label: {
                        Label("Resume Latest", systemImage: "clock.arrow.circlepath")
                    }
                    .buttonStyle(TeacherButtonStyle(isDim: isDim))
                    .disabled(viewModel.isRunning || !viewModel.hasResumableRun)
                }

                Divider().opacity(0.55)

                Toggle("Require charging", isOn: $viewModel.requireCharging)
                    .tint(TeacherStyle.accent)
                Toggle("Pause on high thermal state", isOn: $viewModel.pauseOnHighThermalState)
                    .tint(TeacherStyle.accent)
                Toggle("Keep screen awake", isOn: $viewModel.keepScreenAwake)
                    .tint(TeacherStyle.accent)
                Toggle("Dim progress screen while running", isOn: $viewModel.useDimProgressView)
                    .tint(TeacherStyle.accent)
            }
        }
    }

    private var logCard: some View {
        card {
            VStack(alignment: .leading, spacing: 8) {
                Label("Log", systemImage: "text.alignleft")
                    .font(.headline)
                    .foregroundStyle(fg)
                Text(viewModel.latestLog)
                    .font(.footnote)
                    .foregroundStyle(muted)
                    .textSelection(.enabled)
            }
        }
    }

    private func imageCard(_ image: UIImage) -> some View {
        card {
            VStack(alignment: .leading, spacing: 12) {
                Label("Last image", systemImage: "photo")
                    .font(.headline)
                    .foregroundStyle(fg)
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(TeacherStyle.line, lineWidth: 1)
                    )
            }
        }
    }

    private func card<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(surface)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(TeacherStyle.line, lineWidth: 1)
            )
    }

    private func valueRow(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption)
                .foregroundStyle(muted)
            Text(value)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(fg)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct TeacherButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    let isDim: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(isEnabled ? (isDim ? TeacherStyle.darkInk : TeacherStyle.ink) : TeacherStyle.muted.opacity(0.65))
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(configuration.isPressed ? TeacherStyle.surfaceInset.opacity(0.7) : (isDim ? TeacherStyle.darkInset : TeacherStyle.surfaceInset))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(TeacherStyle.line, lineWidth: 1)
            )
    }
}

private enum ResolutionPreset: Int {
    case p512 = 512
    var pixelSize: Int { rawValue }
}

private enum SchedulerOption: String, Codable {
    case dpmpp2MKarras = "dpmpp_2m_karras"

    var stableDiffusionScheduler: StableDiffusionScheduler {
        .dpmSolverMultistepScheduler
    }
}

private enum TeacherRunPreset: String, CaseIterable, Identifiable, Codable {
    case smoke
    case pilot
    case firstTrain = "first_train"
    case strongerTrain = "stronger_train"

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .smoke: return "smoke"
        case .pilot: return "pilot"
        case .firstTrain: return "first_train"
        case .strongerTrain: return "stronger_train"
        }
    }
    var categoryCount: Int {
        16
    }
    var variantsPerCategory: Int {
        8
    }
    var seedsPerVariant: Int {
        switch self {
        case .smoke: return 1
        case .pilot: return 8
        case .firstTrain: return 32
        case .strongerTrain: return 64
        }
    }
    var detail: String {
        "\(categoryCount) categories x \(variantsPerCategory) variants x \(seedsPerVariant) seeds = \(totalJobs) jobs"
    }
    var totalJobs: Int { categoryCount * variantsPerCategory * seedsPerVariant }
}

private struct TeacherPromptPreset: Codable, Hashable {
    let key: String
    let title: String
    let subject: String
    let variants: [TeacherPromptVariant]
}

private struct TeacherPromptVariant: Codable, Hashable {
    let variant: String
    let prompt: String
    let qcFlags: [String]
}

private struct TeacherGenerationJob: Hashable, Codable {
    let id: String
    let key: String
    let title: String
    let variant: String
    let prompt: String
    let promptQCFlags: [String]
    let seed: UInt32
}

private struct TeacherDatasetSavedImages: Codable, Hashable {
    let size512: String
    let size256: String
    let size128: String
    let size64: String

    enum CodingKeys: String, CodingKey {
        case size512 = "512"
        case size256 = "256"
        case size128 = "128"
        case size64 = "64"
    }
}

private struct TeacherDatasetMetadataLine: Codable, Hashable {
    let id: String
    let key: String
    let title: String
    let variant: String
    let prompt: String
    let negativePrompt: String
    let seed: UInt32
    let modelFamily: String
    let steps: Int
    let guidanceScale: Float
    let scheduler: String
    let sourceWidth: Int
    let sourceHeight: Int
    let savedImages: TeacherDatasetSavedImages
    let accepted: Bool
    let qcFlags: [String]
    let rejectReason: String?
    let imageMode: String
    let fileSizeBytes: Int
    let elapsedSeconds: Double
    let thermalState: String
    let batteryLevel: Float
    let batteryState: String
    let createdAtUnix: Int

    enum CodingKeys: String, CodingKey {
        case id
        case key
        case title
        case variant
        case prompt
        case negativePrompt = "negative_prompt"
        case seed
        case modelFamily = "model_family"
        case steps
        case guidanceScale = "guidance_scale"
        case scheduler
        case sourceWidth = "source_width"
        case sourceHeight = "source_height"
        case savedImages = "saved_images"
        case accepted
        case qcFlags = "qc_flags"
        case rejectReason = "reject_reason"
        case imageMode = "image_mode"
        case fileSizeBytes = "file_size_bytes"
        case elapsedSeconds = "elapsed_seconds"
        case thermalState = "thermal_state"
        case batteryLevel = "battery_level"
        case batteryState = "battery_state"
        case createdAtUnix = "created_at_unix"
    }
}

private struct FailedJobLine: Codable, Hashable {
    let id: String
    let key: String
    let variant: String
    let seed: UInt32
    let error: String
    let createdAtUnix: Int

    enum CodingKeys: String, CodingKey {
        case id
        case key
        case variant
        case seed
        case error
        case createdAtUnix = "created_at_unix"
    }
}

private struct TeacherDatasetManifest: Codable, Hashable {
    let runID: String
    let preset: TeacherRunPreset
    let appVersion: String
    let appBuild: String
    let modelFamily: String
    let resourceDirectory: String
    let sourceResolution: Int
    let totalPlannedJobs: Int
    var completedJobs: Int
    var failedJobs: Int
    let startedAtUnix: Int
    var endedAtUnix: Int?
    var status: String
    let promptSetVersion: String
    let generationSettings: TeacherDatasetGenerationSettings
    let deviceName: String

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case preset
        case appVersion = "app_version"
        case appBuild = "app_build"
        case modelFamily = "model_family"
        case resourceDirectory = "resource_directory"
        case sourceResolution = "source_resolution"
        case totalPlannedJobs = "total_planned_jobs"
        case completedJobs = "completed_jobs"
        case failedJobs = "failed_jobs"
        case startedAtUnix = "started_at_unix"
        case endedAtUnix = "ended_at_unix"
        case status
        case promptSetVersion = "prompt_set_version"
        case generationSettings = "generation_settings"
        case deviceName = "device_name"
    }
}

private struct TeacherDatasetGenerationSettings: Codable, Hashable {
    let steps: Int
    let guidanceScale: Float
    let scheduler: String
    let negativePrompt: String
    let sourceWidth: Int
    let sourceHeight: Int
    let reduceMemory: Bool

    enum CodingKeys: String, CodingKey {
        case steps
        case guidanceScale = "guidance_scale"
        case scheduler
        case negativePrompt = "negative_prompt"
        case sourceWidth = "source_width"
        case sourceHeight = "source_height"
        case reduceMemory = "reduce_memory"
    }
}

private struct SavedImageResult {
    let paths: TeacherDatasetSavedImages
    let totalFileSizeBytes: Int
}

private enum TeacherDatasetFileWriter {
    static func teacherDatasetsDirectory() throws -> URL {
        let documentsURL = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let url = documentsURL.appending(path: "TeacherDatasets", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func makeRunDirectory(runID: String) throws -> URL {
        let runURL = try teacherDatasetsDirectory().appending(path: runID, directoryHint: .isDirectory)
        for folder in ["images_512", "images_256", "images_128", "images_64", "reports"] {
            try FileManager.default.createDirectory(
                at: runURL.appending(path: folder, directoryHint: .isDirectory),
                withIntermediateDirectories: true
            )
        }
        return runURL
    }

    static func runDirectory(runID: String) throws -> URL {
        try teacherDatasetsDirectory().appending(path: runID, directoryHint: .isDirectory)
    }

    static func latestIncompleteRunID() throws -> String? {
        let root = try teacherDatasetsDirectory()
        let urls = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )
        let manifests = urls.compactMap { url -> (URL, TeacherDatasetManifest, Date)? in
            let manifestURL = url.appending(path: "manifest.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder().decode(TeacherDatasetManifest.self, from: data) else {
                return nil
            }
            let modified = (try? manifestURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return (url, manifest, modified)
        }
        return manifests
            .filter { $0.1.status != "complete" && $0.1.completedJobs < $0.1.totalPlannedJobs }
            .sorted { $0.2 > $1.2 }
            .first?
            .1
            .runID
    }

    static func readManifest(from runURL: URL) throws -> TeacherDatasetManifest {
        let data = try Data(contentsOf: runURL.appending(path: "manifest.json"))
        return try JSONDecoder().decode(TeacherDatasetManifest.self, from: data)
    }

    static func writeManifest(_ manifest: TeacherDatasetManifest, to runURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: runURL.appending(path: "manifest.json"), options: .atomic)
    }

    static func appendMetadata(_ metadata: TeacherDatasetMetadataLine, to runURL: URL) throws {
        try appendJSONLine(metadata, fileName: "metadata.jsonl", to: runURL)
    }

    static func appendFailedJob(_ failed: FailedJobLine, to runURL: URL) throws {
        try appendJSONLine(failed, fileName: "failed_jobs.jsonl", to: runURL)
    }

    private static func appendJSONLine<T: Encodable>(_ value: T, fileName: String, to runURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        let url = runURL.appending(path: fileName)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.write(contentsOf: Data("\n".utf8))
        try handle.synchronize()
    }

    static func completedIDs(in runURL: URL) -> Set<String> {
        let metadataURL = runURL.appending(path: "metadata.jsonl")
        guard let text = try? String(contentsOf: metadataURL, encoding: .utf8) else { return [] }
        return Set(text.split(separator: "\n").compactMap { line in
            guard let data = String(line).data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = object["id"] as? String else {
                return nil
            }
            return id
        })
    }

    static func imageFilesExist(for jobID: String, in runURL: URL) -> Bool {
        ["images_512", "images_256", "images_128", "images_64"].allSatisfy { folder in
            FileManager.default.fileExists(atPath: runURL.appending(path: folder).appending(path: "\(jobID).png").path)
        }
    }

    static func saveImages(_ image: UIImage, baseName: String, to runURL: URL) throws -> SavedImageResult {
        let paths = TeacherDatasetSavedImages(
            size512: "images_512/\(baseName).png",
            size256: "images_256/\(baseName).png",
            size128: "images_128/\(baseName).png",
            size64: "images_64/\(baseName).png"
        )
        let urls = [
            runURL.appending(path: paths.size512),
            runURL.appending(path: paths.size256),
            runURL.appending(path: paths.size128),
            runURL.appending(path: paths.size64),
        ]
        try savePNG(image, to: urls[0])
        try savePNG(resize(image, sideLength: 256), to: urls[1])
        try savePNG(resize(image, sideLength: 128), to: urls[2])
        try savePNG(resize(image, sideLength: 64), to: urls[3])
        let bytes = urls.reduce(0) { total, url in
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return total + size
        }
        return SavedImageResult(paths: paths, totalFileSizeBytes: bytes)
    }

    static func writeContactSheet(runURL: URL, imageFolder: String, sideLength: Int, maxImages: Int, fileName: String, labeled: Bool) {
        let folder = runURL.appending(path: imageFolder, directoryHint: .isDirectory)
        guard let imageURLs = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .filter({ $0.pathExtension == "png" })
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            .prefix(maxImages),
            !imageURLs.isEmpty else {
            return
        }
        let columns = labeled ? 6 : 10
        let labelHeight = labeled ? 22 : 0
        let cell = sideLength
        let rows = Int(ceil(Double(imageURLs.count) / Double(columns)))
        let size = CGSize(width: columns * cell, height: rows * (cell + labelHeight))
        let renderer = UIGraphicsImageRenderer(size: size)
        let sheet = renderer.image { context in
            UIColor.white.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            for (index, url) in imageURLs.enumerated() {
                guard let image = UIImage(contentsOfFile: url.path) else { continue }
                let x = (index % columns) * cell
                let y = (index / columns) * (cell + labelHeight)
                image.draw(in: CGRect(x: x, y: y, width: cell, height: cell))
                if labeled {
                    let name = url.deletingPathExtension().lastPathComponent
                    let label = name.split(separator: "_").first.map(String.init) ?? name
                    let paragraph = NSMutableParagraphStyle()
                    paragraph.alignment = .center
                    let attributes: [NSAttributedString.Key: Any] = [
                        .font: UIFont.monospacedSystemFont(ofSize: 10, weight: .regular),
                        .foregroundColor: UIColor.black,
                        .paragraphStyle: paragraph,
                    ]
                    label.draw(
                        in: CGRect(x: x, y: y + cell + 3, width: cell, height: labelHeight - 3),
                        withAttributes: attributes
                    )
                }
            }
        }
        let reportsURL = runURL.appending(path: "reports", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: reportsURL, withIntermediateDirectories: true)
        try? sheet.pngData()?.write(to: reportsURL.appending(path: fileName), options: .atomic)
    }

    static func writeSummary(manifest: TeacherDatasetManifest, runURL: URL, elapsedSamples: [Double]) {
        let reportsURL = runURL.appending(path: "reports", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: reportsURL, withIntermediateDirectories: true)
        let averageElapsed = elapsedSamples.isEmpty ? nil : elapsedSamples.reduce(0, +) / Double(elapsedSamples.count)
        let summary: [String: Any?] = [
            "run_id": manifest.runID,
            "model_family": manifest.modelFamily,
            "status": manifest.status,
            "preset": manifest.preset.rawValue,
            "completed_jobs": manifest.completedJobs,
            "failed_jobs": manifest.failedJobs,
            "total_planned_jobs": manifest.totalPlannedJobs,
            "source_resolution": manifest.sourceResolution,
            "prompt_set_version": manifest.promptSetVersion,
            "average_recent_elapsed_seconds": averageElapsed,
            "generated_at_unix": Int(Date().timeIntervalSince1970),
        ]
        let jsonObject = summary.compactMapValues { $0 }
        if let data = try? JSONSerialization.data(withJSONObject: jsonObject, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: reportsURL.appending(path: "summary.json"), options: .atomic)
        }
    }

    private static func savePNG(_ image: UIImage, to url: URL) throws {
        guard let data = image.pngData() else {
            throw TeacherDatasetError.imageEncodingFailed(url.path)
        }
        try data.write(to: url, options: .atomic)
    }

    private static func resize(_ image: UIImage, sideLength: CGFloat) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: sideLength, height: sideLength), format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: sideLength, height: sideLength))
        }
    }
}

private final class RuntimeResourceCompiler: @unchecked Sendable {
    private let fileManager = FileManager.default

    func prepareCompiledResources(publish: @escaping @Sendable (String) -> Void) throws -> URL {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw TeacherDatasetError.missingResources("Bundle.main.resourceURL")
        }
        let url = resourceURL
            .appending(path: "BundledResources", directoryHint: .isDirectory)
            .appending(path: "sd15", directoryHint: .isDirectory)
            .appending(path: "512", directoryHint: .isDirectory)
            .appending(path: "Resources", directoryHint: .isDirectory)
        guard fileManager.fileExists(atPath: url.path) else {
            throw TeacherDatasetError.missingResources(url.path)
        }
        for modelName in ["TextEncoder.mlmodelc", "Unet.mlmodelc", "VAEDecoder.mlmodelc"] {
            let modelURL = url.appending(path: modelName, directoryHint: .isDirectory)
            guard fileManager.fileExists(atPath: modelURL.path) else {
                throw TeacherDatasetError.missingResources(modelURL.path)
            }
        }
        for tokenFile in ["vocab.json", "merges.txt"] {
            let tokenURL = url.appending(path: tokenFile)
            guard fileManager.fileExists(atPath: tokenURL.path) else {
                throw TeacherDatasetError.missingResources(tokenURL.path)
            }
        }
        publish("Using bundled 512 SD 1.5 resources.")
        return url
    }
}

private final class PipelineCache: @unchecked Sendable {
    private let queue = DispatchQueue(label: "SD15TeacherDataset.pipeline-cache")
    private var cachedPipeline: StableDiffusionPipeline?
    private var cachedResourcesPath: String?

    func generateImage(
        resolution: ResolutionPreset,
        resourcesURL: URL,
        prompt: String,
        negativePrompt: String,
        stepCount: Int,
        seed: UInt32,
        guidanceScale: Float,
        schedulerOption: SchedulerOption,
        shouldContinue: @escaping @Sendable () -> Bool,
        publish: @escaping @Sendable (String) -> Void
    ) async throws -> UIImage {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    let startTime = Date()
                    func update(_ message: String) {
                        print("[TeacherDataset] \(message)")
                        publish(message)
                    }
                    let pipeline: StableDiffusionPipeline
                    if let cachedPipeline, cachedResourcesPath == resourcesURL.path {
                        pipeline = cachedPipeline
                        update("Reusing loaded pipeline.")
                    } else {
                        cachedPipeline?.unloadResources()
                        cachedPipeline = nil
                        cachedResourcesPath = nil
                        let config = MLModelConfiguration()
                        #if targetEnvironment(macCatalyst)
                        config.computeUnits = .all
                        #else
                        config.computeUnits = .cpuAndNeuralEngine
                        #endif
                        let newPipeline = try StableDiffusionPipeline(
                            resourcesAt: resourcesURL,
                            controlNet: [],
                            configuration: config,
                            disableSafety: false,
                            reduceMemory: true
                        )
                        try newPipeline.loadResources()
                        update("Loaded SD 1.5 pipeline.")
                        cachedPipeline = newPipeline
                        cachedResourcesPath = resourcesURL.path
                        pipeline = newPipeline
                    }

                    var generationConfig = PipelineConfiguration(prompt: prompt)
                    generationConfig.negativePrompt = negativePrompt
                    generationConfig.imageCount = 1
                    generationConfig.stepCount = stepCount
                    generationConfig.seed = seed
                    generationConfig.guidanceScale = guidanceScale
                    generationConfig.schedulerType = schedulerOption.stableDiffusionScheduler
                    generationConfig.disableSafety = false
                    generationConfig.useDenoisedIntermediates = false

                    let images = try pipeline.generateImages(configuration: generationConfig) { progress in
                        guard shouldContinue() else { return false }
                        let step = progress.step + 1
                        let elapsed = Date().timeIntervalSince(startTime)
                        update("Step \(step)/\(progress.stepCount), elapsed \(String(format: "%.1f", elapsed))s")
                        return true
                    }
                    guard let cgImage = images.first ?? nil else {
                        throw TeacherDatasetError.noImageReturned
                    }
                    continuation.resume(returning: UIImage(cgImage: cgImage))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

@MainActor
private final class TeacherDatasetViewModel: ObservableObject {
    @Published var selectedPreset: TeacherRunPreset = .pilot
    @Published var isRunning = false
    @Published var isPaused = false
    @Published var hasResumableRun = false
    @Published var statusTitle = "Ready"
    @Published var currentJobTitle = "Not started"
    @Published var completedJobs = 0
    @Published var totalJobs = 0
    @Published var failedCount = 0
    @Published var outputPath = ""
    @Published var latestLog = "No logs yet."
    @Published var lastElapsedSeconds: Double?
    @Published var lastImage: UIImage?
    @Published var stepCount = 25
    @Published var requireCharging = false
    @Published var pauseOnHighThermalState = true
    @Published var keepScreenAwake = true
    @Published var useDimProgressView = true
    @Published var isPostRunBlackout = false

    private let resourceCompiler = RuntimeResourceCompiler()
    private let pipelineCache = PipelineCache()
    private var runTask: Task<Void, Never>?
    private let stopFlag = LockedFlag()
    private let pauseFlag = LockedFlag()
    private var activeRunID: String?
    private var activePreset: TeacherRunPreset = .pilot
    private var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid
    private var elapsedSamples: [Double] = []
    private var postRunBlackoutTask: Task<Void, Never>?
    private let activeRunDefaultsKey = "TeacherDataset.activeRunID"
    private var didAttemptAutomation = false

    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        refreshResumableState()
    }

    var progressText: String {
        guard totalJobs > 0 else { return "0 / 0" }
        return "\(completedJobs) / \(totalJobs)"
    }

    var progressValue: Double {
        guard totalJobs > 0 else { return 0 }
        return Double(completedJobs) / Double(totalJobs)
    }

    var lastElapsedText: String {
        guard let lastElapsedSeconds else { return "None" }
        return String(format: "%.1fs", lastElapsedSeconds)
    }

    var estimatedRemainingText: String {
        guard totalJobs > completedJobs else { return "Done" }
        let average = elapsedSamples.isEmpty ? (lastElapsedSeconds ?? 55) : elapsedSamples.reduce(0, +) / Double(elapsedSamples.count)
        let seconds = average * Double(totalJobs - completedJobs)
        return Self.durationText(seconds)
    }

    var thermalStateText: String {
        Self.thermalStateText(ProcessInfo.processInfo.thermalState)
    }

    var batteryStateText: String {
        "\(Self.batteryStateText(UIDevice.current.batteryState)) \(Int(max(UIDevice.current.batteryLevel, 0) * 100))%"
    }

    func startSelectedPreset() {
        start(preset: selectedPreset, runID: Self.makeRunID(prefix: selectedPreset.rawValue))
    }

    func startAutomationIfNeeded() {
        guard !didAttemptAutomation, !isRunning else { return }
        didAttemptAutomation = true
        do {
            if let runID = try TeacherDatasetFileWriter.latestIncompleteRunID() {
                latestLog = "Auto-resuming incomplete run \(runID)."
                try resume(runID: runID)
            } else {
                selectedPreset = .pilot
                latestLog = "Auto-starting pilot run from fixed16_v1."
                start(preset: .pilot, runID: Self.makeRunID(prefix: "sd15_teacher_pilot"))
            }
        } catch {
            latestLog = "Automation setup failed: \(error.localizedDescription)"
            refreshResumableState()
        }
    }

    func resumeLatestRun() {
        guard !isRunning else { return }
        do {
            guard let runID = try TeacherDatasetFileWriter.latestIncompleteRunID() else {
                latestLog = "No incomplete run found."
                refreshResumableState()
                return
            }
            try resume(runID: runID)
        } catch {
            latestLog = error.localizedDescription
        }
    }

    func resume() {
        guard !isRunning else { return }
        if let activeRunID {
            try? resume(runID: activeRunID)
        } else {
            resumeLatestRun()
        }
    }

    func pause() {
        guard isRunning else { return }
        pauseFlag.set(true)
        isPaused = true
        statusTitle = "Pausing"
        latestLog = "Pause requested. The current image will finish or stop at the next safe checkpoint."
    }

    func stop() {
        stopFlag.set(true)
        pauseFlag.set(false)
        isPaused = false
        statusTitle = "Stopping"
        latestLog = "Stop requested."
        if !isRunning {
            clearActiveRun()
            refreshResumableState()
        }
    }

    func handleScenePhase(_ phase: ScenePhase) {
        switch phase {
        case .active:
            endBackgroundTask()
        case .background:
            guard isRunning else { return }
            beginBackgroundTaskForCurrentJob()
        case .inactive:
            break
        @unknown default:
            break
        }
    }

    private func start(preset: TeacherRunPreset, runID: String) {
        guard !isRunning else { return }
        cancelPostRunBlackout()
        isPostRunBlackout = false
        stopFlag.set(false)
        pauseFlag.set(false)
        isPaused = false
        isRunning = true
        activeRunID = runID
        activePreset = preset
        selectedPreset = preset
        UserDefaults.standard.set(runID, forKey: activeRunDefaultsKey)
        UIApplication.shared.isIdleTimerDisabled = keepScreenAwake

        let jobs = Self.makeJobs(for: preset)
        totalJobs = jobs.count
        completedJobs = 0
        failedCount = 0
        outputPath = ""
        lastElapsedSeconds = nil
        lastImage = nil
        elapsedSamples = []
        statusTitle = "Preparing"
        currentJobTitle = "Preparing resources"
        latestLog = "Starting \(preset.displayName) run \(runID)."
        runTask = Task { [weak self] in
            await self?.run(preset: preset, jobs: jobs, runID: runID, resumeExisting: false)
        }
    }

    private func resume(runID: String) throws {
        cancelPostRunBlackout()
        isPostRunBlackout = false
        let runURL = try TeacherDatasetFileWriter.runDirectory(runID: runID)
        let manifest = try TeacherDatasetFileWriter.readManifest(from: runURL)
        let jobs = Self.makeJobs(for: manifest.preset)
        stopFlag.set(false)
        pauseFlag.set(false)
        isPaused = false
        isRunning = true
        activeRunID = runID
        activePreset = manifest.preset
        selectedPreset = manifest.preset
        stepCount = manifest.generationSettings.steps
        UserDefaults.standard.set(runID, forKey: activeRunDefaultsKey)
        UIApplication.shared.isIdleTimerDisabled = keepScreenAwake
        totalJobs = jobs.count
        completedJobs = manifest.completedJobs
        failedCount = manifest.failedJobs
        outputPath = runURL.path
        statusTitle = "Resuming"
        latestLog = "Resuming \(runID)."
        runTask = Task { [weak self] in
            await self?.run(preset: manifest.preset, jobs: jobs, runID: runID, resumeExisting: true)
        }
    }

    private func run(preset: TeacherRunPreset, jobs: [TeacherGenerationJob], runID: String, resumeExisting: Bool) async {
        do {
            try await runThrowing(preset: preset, jobs: jobs, runID: runID, resumeExisting: resumeExisting)
        } catch {
            await MainActor.run {
                latestLog = error.localizedDescription
                statusTitle = "Failed"
                finish(clearRun: false)
            }
        }
    }

    private func runThrowing(preset: TeacherRunPreset, jobs: [TeacherGenerationJob], runID: String, resumeExisting: Bool) async throws {
        let resolution = ResolutionPreset.p512
        let steps = stepCount
        let guidanceScale: Float = 7.0
        let scheduler = SchedulerOption.dpmpp2MKarras
        let negativePrompt = "text, logo, watermark, caption, low quality, blurry, distorted, deformed, cluttered background, multiple subjects, cropped subject, extra limbs"
        let resourcesURL = try await prepareResources()
        let runURL = try TeacherDatasetFileWriter.makeRunDirectory(runID: runID)
        var manifest: TeacherDatasetManifest
        if resumeExisting, let existing = try? TeacherDatasetFileWriter.readManifest(from: runURL) {
            manifest = existing
            manifest.status = "running"
        } else {
            manifest = makeManifest(
                runID: runID,
                preset: preset,
                resourcesURL: resourcesURL,
                totalJobs: jobs.count,
                steps: steps,
                guidanceScale: guidanceScale,
                scheduler: scheduler,
                negativePrompt: negativePrompt,
                resolution: resolution
            )
        }
        var completedIDs = TeacherDatasetFileWriter.completedIDs(in: runURL)
        completedIDs = completedIDs.filter { TeacherDatasetFileWriter.imageFilesExist(for: $0, in: runURL) }
        manifest.completedJobs = completedIDs.count
        try TeacherDatasetFileWriter.writeManifest(manifest, to: runURL)

        await MainActor.run {
            outputPath = runURL.path
            totalJobs = jobs.count
            completedJobs = manifest.completedJobs
            failedCount = manifest.failedJobs
            statusTitle = "Running"
            latestLog = "Output: \(runURL.path)"
        }

        for (index, job) in jobs.enumerated() {
            if completedIDs.contains(job.id) || TeacherDatasetFileWriter.imageFilesExist(for: job.id, in: runURL) {
                if !completedIDs.contains(job.id) {
                    completedIDs.insert(job.id)
                    manifest.completedJobs = completedIDs.count
                    try TeacherDatasetFileWriter.writeManifest(manifest, to: runURL)
                }
                continue
            }
            if stopFlag.get() || pauseFlag.get() { break }
            if shouldPauseForBatteryOrThermal() {
                pauseFlag.set(true)
                await MainActor.run {
                    isPaused = true
                    statusTitle = "Paused"
                    latestLog = "Paused by battery or thermal policy."
                }
                break
            }

            let start = Date()
            print("[TeacherDataset] start \(job.id) seed=\(job.seed)")
            await MainActor.run {
                currentJobTitle = "\(job.title) \(job.variant) seed \(job.seed)"
                latestLog = "Starting \(index + 1)/\(jobs.count): \(job.id)"
            }

            do {
                let image = try await pipelineCache.generateImage(
                    resolution: resolution,
                    resourcesURL: resourcesURL,
                    prompt: job.prompt,
                    negativePrompt: negativePrompt,
                    stepCount: steps,
                    seed: job.seed,
                    guidanceScale: guidanceScale,
                    schedulerOption: scheduler,
                    shouldContinue: { [stopFlag] in !stopFlag.get() },
                    publish: { [weak self] message in
                        Task { @MainActor in self?.latestLog = message }
                    }
                )
                if stopFlag.get() { break }

                let elapsed = Date().timeIntervalSince(start)
                let saved = try TeacherDatasetFileWriter.saveImages(image, baseName: job.id, to: runURL)
                let metadata = TeacherDatasetMetadataLine(
                    id: job.id,
                    key: job.key,
                    title: job.title,
                    variant: job.variant,
                    prompt: job.prompt,
                    negativePrompt: negativePrompt,
                    seed: job.seed,
                    modelFamily: "sd15",
                    steps: steps,
                    guidanceScale: guidanceScale,
                    scheduler: scheduler.rawValue,
                    sourceWidth: resolution.pixelSize,
                    sourceHeight: resolution.pixelSize,
                    savedImages: saved.paths,
                    accepted: true,
                    qcFlags: job.promptQCFlags,
                    rejectReason: nil,
                    imageMode: "png_512_256_128_64",
                    fileSizeBytes: saved.totalFileSizeBytes,
                    elapsedSeconds: elapsed,
                    thermalState: Self.thermalStateText(ProcessInfo.processInfo.thermalState),
                    batteryLevel: UIDevice.current.batteryLevel,
                    batteryState: Self.batteryStateText(UIDevice.current.batteryState),
                    createdAtUnix: Int(Date().timeIntervalSince1970)
                )
                try TeacherDatasetFileWriter.appendMetadata(metadata, to: runURL)
                completedIDs.insert(job.id)
                manifest.completedJobs = completedIDs.count
                manifest.status = "running"
                try TeacherDatasetFileWriter.writeManifest(manifest, to: runURL)
                print("[TeacherDataset] end \(job.id) elapsed=\(String(format: "%.1f", elapsed))s")

                await MainActor.run {
                    completedJobs = manifest.completedJobs
                    lastElapsedSeconds = elapsed
                    elapsedSamples.append(elapsed)
                    if elapsedSamples.count > 20 { elapsedSamples.removeFirst(elapsedSamples.count - 20) }
                    lastImage = image
                    latestLog = "Saved \(job.id)."
                }
            } catch {
                let failed = FailedJobLine(
                    id: job.id,
                    key: job.key,
                    variant: job.variant,
                    seed: job.seed,
                    error: error.localizedDescription,
                    createdAtUnix: Int(Date().timeIntervalSince1970)
                )
                try? TeacherDatasetFileWriter.appendFailedJob(failed, to: runURL)
                manifest.failedJobs += 1
                manifest.status = "running"
                try TeacherDatasetFileWriter.writeManifest(manifest, to: runURL)
                await MainActor.run {
                    failedCount = manifest.failedJobs
                    latestLog = "Failed \(job.id): \(error.localizedDescription)"
                }
            }
        }

        if manifest.completedJobs >= manifest.totalPlannedJobs {
            manifest.status = "complete"
            manifest.endedAtUnix = Int(Date().timeIntervalSince1970)
            clearActiveRun()
        } else if pauseFlag.get() {
            manifest.status = "paused"
        } else if stopFlag.get() {
            manifest.status = "stopped"
        } else {
            manifest.status = "interrupted"
        }
        manifest.completedJobs = completedIDs.count
        try TeacherDatasetFileWriter.writeManifest(manifest, to: runURL)
        TeacherDatasetFileWriter.writeContactSheet(
            runURL: runURL,
            imageFolder: "images_64",
            sideLength: 64,
            maxImages: Int.max,
            fileName: "contact_sheet_64_all.png",
            labeled: false
        )
        TeacherDatasetFileWriter.writeContactSheet(
            runURL: runURL,
            imageFolder: "images_128",
            sideLength: 128,
            maxImages: min(manifest.completedJobs, 512),
            fileName: "contact_sheet_128_labeled.png",
            labeled: true
        )
        TeacherDatasetFileWriter.writeSummary(manifest: manifest, runURL: runURL, elapsedSamples: elapsedSamples)

        await MainActor.run {
            statusTitle = manifest.status.capitalized
            currentJobTitle = "\(manifest.completedJobs) / \(jobs.count) saved"
            latestLog = "\(statusTitle): \(runURL.path)"
            finish(clearRun: manifest.status == "complete")
        }
    }

    private func prepareResources() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [resourceCompiler] in
                do {
                    let url = try resourceCompiler.prepareCompiledResources { message in
                        print("[TeacherDataset] \(message)")
                    }
                    continuation.resume(returning: url)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func shouldPauseForBatteryOrThermal() -> Bool {
        if pauseOnHighThermalState {
            let state = ProcessInfo.processInfo.thermalState
            if state == .serious || state == .critical { return true }
        }
        if requireCharging {
            let state = UIDevice.current.batteryState
            if state != .charging && state != .full { return true }
        }
        return false
    }

    private func beginBackgroundTaskForCurrentJob() {
        guard backgroundTaskID == .invalid else { return }
        backgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: "TeacherDatasetCurrentJob") { [weak self] in
            Task { @MainActor in
                self?.latestLog = "Background time expired. Checkpoint is safe; run can resume later."
                self?.stopFlag.set(true)
                self?.endBackgroundTask()
            }
        }
        latestLog = "Entered background. Using finite background task only to finish or checkpoint the current job."
    }

    private func endBackgroundTask() {
        guard backgroundTaskID != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTaskID)
        backgroundTaskID = .invalid
    }

    private func makeManifest(
        runID: String,
        preset: TeacherRunPreset,
        resourcesURL: URL,
        totalJobs: Int,
        steps: Int,
        guidanceScale: Float,
        scheduler: SchedulerOption,
        negativePrompt: String,
        resolution: ResolutionPreset
    ) -> TeacherDatasetManifest {
        TeacherDatasetManifest(
            runID: runID,
            preset: preset,
            appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "",
            appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "",
            modelFamily: "sd15",
            resourceDirectory: resourcesURL.path,
            sourceResolution: resolution.pixelSize,
            totalPlannedJobs: totalJobs,
            completedJobs: 0,
            failedJobs: 0,
            startedAtUnix: Int(Date().timeIntervalSince1970),
            endedAtUnix: nil,
            status: "running",
            promptSetVersion: "fixed16_v1",
            generationSettings: TeacherDatasetGenerationSettings(
                steps: steps,
                guidanceScale: guidanceScale,
                scheduler: scheduler.rawValue,
                negativePrompt: negativePrompt,
                sourceWidth: resolution.pixelSize,
                sourceHeight: resolution.pixelSize,
                reduceMemory: true
            ),
            deviceName: UIDevice.current.name
        )
    }

    private func finish(clearRun: Bool) {
        endBackgroundTask()
        UIApplication.shared.isIdleTimerDisabled = false
        isRunning = false
        runTask = nil
        stopFlag.set(false)
        pauseFlag.set(false)
        isPaused = false
        if clearRun { clearActiveRun() }
        refreshResumableState()
        schedulePostRunBlackout()
    }

    func dismissPostRunBlackout() {
        cancelPostRunBlackout()
        isPostRunBlackout = false
    }

    private func schedulePostRunBlackout() {
        cancelPostRunBlackout()
        guard useDimProgressView else { return }
        postRunBlackoutTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: 60 * 1_000_000_000)
            } catch {
                return
            }
            await MainActor.run {
                guard let self, !self.isRunning else { return }
                self.isPostRunBlackout = true
            }
        }
    }

    private func cancelPostRunBlackout() {
        postRunBlackoutTask?.cancel()
        postRunBlackoutTask = nil
    }

    private func clearActiveRun() {
        activeRunID = nil
        UserDefaults.standard.removeObject(forKey: activeRunDefaultsKey)
    }

    private func refreshResumableState() {
        let stored = UserDefaults.standard.string(forKey: activeRunDefaultsKey)
        activeRunID = stored
        hasResumableRun = stored != nil || ((try? TeacherDatasetFileWriter.latestIncompleteRunID()) != nil)
    }

    private static func makeJobs(for preset: TeacherRunPreset) -> [TeacherGenerationJob] {
        let selected = Array(promptPresets.prefix(preset.categoryCount))
        return selected.flatMap { promptPreset in
            Array(promptPreset.variants.prefix(preset.variantsPerCategory)).flatMap { variant in
                (0..<preset.seedsPerVariant).map { seed in
                    TeacherGenerationJob(
                        id: "\(promptPreset.key)_\(variant.variant)_seed\(String(format: "%06d", seed))",
                        key: promptPreset.key,
                        title: promptPreset.title,
                        variant: variant.variant,
                        prompt: variant.prompt,
                        promptQCFlags: variant.qcFlags,
                        seed: UInt32(seed)
                    )
                }
            }
        }
    }

    private static func makeRunID(prefix: String) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return "\(prefix)_\(formatter.string(from: Date()))"
    }

    private static func durationText(_ seconds: Double) -> String {
        let total = max(Int(seconds), 0)
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        if hours > 0 { return "\(hours)h \(minutes)m" }
        return "\(minutes)m"
    }

    private static func thermalStateText(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func batteryStateText(_ state: UIDevice.BatteryState) -> String {
        switch state {
        case .unknown: return "unknown"
        case .unplugged: return "unplugged"
        case .charging: return "charging"
        case .full: return "full"
        @unknown default: return "unknown"
        }
    }

    private static let promptPresets: [TeacherPromptPreset] = TeacherPromptCatalog.presets
}

private enum TeacherPromptCatalog {
    static let categoryData: [(String, String, String, [String])] = [
        ("apple", "Apple", "red apple fruit with stem and leaf", ["possible_brand_shape"]),
        ("bird", "Bird", "small bird", []),
        ("car", "Car", "small toy car vehicle", ["avoid_people"]),
        ("castle", "Castle", "simple castle", []),
        ("cat", "Cat", "cute cat", []),
        ("dog", "Dog", "cute dog", []),
        ("face", "Face", "single round human face only", ["face_may_be_multiple"]),
        ("fish", "Fish", "single fish", []),
        ("flower", "Flower", "single flower", []),
        ("house", "House", "small house", []),
        ("moon", "Moon", "moon shape", []),
        ("robot", "Robot", "small robot", []),
        ("star", "Star", "single five pointed star", []),
        ("sun", "Sun", "single yellow sun", ["may_generate_multiple_icons"]),
        ("train", "Train", "toy train engine", []),
        ("tree", "Tree", "single tree", []),
        ("book", "Book", "closed book", []),
        ("chair", "Chair", "simple chair", []),
        ("clock", "Clock", "round wall clock", []),
        ("cup", "Cup", "simple cup", []),
        ("airplane", "Airplane", "small airplane", []),
        ("boat", "Boat", "small boat", []),
        ("bus", "Bus", "simple bus", []),
        ("bicycle", "Bicycle", "simple bicycle", []),
        ("mushroom", "Mushroom", "single mushroom", []),
        ("mountain", "Mountain", "single mountain peak", []),
        ("cloud", "Cloud", "single fluffy cloud", []),
        ("heart", "Heart", "single heart shape", []),
        ("ball", "Ball", "single colored ball", []),
        ("guitar", "Guitar", "simple acoustic guitar", []),
        ("camera", "Camera", "simple camera", []),
        ("shoe", "Shoe", "single shoe", []),
        ("key", "Key", "single key", []),
        ("umbrella", "Umbrella", "open umbrella", []),
        ("lamp", "Lamp", "simple table lamp", []),
        ("bottle", "Bottle", "simple bottle", []),
        ("pencil", "Pencil", "single pencil", []),
        ("cake", "Cake", "simple cake", []),
        ("leaf", "Leaf", "single green leaf", []),
        ("snowman", "Snowman", "simple snowman", []),
        ("rocket", "Rocket", "small rocket", []),
        ("butterfly", "Butterfly", "single butterfly", []),
        ("turtle", "Turtle", "single turtle", []),
        ("rabbit", "Rabbit", "cute rabbit", []),
        ("bear", "Bear", "simple bear", []),
        ("whale", "Whale", "single whale", []),
        ("ship", "Ship", "simple ship", []),
        ("truck", "Truck", "simple truck", []),
        ("crown", "Crown", "single crown", []),
        ("diamond", "Diamond", "single diamond gemstone", []),
        ("bell", "Bell", "single bell", []),
        ("door", "Door", "single door", []),
        ("window", "Window", "single window", []),
        ("bridge", "Bridge", "simple bridge", []),
        ("tower", "Tower", "simple tower", []),
        ("tent", "Tent", "single camping tent", []),
        ("fire", "Fire", "single flame", []),
        ("snowflake", "Snowflake", "single snowflake", []),
        ("planet", "Planet", "single planet", []),
        ("ring", "Ring", "single ring", []),
        ("hammer", "Hammer", "single hammer", []),
        ("sword", "Sword", "single sword", []),
        ("shield", "Shield", "single shield", []),
        ("drum", "Drum", "simple drum", []),
        ("piano", "Piano", "simple piano keyboard", []),
    ]

    static let promptGuards: [String: String] = [
        "apple": "fruit only, no apple logo shape, no bite mark, no repeating pattern",
        "car": "vehicle only, no people, no driver, no buildings, no street scene",
    ]

    static let presets: [TeacherPromptPreset] = categoryData.map { key, title, subject, flags in
        let guardText = promptGuards[key].map { ", \($0)" } ?? ""
        let variants = [
            TeacherPromptVariant(
                variant: "v00",
                prompt: "\(subject), centered subject, single subject, simple clean background, readable silhouette, clean illustration\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
            TeacherPromptVariant(
                variant: "v01",
                prompt: "\(subject) front view, centered subject, single subject, plain background, clear simple shape, clean illustration\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
            TeacherPromptVariant(
                variant: "v02",
                prompt: "\(subject) side view, centered subject, single subject, plain background, readable silhouette, toy-like 3D render\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
            TeacherPromptVariant(
                variant: "v03",
                prompt: "\(subject) object, centered subject, single subject only, blank background, bold silhouette, clean low detail illustration\(guardText), no text, no logo, no watermark",
                qcFlags: flags + (key == "sun" ? ["avoid_icon_collection"] : [])
            ),
            TeacherPromptVariant(
                variant: "v04",
                prompt: "\(subject) as a simple toy object, centered subject, single subject, simple clean background, chunky readable shape, toy-like 3D render\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
            TeacherPromptVariant(
                variant: "v05",
                prompt: "\(subject), icon-like composition, centered, full object visible, single subject, plain light background, readable at small size\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
            TeacherPromptVariant(
                variant: "v06",
                prompt: "\(subject), simple cartoon style, centered subject, full object visible, uncluttered background, strong silhouette\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
            TeacherPromptVariant(
                variant: "v07",
                prompt: "\(subject), clean vector-like illustration, centered single subject, plain background, full object visible, low detail readable shape\(guardText), no text, no logo, no watermark",
                qcFlags: flags
            ),
        ]
        return TeacherPromptPreset(key: key, title: title, subject: subject, variants: variants)
    }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func set(_ newValue: Bool) {
        lock.lock()
        value = newValue
        lock.unlock()
    }

    func get() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private enum TeacherDatasetError: LocalizedError {
    case missingResources(String)
    case noImageReturned
    case imageEncodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingResources(let path):
            return "Missing model resources: \(path)"
        case .noImageReturned:
            return "Generation completed but no image was returned."
        case .imageEncodingFailed(let path):
            return "Could not encode PNG: \(path)"
        }
    }
}
