import CoreML
import Foundation
import os
import SwiftUI

struct StressLogLine: Identifiable {
    let id = UUID()
    let text: String
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
                modelElapsed += try predict(model: unet.model, label: unet.label, iteration: step)
                await Task.yield()
            }
            modelElapsed += try predict(model: decoder.model, label: decoder.label, iteration: 1)
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
    private func predict(model: MLModel, label: String, iteration: Int) throws -> TimeInterval {
        let input = try makeInputProvider(for: model)
        let start = Date()
        let output = try model.prediction(from: input)
        let elapsed = Date().timeIntervalSince(start)
        let outputs = output.featureNames.sorted().joined(separator: ",")
        log("predict: \(label) #\(iteration) \(format(seconds: elapsed)) outputs=[\(outputs)]")
        return elapsed
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
                    ForEach(viewModel.logLines.prefix(12)) { line in
                        Text(line.text)
                            .font(.system(size: 9, design: .monospaced))
                            .lineLimit(3)
                    }
                }
            }
            .navigationTitle("Stress")
            .onAppear {
                viewModel.scanModels()
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

    var errorDescription: String? {
        switch self {
        case .unsupportedInput(let name):
            return "unsupported input: \(name)"
        }
    }
}
