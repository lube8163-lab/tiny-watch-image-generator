import Foundation
import TinyWatchGenerator

private struct EvalSuite: Decodable {
    let version: Int
    let groups: [EvalGroup]
}

private struct EvalGroup: Decodable {
    let key: String
    let prompts: [String]
}

private struct EvalManifest: Encodable {
    let generatedAt: String
    let config: String
    let outDir: String
    let size: Int
    let postprocess: String
    let seeds: [UInt64]
    let groups: [GroupResult]
}

private struct GroupResult: Encodable {
    let key: String
    let entries: [ImageEntry]
}

private struct ImageEntry: Encodable {
    let group: String
    let promptIndex: Int
    let prompt: String
    let seed: UInt64
    let image: String
    let elapsedMs: Int
}

private struct Options {
    var config = "configs/prompt_eval_suite.json"
    var outDir = "reports/watch_eval/current"
    var seeds: [UInt64] = [0]
    var size = TinyImageGenerator.defaultSize
    var groups: Set<String>?
    var promptsPerGroup: Int?
    var postprocess = TinyImagePostprocess.watchDenoise
}

private enum EvalError: Error, CustomStringConvertible {
    case missingValue(String)
    case invalidValue(String)

    var description: String {
        switch self {
        case .missingValue(let flag):
            return "Missing value for \(flag)"
        case .invalidValue(let message):
            return message
        }
    }
}

private func parseOptions() throws -> Options {
    var options = Options()
    var args = Array(CommandLine.arguments.dropFirst())

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--config":
            options.config = try takeValue(flag, from: &args)
        case "--out-dir":
            options.outDir = try takeValue(flag, from: &args)
        case "--seeds":
            options.seeds = try parseSeeds(takeValue(flag, from: &args))
        case "--size":
            guard let size = Int(try takeValue(flag, from: &args)), size > 0 else {
                throw EvalError.invalidValue("--size must be a positive integer")
            }
            options.size = size
        case "--groups":
            let value = try takeValue(flag, from: &args)
            if value.lowercased() == "all" {
                options.groups = nil
            } else {
                options.groups = Set(splitCSV(value))
            }
        case "--prompts-per-group":
            guard let count = Int(try takeValue(flag, from: &args)), count >= 0 else {
                throw EvalError.invalidValue("--prompts-per-group must be zero or greater")
            }
            options.promptsPerGroup = count == 0 ? nil : count
        case "--raw":
            options.postprocess = .none
        case "--help", "-h":
            printHelp()
            exit(0)
        default:
            throw EvalError.invalidValue("Unknown argument: \(flag)")
        }
    }

    guard !options.seeds.isEmpty else {
        throw EvalError.invalidValue("--seeds must contain at least one integer")
    }
    return options
}

private func takeValue(_ flag: String, from args: inout [String]) throws -> String {
    guard !args.isEmpty else {
        throw EvalError.missingValue(flag)
    }
    return args.removeFirst()
}

private func parseSeeds(_ value: String) throws -> [UInt64] {
    try splitCSV(value).map { item in
        guard let seed = UInt64(item) else {
            throw EvalError.invalidValue("Invalid seed: \(item)")
        }
        return seed
    }
}

private func splitCSV(_ value: String) -> [String] {
    value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
}

private func printHelp() {
    print(
        """
        Usage:
          swift run TinyWatchEval [options]

        Options:
          --config PATH              Eval suite JSON. Default: configs/prompt_eval_suite.json
          --out-dir PATH             Output directory. Default: reports/watch_eval/current
          --seeds 0,7                Comma-separated seeds. Default: 0
          --size 128                 Output size. Default: 128
          --groups core_nouns,styles Comma-separated group keys, or all. Default: all
          --prompts-per-group 4      Limit prompts per selected group. 0 means all.
          --raw                      Disable watch postprocess.
        """
    )
}

private func loadSuite(path: String) throws -> EvalSuite {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(EvalSuite.self, from: data)
}

private func filteredGroups(from suite: EvalSuite, options: Options) -> [EvalGroup] {
    suite.groups.compactMap { group in
        if let selected = options.groups, !selected.contains(group.key) {
            return nil
        }
        if let limit = options.promptsPerGroup {
            return EvalGroup(key: group.key, prompts: Array(group.prompts.prefix(limit)))
        }
        return group
    }
}

private func writePPM(_ image: TinyGeneratedImage, to url: URL) throws {
    var data = Data("P6\n\(image.width) \(image.height)\n255\n".utf8)
    data.reserveCapacity(data.count + image.width * image.height * 3)
    for index in stride(from: 0, to: image.rgba.count, by: 4) {
        data.append(image.rgba[index])
        data.append(image.rgba[index + 1])
        data.append(image.rgba[index + 2])
    }
    try data.write(to: url)
}

private func imageFileName(group: String, prompt: String, promptIndex: Int, seed: UInt64) -> String {
    let slug = asciiSlug(prompt)
    let promptPart = slug.isEmpty ? "prompt\(promptIndex)" : slug
    return "\(group)_\(String(format: "%03d", promptIndex))_\(promptPart)_s\(seed).ppm"
}

private func asciiSlug(_ value: String) -> String {
    let lower = value.lowercased()
    var output = ""
    var previousWasDash = false

    for scalar in lower.unicodeScalars {
        let isAlnum = (scalar.value >= 48 && scalar.value <= 57) ||
            (scalar.value >= 97 && scalar.value <= 122)
        if isAlnum {
            output.unicodeScalars.append(scalar)
            previousWasDash = false
        } else if !previousWasDash {
            output.append("-")
            previousWasDash = true
        }
        if output.count >= 60 {
            break
        }
    }

    return output.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
}

private func relativePath(from base: URL, to url: URL) -> String {
    let basePath = base.standardizedFileURL.path
    let path = url.standardizedFileURL.path
    if path.hasPrefix(basePath + "/") {
        return String(path.dropFirst(basePath.count + 1))
    }
    return path
}

private func writeManifest(_ manifest: EvalManifest, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(manifest)
    try data.write(to: url)
}

private func run() throws {
    let options = try parseOptions()
    let suite = try loadSuite(path: options.config)
    let groups = filteredGroups(from: suite, options: options)
    if groups.isEmpty {
        throw EvalError.invalidValue("No eval groups selected")
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let outURL = URL(fileURLWithPath: options.outDir, relativeTo: root).standardizedFileURL
    let imagesURL = outURL.appendingPathComponent("images", isDirectory: true)
    try FileManager.default.createDirectory(at: imagesURL, withIntermediateDirectories: true)

    let generator = TinyImageGenerator()
    var results: [GroupResult] = []

    for group in groups {
        var entries: [ImageEntry] = []
        let groupURL = imagesURL.appendingPathComponent(group.key, isDirectory: true)
        try FileManager.default.createDirectory(at: groupURL, withIntermediateDirectories: true)

        for (promptIndex, prompt) in group.prompts.enumerated() {
            for seed in options.seeds {
                let start = ContinuousClock.now
                let image = generator.generate(
                    prompt: prompt,
                    seed: seed,
                    size: options.size,
                    postprocess: options.postprocess
                )
                let elapsed = start.duration(to: .now)
                let elapsedMs = Int(
                    (
                        Double(elapsed.components.seconds) * 1000.0 +
                        Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
                    ).rounded()
                )

                let imageURL = groupURL.appendingPathComponent(
                    imageFileName(group: group.key, prompt: prompt, promptIndex: promptIndex, seed: seed)
                )
                try writePPM(image, to: imageURL)
                entries.append(
                    ImageEntry(
                        group: group.key,
                        promptIndex: promptIndex,
                        prompt: prompt,
                        seed: seed,
                        image: relativePath(from: outURL, to: imageURL),
                        elapsedMs: elapsedMs
                    )
                )
                print("\(group.key) | \(prompt) | seed \(seed) | \(elapsedMs)ms")
            }
        }

        results.append(GroupResult(key: group.key, entries: entries))
    }

    let manifest = EvalManifest(
        generatedAt: ISO8601DateFormatter().string(from: Date()),
        config: options.config,
        outDir: outURL.path,
        size: options.size,
        postprocess: options.postprocess == .none ? "none" : "watchDenoise",
        seeds: options.seeds,
        groups: results
    )
    let manifestURL = outURL.appendingPathComponent("manifest.json")
    try writeManifest(manifest, to: manifestURL)
    print(manifestURL.path)
}

do {
    try run()
} catch {
    fputs("TinyWatchEval error: \(error)\n", stderr)
    exit(1)
}
