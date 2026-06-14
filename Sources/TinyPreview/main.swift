import Foundation
import TinyWatchGenerator

let args = Array(CommandLine.arguments.dropFirst())
let rawOutput = args.contains("--raw")
let filteredArgs = args.filter { $0 != "--raw" }
let seed = UInt64(filteredArgs.first ?? "1") ?? 1
let prompt = filteredArgs.dropFirst().joined(separator: " ")
let image = TinyImageGenerator().generate(
    prompt: prompt,
    seed: seed,
    postprocess: rawOutput ? .none : .watchDenoise
)

print("P3")
print("\(image.width) \(image.height)")
print("255")

for y in 0..<image.height {
    var row: [String] = []
    for x in 0..<image.width {
        let i = (y * image.width + x) * 4
        row.append("\(image.rgba[i]) \(image.rgba[i + 1]) \(image.rgba[i + 2])")
    }
    print(row.joined(separator: " "))
}
