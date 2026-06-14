import SwiftUI

struct WatchPromptPreset: Identifiable, Hashable {
    let key: String
    let title: String

    var id: String { key }
}

struct ContentView: View {
    private static let generationSize = 128
    private static let presets: [WatchPromptPreset] = [
        .init(key: "cat", title: "Cat"),
        .init(key: "dog", title: "Dog"),
        .init(key: "rabbit", title: "Rabbit"),
        .init(key: "horse", title: "Horse"),
        .init(key: "bear", title: "Bear"),
        .init(key: "fox", title: "Fox"),
        .init(key: "owl", title: "Owl"),
        .init(key: "butterfly", title: "Butterfly"),
        .init(key: "apple", title: "Apple"),
        .init(key: "banana", title: "Banana"),
        .init(key: "orange", title: "Orange"),
        .init(key: "strawberry", title: "Strawberry"),
        .init(key: "cake", title: "Cake"),
        .init(key: "pizza", title: "Pizza"),
        .init(key: "bread", title: "Bread"),
        .init(key: "robot", title: "Robot"),
        .init(key: "star", title: "Star"),
        .init(key: "sun", title: "Sun"),
        .init(key: "moon", title: "Moon"),
        .init(key: "car", title: "Car"),
        .init(key: "bus", title: "Bus"),
        .init(key: "bicycle", title: "Bicycle"),
        .init(key: "airplane", title: "Airplane"),
        .init(key: "boat", title: "Boat"),
        .init(key: "tree", title: "Tree"),
        .init(key: "mountain", title: "Mountain"),
        .init(key: "cloud", title: "Cloud"),
        .init(key: "flower", title: "Flower"),
        .init(key: "house", title: "House"),
        .init(key: "bird", title: "Bird"),
        .init(key: "fish", title: "Fish"),
        .init(key: "train", title: "Train"),
        .init(key: "castle", title: "Castle"),
        .init(key: "book", title: "Book"),
        .init(key: "chair", title: "Chair"),
        .init(key: "clock", title: "Clock"),
        .init(key: "cup", title: "Cup"),
        .init(key: "mushroom", title: "Mushroom"),
        .init(key: "heart", title: "Heart"),
        .init(key: "ball", title: "Ball"),
        .init(key: "guitar", title: "Guitar"),
        .init(key: "camera", title: "Camera"),
        .init(key: "shoe", title: "Shoe"),
        .init(key: "face", title: "Face")
    ]

    @State private var selectedPreset = Self.presets[0]
    @State private var promptText = Self.presets[0].key
    @State private var seed = Self.randomSeed()
    @State private var generatedImage = Self.placeholderImage()
    @State private var isGenerating = false
    @State private var status = "Tap Generate"

    private var promptPickerItems: [WatchPromptPreset] {
        var items = Self.presets
        let trimmedPrompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedPrompt.isEmpty,
           Self.preset(matching: trimmedPrompt) == nil,
           !items.contains(where: { $0.key == selectedPreset.key }) {
            items.append(selectedPreset)
        }
        return items
    }

    var body: some View {
        ScrollView(.vertical) {
            VStack(spacing: 8) {
                imageView

                Picker("Prompt", selection: $selectedPreset) {
                    ForEach(promptPickerItems) { preset in
                        Text(preset.title).tag(preset)
                    }
                }
                .labelsHidden()
                .frame(height: 36)
                .onChange(of: selectedPreset) {
                    promptText = selectedPreset.key
                }

                TextField("Prompt", text: $promptText)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .textInputAutocapitalization(.never)
                    .disableAutocorrection(true)
                    .onChange(of: promptText) {
                        syncSelectedPresetWithPrompt()
                    }

                HStack(spacing: 8) {
                    Button {
                        shufflePromptAndGenerate()
                    } label: {
                        buttonLabel(title: "Shuffle", systemImage: "shuffle")
                    }
                    .disabled(isGenerating)

                    Button {
                        randomizeAndGenerate()
                    } label: {
                        buttonLabel(title: "Generate", systemImage: "sparkles")
                    }
                    .disabled(isGenerating)
                }
                .buttonStyle(.bordered)

                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 4)
            .padding(.top, 18)
            .padding(.bottom, 10)
        }
    }

    private var imageView: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(.quaternary)

            if isGenerating {
                VStack(spacing: 7) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            } else {
                Image(decorative: cgImage(from: generatedImage), scale: 1)
                    .interpolation(.high)
                    .resizable()
                    .scaledToFit()
            }
        }
        .frame(width: 132, height: 132)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func buttonLabel(title: String, systemImage: String) -> some View {
        VStack(spacing: 2) {
            Image(systemName: systemImage)
                .font(.headline)
            Text(title)
                .font(.caption2)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, minHeight: 42)
    }

    private func shufflePromptAndGenerate() {
        let currentPreset = selectedPreset
        if Self.presets.count > 1 {
            var nextPreset = currentPreset
            while nextPreset == currentPreset {
                nextPreset = Self.presets.randomElement() ?? currentPreset
            }
            selectedPreset = nextPreset
            promptText = nextPreset.key
        }
        randomizeAndGenerate()
    }

    private func randomizeAndGenerate() {
        seed = Self.randomSeed()
        generate()
    }

    private static func randomSeed() -> UInt64 {
        UInt64.random(in: 0..<UInt64(max(1, TinyWeights.trainedSeedCount)))
    }

    private func syncSelectedPresetWithPrompt() {
        let trimmedPrompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return }

        let nextPreset = Self.preset(matching: trimmedPrompt)
            ?? WatchPromptPreset(key: trimmedPrompt, title: trimmedPrompt)
        if nextPreset != selectedPreset {
            selectedPreset = nextPreset
        }
    }

    private static func preset(matching prompt: String) -> WatchPromptPreset? {
        let normalizedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return presets.first { preset in
            preset.key.lowercased() == normalizedPrompt || preset.title.lowercased() == normalizedPrompt
        }
    }

    private func generate() {
        guard !isGenerating else { return }
        isGenerating = true
        let trimmedPrompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt = trimmedPrompt.isEmpty ? selectedPreset.key : trimmedPrompt
        let seed = seed
        let generationSize = Self.generationSize
        status = "Generating \(prompt)"
        print("[TinyImageWatch] generate start prompt=\"\(prompt)\" seed=\(seed) size=\(generationSize)")

        Task.detached(priority: .userInitiated) {
            let start = ContinuousClock.now
            let image = TinyImageGenerator().generate(prompt: prompt, seed: seed, size: generationSize)
            let elapsed = start.duration(to: .now)
            let elapsedMs = Double(elapsed.components.seconds) * 1000.0
                + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
            print(
                "[TinyImageWatch] generate done prompt=\"\(prompt)\" seed=\(seed) " +
                "size=\(image.width)x\(image.height) elapsedMs=\(Int(elapsedMs.rounded()))"
            )

            await MainActor.run {
                generatedImage = image
                status = "\(image.width)x\(image.height) \(Int(elapsedMs.rounded()))ms"
                isGenerating = false
            }
        }
    }

    private func cgImage(from image: TinyGeneratedImage) -> CGImage {
        let provider = CGDataProvider(data: Data(image.rgba) as CFData)!
        return CGImage(
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: image.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    private static func placeholderImage() -> TinyGeneratedImage {
        let size = generationSize
        var rgba = [UInt8]()
        rgba.reserveCapacity(size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let checker = ((x / 8) + (y / 8)).isMultiple(of: 2)
                let value: UInt8 = checker ? 220 : 196
                rgba.append(value)
                rgba.append(value)
                rgba.append(value)
                rgba.append(255)
            }
        }
        return TinyGeneratedImage(width: size, height: size, rgba: rgba)
    }
}
