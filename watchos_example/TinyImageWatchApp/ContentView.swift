import SwiftUI

struct WatchPromptPreset: Identifiable, Hashable {
    let key: String
    let title: String

    var id: String { key }
}

private struct WatchPromptSlotOption: Identifiable, Hashable {
    let key: String
    let title: String

    var id: String { key }
}

private enum WatchPromptInputMode: String, CaseIterable, Identifiable {
    case slots
    case text

    var id: String { rawValue }

    var title: String {
        switch self {
        case .slots:
            return "Slots"
        case .text:
            return "Text"
        }
    }

    var systemImage: String {
        switch self {
        case .slots:
            return "rectangle.grid.2x2"
        case .text:
            return "text.cursor"
        }
    }
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
    private static let subjectSlots = presets.map {
        WatchPromptSlotOption(key: $0.key, title: $0.title)
    }
    private static let colorSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Any"),
        .init(key: "red", title: "Red"),
        .init(key: "orange", title: "Orange"),
        .init(key: "yellow", title: "Yellow"),
        .init(key: "green", title: "Green"),
        .init(key: "blue", title: "Blue"),
        .init(key: "purple", title: "Purple"),
        .init(key: "pink", title: "Pink"),
        .init(key: "brown", title: "Brown"),
        .init(key: "black", title: "Black"),
        .init(key: "white", title: "White"),
        .init(key: "gray", title: "Gray")
    ]
    private static let actionSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Still"),
        .init(key: "sitting", title: "Sitting"),
        .init(key: "standing", title: "Standing"),
        .init(key: "running", title: "Running"),
        .init(key: "walking", title: "Walking"),
        .init(key: "flying", title: "Flying"),
        .init(key: "swimming", title: "Swimming"),
        .init(key: "sleeping", title: "Sleeping"),
        .init(key: "eating", title: "Eating"),
        .init(key: "holding", title: "Holding"),
        .init(key: "jumping", title: "Jumping"),
        .init(key: "floating", title: "Floating"),
        .init(key: "tilted", title: "Tilted"),
        .init(key: "shining", title: "Shining"),
        .init(key: "smiling", title: "Smiling")
    ]
    private static let viewSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Any"),
        .init(key: "front view", title: "Front"),
        .init(key: "side view", title: "Side"),
        .init(key: "back view", title: "Back"),
        .init(key: "top view", title: "Top"),
        .init(key: "closeup", title: "Close")
    ]
    private static let styleSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Plain"),
        .init(key: "icon", title: "Icon"),
        .init(key: "cartoon", title: "Cartoon"),
        .init(key: "anime", title: "Anime"),
        .init(key: "photo", title: "Photo"),
        .init(key: "toy", title: "Toy"),
        .init(key: "watercolor", title: "Paint"),
        .init(key: "sketch", title: "Sketch")
    ]

    @State private var inputMode: WatchPromptInputMode = .slots
    @State private var selectedPreset = Self.presets[0]
    @State private var selectedSubject = Self.presets[0].key
    @State private var selectedColor = ""
    @State private var selectedAction = ""
    @State private var selectedView = ""
    @State private var selectedStyle = ""
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

    private var slotPrompt: String {
        [selectedStyle, selectedColor, selectedSubject, selectedAction, selectedView]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private var activePrompt: String {
        let source = inputMode == .slots ? slotPrompt : promptText
        let trimmedPrompt = source.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmedPrompt.isEmpty ? selectedPreset.key : trimmedPrompt
    }

    var body: some View {
        ScrollView(.vertical) {
            VStack(spacing: 8) {
                imageView

                inputControls

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

    private var inputControls: some View {
        VStack(spacing: 7) {
            HStack(spacing: 6) {
                ForEach(WatchPromptInputMode.allCases) { mode in
                    modeButton(mode)
                }
            }

            if inputMode == .slots {
                slotInputView
            } else {
                textInputView
            }
        }
    }

    private var slotInputView: some View {
        VStack(spacing: 6) {
            slotRow(
                systemImage: "square.grid.2x2",
                selection: $selectedSubject,
                options: Self.subjectSlots
            )
            slotRow(
                systemImage: "paintpalette",
                selection: $selectedColor,
                options: Self.colorSlots
            )
            slotRow(
                systemImage: "figure.run",
                selection: $selectedAction,
                options: Self.actionSlots
            )
            slotRow(
                systemImage: "camera.viewfinder",
                selection: $selectedView,
                options: Self.viewSlots
            )
            slotRow(
                systemImage: "sparkles",
                selection: $selectedStyle,
                options: Self.styleSlots
            )

            Text(slotPrompt)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.65)
                .frame(maxWidth: .infinity)
        }
    }

    private var textInputView: some View {
        VStack(spacing: 6) {
            Picker("Prompt", selection: $selectedPreset) {
                ForEach(promptPickerItems) { preset in
                    Text(preset.title).tag(preset)
                }
            }
            .labelsHidden()
            .frame(height: 36)
            .onChange(of: selectedPreset) {
                promptText = selectedPreset.key
                syncSlotsWithPreset()
            }

            TextField("Prompt", text: $promptText)
                .font(.caption)
                .multilineTextAlignment(.center)
                .textInputAutocapitalization(.never)
                .disableAutocorrection(true)
                .onChange(of: promptText) {
                    syncSelectedPresetWithPrompt()
                }
        }
    }

    private func modeButton(_ mode: WatchPromptInputMode) -> some View {
        let isSelected = inputMode == mode
        return Button {
            switchInputMode(to: mode)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: mode.systemImage)
                    .font(.caption)
                Text(mode.title)
                    .font(.caption2)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
            .frame(maxWidth: .infinity, minHeight: 28)
            .foregroundColor(isSelected ? Color.white : Color.primary)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.14))
            )
        }
        .buttonStyle(.plain)
        .disabled(isGenerating)
    }

    private func slotRow(
        systemImage: String,
        selection: Binding<String>,
        options: [WatchPromptSlotOption]
    ) -> some View {
        HStack(spacing: 5) {
            Image(systemName: systemImage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 17)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 5) {
                    ForEach(options) { option in
                        slotChip(option: option, selection: selection)
                    }
                }
            }
            .frame(height: 27)
        }
        .frame(maxWidth: .infinity)
    }

    private func slotChip(
        option: WatchPromptSlotOption,
        selection: Binding<String>
    ) -> some View {
        let isSelected = selection.wrappedValue == option.key
        return Button {
            selection.wrappedValue = option.key
            syncPromptFromSlots()
        } label: {
            Text(option.title)
                .font(.caption2)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .padding(.horizontal, 8)
                .frame(height: 26)
                .foregroundColor(isSelected ? Color.white : Color.primary)
                .background(
                    RoundedRectangle(cornerRadius: 7)
                        .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.15))
                )
        }
        .buttonStyle(.plain)
        .disabled(isGenerating)
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
        if inputMode == .slots {
            shuffleSlots()
        } else {
            let currentPreset = selectedPreset
            if Self.presets.count > 1 {
                var nextPreset = currentPreset
                while nextPreset == currentPreset {
                    nextPreset = Self.presets.randomElement() ?? currentPreset
                }
                selectedPreset = nextPreset
                selectedSubject = nextPreset.key
                promptText = nextPreset.key
            }
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

    private func switchInputMode(to mode: WatchPromptInputMode) {
        guard inputMode != mode else { return }
        inputMode = mode
        switch mode {
        case .slots:
            syncSlotsFromPrompt()
            syncPromptFromSlots()
        case .text:
            promptText = slotPrompt
            syncSelectedPresetWithPrompt()
        }
    }

    private func syncPromptFromSlots() {
        promptText = slotPrompt
        syncSelectedPresetWithPrompt()
    }

    private func syncSlotsWithPreset() {
        guard Self.subjectSlots.contains(where: { $0.key == selectedPreset.key }) else {
            return
        }
        selectedSubject = selectedPreset.key
        selectedColor = ""
        selectedAction = ""
        selectedView = ""
        selectedStyle = ""
    }

    private func syncSlotsFromPrompt() {
        let normalizedPrompt = Self.normalizedPrompt(promptText)
        if let exactPreset = Self.preset(matching: normalizedPrompt) {
            selectedPreset = exactPreset
            selectedSubject = exactPreset.key
            selectedColor = ""
            selectedAction = ""
            selectedView = ""
            selectedStyle = ""
            return
        }

        if let subject = Self.matchingOption(in: Self.subjectSlots, prompt: normalizedPrompt) {
            selectedSubject = subject.key
            if let preset = Self.preset(matching: subject.key) {
                selectedPreset = preset
            }
        }
        selectedColor = Self.matchingOption(in: Self.colorSlots, prompt: normalizedPrompt)?.key ?? ""
        selectedAction = Self.matchingOption(in: Self.actionSlots, prompt: normalizedPrompt)?.key ?? ""
        selectedView = Self.matchingOption(in: Self.viewSlots, prompt: normalizedPrompt)?.key ?? ""
        selectedStyle = Self.matchingOption(in: Self.styleSlots, prompt: normalizedPrompt)?.key ?? ""
    }

    private func shuffleSlots() {
        let currentSubject = selectedSubject
        if Self.subjectSlots.count > 1 {
            var nextSubject = currentSubject
            while nextSubject == currentSubject {
                nextSubject = Self.subjectSlots.randomElement()?.key ?? currentSubject
            }
            selectedSubject = nextSubject
        }
        selectedColor = Self.randomOptionalSlot(from: Self.colorSlots, emptyWeight: 4)
        selectedAction = Self.randomOptionalSlot(from: Self.actionSlots, emptyWeight: 3)
        selectedView = Self.randomOptionalSlot(from: Self.viewSlots, emptyWeight: 3)
        selectedStyle = Self.randomOptionalSlot(from: Self.styleSlots, emptyWeight: 4)
        syncPromptFromSlots()
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
        let normalizedPrompt = normalizedPrompt(prompt)
        return presets.first { preset in
            preset.key.lowercased() == normalizedPrompt || preset.title.lowercased() == normalizedPrompt
        }
    }

    private static func normalizedPrompt(_ prompt: String) -> String {
        prompt
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
    }

    private static func matchingOption(
        in options: [WatchPromptSlotOption],
        prompt: String
    ) -> WatchPromptSlotOption? {
        let normalizedPrompt = Self.normalizedPrompt(prompt)
        return options
            .filter { !$0.key.isEmpty }
            .sorted { $0.key.count > $1.key.count }
            .first { option in
                let key = Self.normalizedPrompt(option.key)
                return normalizedPrompt == key || normalizedPrompt.contains(key)
            }
    }

    private static func randomOptionalSlot(
        from options: [WatchPromptSlotOption],
        emptyWeight: Int
    ) -> String {
        guard !options.isEmpty else { return "" }
        let weightedEmpty = Array(repeating: "", count: max(0, emptyWeight))
        let keys = options.dropFirst().map(\.key)
        return (weightedEmpty + keys).randomElement() ?? ""
    }

    private func generate() {
        guard !isGenerating else { return }
        isGenerating = true
        let prompt = activePrompt
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
