import SwiftUI

struct WatchPromptPreset: Identifiable, Hashable {
    let key: String
    let title: String

    var id: String { key }
}

private struct WatchPromptSlotOption: Identifiable, Hashable {
    let key: String
    let title: String
    let aliases: [String]

    var id: String { key }

    init(key: String, title: String, aliases: [String] = []) {
        self.key = key
        self.title = title
        self.aliases = aliases
    }
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
        .init(key: "face", title: "Face"),
        .init(key: "astronaut", title: "Astronaut"),
        .init(key: "alien", title: "Alien"),
        .init(key: "dragon", title: "Dragon"),
        .init(key: "penguin", title: "Penguin"),
        .init(key: "turtle", title: "Turtle"),
        .init(key: "elephant", title: "Elephant"),
        .init(key: "lion", title: "Lion"),
        .init(key: "monkey", title: "Monkey"),
        .init(key: "frog", title: "Frog"),
        .init(key: "duck", title: "Duck"),
        .init(key: "deer", title: "Deer"),
        .init(key: "whale", title: "Whale"),
        .init(key: "umbrella", title: "Umbrella"),
        .init(key: "key", title: "Key"),
        .init(key: "bottle", title: "Bottle"),
        .init(key: "pencil", title: "Pencil"),
        .init(key: "lamp", title: "Lamp"),
        .init(key: "phone", title: "Phone"),
        .init(key: "computer", title: "Computer"),
        .init(key: "crown", title: "Crown"),
        .init(key: "diamond", title: "Diamond"),
        .init(key: "sword", title: "Sword"),
        .init(key: "shield", title: "Shield"),
        .init(key: "cactus", title: "Cactus"),
        .init(key: "volcano", title: "Volcano"),
        .init(key: "fire", title: "Fire"),
        .init(key: "icecream", title: "Ice Cream"),
        .init(key: "donut", title: "Donut"),
        .init(key: "sushi", title: "Sushi")
    ]
    private static let subjectSlots = presets.map {
        WatchPromptSlotOption(key: $0.key, title: $0.title, aliases: subjectAliases[$0.key] ?? [])
    }
    private static let colorSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Any"),
        .init(key: "red", title: "Red", aliases: ["scarlet", "赤", "赤い"]),
        .init(key: "orange", title: "Orange", aliases: ["橙", "オレンジ色"]),
        .init(key: "yellow", title: "Yellow", aliases: ["gold", "golden", "黄色", "金色"]),
        .init(key: "green", title: "Green", aliases: ["緑", "緑色"]),
        .init(key: "blue", title: "Blue", aliases: ["cyan", "青", "青い", "水色"]),
        .init(key: "purple", title: "Purple", aliases: ["violet", "紫"]),
        .init(key: "pink", title: "Pink", aliases: ["ピンク"]),
        .init(key: "brown", title: "Brown", aliases: ["茶色"]),
        .init(key: "black", title: "Black", aliases: ["黒", "黒い"]),
        .init(key: "white", title: "White", aliases: ["白", "白い"]),
        .init(key: "gray", title: "Gray", aliases: ["grey", "silver", "銀色", "灰色"])
    ]
    private static let actionSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Still"),
        .init(key: "sitting", title: "Sitting", aliases: ["sit", "seated", "座る", "座っている"]),
        .init(key: "standing", title: "Standing", aliases: ["stand", "立つ", "立っている"]),
        .init(key: "running", title: "Running", aliases: ["run", "走る", "走っている"]),
        .init(key: "walking", title: "Walking", aliases: ["walk", "歩く", "歩いている"]),
        .init(key: "flying", title: "Flying", aliases: ["fly", "飛ぶ", "飛んでいる"]),
        .init(key: "swimming", title: "Swimming", aliases: ["swim", "泳ぐ", "泳いでいる"]),
        .init(key: "sleeping", title: "Sleeping", aliases: ["sleep", "眠る", "寝ている"]),
        .init(key: "eating", title: "Eating", aliases: ["eat", "食べる", "食べている"]),
        .init(key: "holding", title: "Holding", aliases: ["hold", "持つ", "持っている"]),
        .init(key: "jumping", title: "Jumping", aliases: ["jump", "跳ぶ", "ジャンプ"]),
        .init(key: "floating", title: "Floating", aliases: ["float", "drifting", "drift", "浮く", "浮いている"]),
        .init(key: "tilted", title: "Tilted", aliases: ["leaning", "lean", "turning", "turn", "傾く", "斜め"]),
        .init(key: "shining", title: "Shining", aliases: ["shine", "glowing", "glow", "sparkling", "sparkle", "輝く", "光る"]),
        .init(key: "smiling", title: "Smiling", aliases: ["smile", "happy", "笑顔", "笑う"]),
        .init(key: "parked", title: "Parked", aliases: ["parking", "駐車"]),
        .init(key: "rolling", title: "Rolling", aliases: ["roll", "転がる", "転がっている"]),
        .init(key: "bouncing", title: "Bounce", aliases: ["bounce", "跳ねる", "弾む"]),
        .init(key: "sliding", title: "Sliding", aliases: ["slide", "滑る", "滑っている"])
    ]
    private static let viewSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Any"),
        .init(key: "front view", title: "Front", aliases: ["front", "正面"]),
        .init(key: "side view", title: "Side", aliases: ["side", "profile", "横向き", "横"]),
        .init(key: "back view", title: "Back", aliases: ["back", "rear", "後ろ"]),
        .init(key: "top view", title: "Top", aliases: ["top", "overhead", "上から"]),
        .init(key: "closeup", title: "Close", aliases: ["close up", "close-up", "macro", "アップ"])
    ]
    private static let styleSlots: [WatchPromptSlotOption] = [
        .init(key: "", title: "Plain"),
        .init(key: "icon", title: "Icon", aliases: ["symbol", "emoji", "sticker", "アイコン"]),
        .init(key: "cartoon", title: "Cartoon", aliases: ["toon", "comic"]),
        .init(key: "anime", title: "Anime", aliases: ["manga", "アニメ"]),
        .init(key: "photo", title: "Photo", aliases: ["photograph", "realistic", "写真"]),
        .init(key: "toy", title: "Toy", aliases: ["plush", "figurine", "おもちゃ"]),
        .init(key: "watercolor", title: "Paint", aliases: ["painting", "painted", "水彩"]),
        .init(key: "sketch", title: "Sketch", aliases: ["drawing", "lineart", "線画"])
    ]
    private static let subjectAliases: [String: [String]] = [
        "astronaut": ["spaceperson", "spaceman", "宇宙飛行士"],
        "alien": ["aliens", "extraterrestrial", "宇宙人"],
        "dragon": ["dragons", "竜", "ドラゴン"],
        "penguin": ["penguins", "ペンギン"],
        "turtle": ["turtles", "亀", "カメ"],
        "elephant": ["elephants", "象", "ゾウ"],
        "lion": ["lions", "ライオン"],
        "monkey": ["monkeys", "猿", "サル"],
        "frog": ["frogs", "蛙", "カエル"],
        "duck": ["ducks", "アヒル", "鴨"],
        "deer": ["鹿", "シカ"],
        "whale": ["whales", "くじら", "クジラ"],
        "cat": ["cats", "kitten", "kitty", "ねこ", "ネコ", "猫"],
        "dog": ["dogs", "puppy", "いぬ", "イヌ", "犬"],
        "rabbit": ["rabbits", "bunny", "うさぎ", "兎"],
        "horse": ["horses", "pony", "馬"],
        "bear": ["bears", "熊"],
        "fox": ["foxes", "きつね", "狐"],
        "owl": ["owls", "ふくろう"],
        "butterfly": ["butterflies", "蝶"],
        "apple": ["apples", "りんご", "リンゴ"],
        "strawberry": ["strawberries", "いちご"],
        "car": ["cars", "auto", "automobile", "vehicle", "車"],
        "bicycle": ["bicycles", "bike", "cycle", "自転車"],
        "airplane": ["airplanes", "plane", "aircraft", "飛行機"],
        "boat": ["boats", "ship", "船"],
        "tree": ["trees", "forest", "木", "森"],
        "mountain": ["mountains", "山"],
        "cloud": ["clouds", "雲"],
        "flower": ["flowers", "floral", "blossom", "rose", "tulip", "sunflower", "daisy", "orchid", "花"],
        "house": ["houses", "home", "building", "家"],
        "bird": ["birds", "cardinal", "peacock", "parrot", "eagle", "sparrow", "鳥"],
        "fish": ["fishes", "魚"],
        "train": ["trains", "railway", "電車", "列車"],
        "castle": ["castles", "城"],
        "book": ["books", "本"],
        "chair": ["chairs", "椅子"],
        "clock": ["clocks", "watch", "時計"],
        "cup": ["cups", "mug", "コップ"],
        "mushroom": ["mushrooms", "きのこ"],
        "heart": ["hearts", "ハート"],
        "guitar": ["guitars", "ギター"],
        "camera": ["cameras", "カメラ"],
        "shoe": ["shoes", "sneaker", "靴"],
        "umbrella": ["umbrellas", "傘", "かさ"],
        "key": ["keys", "鍵", "かぎ"],
        "bottle": ["bottles", "ボトル", "瓶"],
        "pencil": ["pencils", "鉛筆", "えんぴつ"],
        "lamp": ["lamps", "light", "ライト", "ランプ"],
        "phone": ["phones", "smartphone", "スマホ", "携帯"],
        "computer": ["computers", "laptop", "pc", "パソコン"],
        "crown": ["crowns", "王冠"],
        "diamond": ["diamonds", "gem", "jewel", "宝石", "ダイヤ"],
        "sword": ["swords", "剣"],
        "shield": ["shields", "盾"],
        "cactus": ["cacti", "サボテン"],
        "volcano": ["volcanoes", "火山"],
        "fire": ["flame", "flames", "炎", "火"],
        "icecream": ["ice cream", "ice-cream", "アイス", "アイスクリーム"],
        "donut": ["donuts", "doughnut", "doughnuts", "ドーナツ"],
        "sushi": ["寿司", "すし"],
        "face": ["faces", "portrait", "person", "girl", "boy", "顔", "人物", "女の子", "男の子"]
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
        let promptTokens = Set(Self.promptTokens(normalizedPrompt))
        return options
            .filter { !$0.key.isEmpty }
            .sorted { $0.key.count > $1.key.count }
            .first { option in
                Self.aliasMatches(tokens: promptTokens, phrase: normalizedPrompt, alias: option.key) ||
                    option.aliases.contains { alias in
                        Self.aliasMatches(tokens: promptTokens, phrase: normalizedPrompt, alias: alias)
                    }
            }
    }

    private static func promptTokens(_ prompt: String) -> [String] {
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

    private static func isPromptTokenScalar(_ scalar: UnicodeScalar) -> Bool {
        let value = scalar.value
        return (value >= 48 && value <= 57) ||
            (value >= 97 && value <= 122) ||
            (value >= 0x3040 && value <= 0x30ff) ||
            (value >= 0x3400 && value <= 0x9fff)
    }

    private static func aliasMatches(tokens: Set<String>, phrase: String, alias: String) -> Bool {
        let aliasTokens = promptTokens(alias)
        guard !aliasTokens.isEmpty else { return false }
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
