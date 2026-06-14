import SwiftUI
import UIKit

enum AppStyle {
    static let background = Color(red: 0.965, green: 0.953, blue: 0.925)
    static let surface = Color(red: 1.0, green: 0.992, blue: 0.972)
    static let surfaceInset = Color(red: 0.945, green: 0.935, blue: 0.905)
    static let ink = Color(red: 0.18, green: 0.21, blue: 0.23)
    static let muted = Color(red: 0.42, green: 0.46, blue: 0.48)
    static let line = Color.black.opacity(0.075)
    static let accent = Color(red: 0.72, green: 0.57, blue: 0.34)
    static let accentSoft = Color(red: 0.91, green: 0.82, blue: 0.64)
    static let slate = Color(red: 0.32, green: 0.37, blue: 0.40)
    static let blueGray = Color(red: 0.48, green: 0.55, blue: 0.61)
}

struct ContentView: View {
    @ObservedObject var viewModel: SDXLGeneratorViewModel
    @FocusState private var focusedField: Field?
    @State private var isShowingShareSheet = false
    @State private var isShowingSettings = false
    @State private var alertMessage: String?

    private enum Field: Hashable {
        case prompt
    }

    var body: some View {
        NavigationStack {
            ZStack {
                AppStyle.background
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 14) {
                        promptCard
                        generateButton
                        progressCard

                        if let image = viewModel.image {
                            resultCard(image: image)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
                    .padding(.bottom, 24)
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 8) {
                        Image("AppLogoMark")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 26, height: 26)
                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        Text("ローカル画像生成")
                            .font(.headline)
                            .foregroundStyle(AppStyle.ink)
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        isShowingSettings = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(AppStyle.ink)
                            .frame(width: 40, height: 40)
                            .background(AppStyle.surface)
                            .clipShape(Circle())
                            .shadow(color: Color.black.opacity(0.08), radius: 8, y: 3)
                    }
                }

                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("閉じる") {
                        hideKeyboard()
                    }
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .sheet(isPresented: $isShowingShareSheet) {
                if let image = viewModel.image {
                    ShareSheet(items: [image])
                }
            }
            .sheet(isPresented: $isShowingSettings) {
                SettingsView(
                    viewModel: viewModel,
                    alertMessage: $alertMessage
                )
            }
            .alert("お知らせ", isPresented: Binding(
                get: { alertMessage != nil },
                set: { if !$0 { alertMessage = nil } }
            )) {
                Button("OK", role: .cancel) {
                    alertMessage = nil
                }
            } message: {
                Text(alertMessage ?? "")
            }
        }
        .preferredColorScheme(.light)
    }

    private var promptCard: some View {
        card {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("プロンプト", systemImage: "text.alignleft")
                        .font(.headline)
                        .foregroundStyle(AppStyle.ink)
                    Spacer()
                    Text(viewModel.modelSummaryText)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(AppStyle.slate)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(AppStyle.surfaceInset)
                        .clipShape(Capsule())
                }

                if viewModel.isShowingExamplePrompt {
                    Label("入力例です。このまま生成できます。", systemImage: "lightbulb")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(AppStyle.slate)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(AppStyle.surfaceInset)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }

                Text(viewModel.promptGuidanceText)
                    .font(.footnote)
                    .foregroundStyle(AppStyle.muted)

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(AppStyle.surfaceInset)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .stroke(AppStyle.line, lineWidth: 1)
                        )

                    TextEditor(text: $viewModel.prompt)
                        .focused($focusedField, equals: .prompt)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 180)
                        .padding(12)
                        .font(.system(.body, design: .default))
                        .foregroundStyle(AppStyle.ink)

                    if viewModel.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Text(viewModel.promptHintText)
                            .font(.body)
                            .foregroundStyle(AppStyle.muted.opacity(0.72))
                            .padding(.horizontal, 18)
                            .padding(.vertical, 20)
                            .allowsHitTesting(false)
                    }
                }
            }
        }
    }

    private var generateButton: some View {
        Button {
            hideKeyboard()
            viewModel.generate()
        } label: {
            HStack(spacing: 10) {
                if viewModel.isGenerating || viewModel.isPreparingModel {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "sparkles")
                }

                Text(viewModel.isPreparingModel ? "モデル準備中..." : (viewModel.isGenerating ? "生成中..." : "画像を生成する"))
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(
                LinearGradient(
                    colors: [AppStyle.slate, Color(red: 0.21, green: 0.25, blue: 0.27)],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .foregroundStyle(.white)
            .overlay(alignment: .top) {
                Rectangle()
                    .fill(AppStyle.accentSoft.opacity(0.34))
                    .frame(height: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .shadow(color: Color.black.opacity(0.12), radius: 12, y: 6)
        }
        .disabled(viewModel.isGenerating || viewModel.isPreparingModel)
    }

    private var progressCard: some View {
        card {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Label(viewModel.progressTitle, systemImage: viewModel.isGenerating ? "wand.and.stars" : (viewModel.isPreparingModel ? "arrow.triangle.2.circlepath" : "checkmark.seal"))
                        .font(.headline)
                        .foregroundStyle(AppStyle.ink)
                    Spacer()
                    if let progressValue = viewModel.progressValue {
                        Text(viewModel.progressStepText)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(AppStyle.slate)
                        Circle()
                            .fill(progressValue < 1 ? AppStyle.accent : Color.green)
                            .frame(width: 10, height: 10)
                    }
                }

                if let progressValue = viewModel.progressValue {
                    ProgressView(value: progressValue)
                        .tint(AppStyle.accent)
                }

                Text(viewModel.progressDetail)
                    .font(.subheadline)
                    .foregroundStyle(AppStyle.slate)

                if let errorMessage = viewModel.errorMessage {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private func resultCard(image: UIImage) -> some View {
        card {
            VStack(alignment: .leading, spacing: 14) {
                Label("生成結果", systemImage: "photo")
                    .font(.headline)
                    .foregroundStyle(AppStyle.ink)

                HStack(spacing: 8) {
                    Text("今回のシード値")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(AppStyle.ink)
                    Spacer()
                    Text(viewModel.lastUsedSeedText)
                        .font(.system(.footnote, design: .monospaced))
                        .foregroundStyle(AppStyle.slate)
                        .textSelection(.enabled)
                }

                Button {
                    UIPasteboard.general.string = viewModel.lastUsedSeedText
                    alertMessage = "シード値をコピーしました。"
                } label: {
                    Label("シード値をコピー", systemImage: "doc.on.doc")
                }
                .buttonStyle(SecondaryActionButtonStyle())

                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(AppStyle.line, lineWidth: 1)
                    )
                    .contextMenu {
                        Button("共有") {
                            isShowingShareSheet = true
                        }
                        Button("写真に保存") {
                            saveImageToPhotos(image)
                        }
                    }

                HStack(spacing: 10) {
                    Button {
                        isShowingShareSheet = true
                    } label: {
                        Label("共有", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(SecondaryActionButtonStyle())

                    Button {
                        saveImageToPhotos(image)
                    } label: {
                        Label("写真に保存", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(SecondaryActionButtonStyle())
                }
            }
        }
    }

    @ViewBuilder
    private func card<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(AppStyle.surface)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(AppStyle.line, lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.05), radius: 12, y: 5)
    }

    private func saveImageToPhotos(_ image: UIImage) {
        PhotoLibrarySaver.save(image) { result in
            switch result {
            case .success:
                alertMessage = "写真アプリに保存しました。"
            case .failure(let error):
                alertMessage = "保存に失敗しました: \(error.localizedDescription)"
            }
        }
    }

    private func hideKeyboard() {
        focusedField = nil
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil,
            from: nil,
            for: nil
        )
    }
}

private struct SecondaryActionButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(isEnabled ? AppStyle.slate : AppStyle.muted.opacity(0.72))
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(configuration.isPressed ? AppStyle.surfaceInset.opacity(0.72) : AppStyle.surfaceInset)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(AppStyle.line, lineWidth: 1)
            )
    }
}

private struct StepperControl: View {
    let value: Int
    let range: ClosedRange<Int>
    let onDecrement: () -> Void
    let onIncrement: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            Button(action: onDecrement) {
                Image(systemName: "minus")
                    .frame(width: 48, height: 36)
            }
            .disabled(value <= range.lowerBound)

            Divider()
                .frame(height: 22)

            Button(action: onIncrement) {
                Image(systemName: "plus")
                    .frame(width: 48, height: 36)
            }
            .disabled(value >= range.upperBound)
        }
        .font(.headline.weight(.semibold))
        .foregroundStyle(AppStyle.ink)
        .background(AppStyle.surfaceInset)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(AppStyle.line, lineWidth: 1)
        )
    }
}

private struct SettingSection<Content: View>: View {
    let title: String
    let systemImage: String
    let content: Content

    init(title: String, systemImage: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.bold))
                .foregroundStyle(AppStyle.slate)

            VStack(alignment: .leading, spacing: 13) {
                content
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(AppStyle.surface)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(AppStyle.line, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.045), radius: 10, y: 4)
    }
}

private struct SettingsView: View {
    @ObservedObject var viewModel: SDXLGeneratorViewModel
    @Binding var alertMessage: String?
    @Environment(\.dismiss) private var dismiss
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case manualSeed
        case negativePrompt
    }

    private var stepBinding: Binding<Int> {
        Binding(
            get: { viewModel.stepCount },
            set: { viewModel.stepCount = min(max($0, 1), 80) }
        )
    }

    private var guidanceBinding: Binding<Double> {
        Binding(
            get: { Double(viewModel.guidanceScale) },
            set: { viewModel.guidanceScale = Float(min(max($0, 1), 12)) }
        )
    }

    private var reduceMemoryBinding: Binding<Bool> {
        Binding(
            get: { viewModel.reduceMemoryEnabled },
            set: { viewModel.setReduceMemoryEnabled($0) }
        )
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    SettingSection(title: "サポート", systemImage: "envelope") {
                        Link(destination: AppSupport.emailURL) {
                            settingRowLabel("問い合わせ", value: AppSupport.email, systemImage: "paperplane")
                        }
                    }

                    SettingSection(title: "法務", systemImage: "doc.text") {
                        ForEach(LegalDocument.allCases) { document in
                            NavigationLink {
                                LegalDocumentView(document: document)
                            } label: {
                                settingRowLabel(
                                    document.title,
                                    value: document.lastUpdatedText.replacingOccurrences(of: "最終更新日: ", with: ""),
                                    systemImage: document == .terms ? "doc.text" : "hand.raised"
                                )
                            }
                        }
                    }

                    SettingSection(title: "モデル", systemImage: "cpu") {
                        labeledValue("使用モデル", value: viewModel.modelSummaryText)
                        helperText("モデルはアプリに同梱された 512px 版を標準で使います。256px 版は検証用です。")

                        labeledValue("検出済み解像度", value: viewModel.resolutionSummaryText)

                        Divider().opacity(0.55)

                        Toggle("メモリ使用量を抑える", isOn: reduceMemoryBinding)
                            .tint(AppStyle.accent)
                            .disabled(viewModel.isGenerating || viewModel.isPreparingModel)

                        helperText("通常はオン推奨です。オフにすると再読み込みは減る場合がありますが、メモリの少ない端末では動作が不安定になる可能性があります。")
                    }

                    SettingSection(title: "生成設定", systemImage: "slider.horizontal.3") {
                        VStack(alignment: .leading, spacing: 8) {
                            labeledValue("ステップ数", value: "\(viewModel.stepCount)")
                            helperText("ノイズ除去の反復回数です。増やすほど時間は伸びます。")
                            HStack {
                                Spacer()
                                StepperControl(
                                    value: viewModel.stepCount,
                                    range: 1...80,
                                    onDecrement: { stepBinding.wrappedValue -= 1 },
                                    onIncrement: { stepBinding.wrappedValue += 1 }
                                )
                            }
                        }

                        Divider().opacity(0.55)

                        VStack(alignment: .leading, spacing: 8) {
                            labeledValue(
                                "指示の強さ",
                                value: viewModel.guidanceScale.formatted(.number.precision(.fractionLength(1)))
                            )
                            helperText("CFG Scale です。高いほどプロンプトに忠実ですが、不自然さが出ることがあります。")
                            Slider(value: guidanceBinding, in: 1...12, step: 0.5)
                                .tint(AppStyle.accent)
                        }

                        Divider().opacity(0.55)

                        Picker("生成アルゴリズム", selection: $viewModel.schedulerOption) {
                            ForEach(SchedulerOption.allCases) { option in
                                Text(option.displayName).tag(option)
                            }
                        }
                        .tint(AppStyle.slate)

                        helperText(viewModel.schedulerOption.detailText)

                        Toggle("安全チェッカーを無効にする", isOn: $viewModel.disableSafety)
                            .tint(AppStyle.accent)

                        helperText("通常はオフのままで問題ありません。")
                    }

                    SettingSection(title: "シード値", systemImage: "number") {
                        Picker("シードの使い方", selection: $viewModel.seedMode) {
                            ForEach(SeedMode.allCases) { mode in
                                Text(mode.displayName).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        .tint(AppStyle.slate)

                        helperText(viewModel.seedMode.detailText)

                        if viewModel.seedMode == .manual {
                            TextField("固定シード値 (0〜4294967295)", text: $viewModel.manualSeedText)
                                .keyboardType(.numberPad)
                                .textInputAutocapitalization(.never)
                                .focused($focusedField, equals: .manualSeed)
                                .padding(12)
                                .background(AppStyle.surfaceInset)
                                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        }

                        labeledValue("直近の生成シード", value: viewModel.lastUsedSeedText)

                        Button {
                            UIPasteboard.general.string = viewModel.lastUsedSeedText
                            alertMessage = "直近のシード値をコピーしました。"
                        } label: {
                            Label("直近のシード値をコピー", systemImage: "doc.on.doc")
                        }
                        .buttonStyle(SecondaryActionButtonStyle())
                        .disabled(viewModel.lastUsedSeed == nil)
                    }

                    SettingSection(title: "ネガティブプロンプト", systemImage: "minus.circle") {
                        Picker("プリセット", selection: Binding(
                            get: { viewModel.selectedNegativePromptPreset },
                            set: { viewModel.applyNegativePromptPreset($0) }
                        )) {
                            ForEach(NegativePromptPreset.allCases) { preset in
                                Text(preset.displayName).tag(preset)
                            }
                        }
                        .tint(AppStyle.slate)

                        helperText(viewModel.selectedNegativePromptPreset.detailText)

                        TextEditor(text: $viewModel.negativePrompt)
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 120)
                            .padding(10)
                            .foregroundStyle(AppStyle.ink)
                            .background(AppStyle.surfaceInset)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .stroke(AppStyle.line, lineWidth: 1)
                            )
                            .focused($focusedField, equals: .negativePrompt)
                    }
                }
                .padding(16)
            }
            .background(AppStyle.background.ignoresSafeArea())
            .navigationTitle("詳細設定")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("閉じる") {
                        dismiss()
                    }
                }

                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("閉じる") {
                        hideKeyboard()
                    }
                }
            }
        }
        .presentationDetents([.large])
    }

    private func settingRowLabel(_ title: String, value: String, systemImage: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(AppStyle.accent)
                .frame(width: 26, height: 26)
                .background(AppStyle.surfaceInset)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

            Text(title)
                .font(.body.weight(.semibold))
                .foregroundStyle(AppStyle.ink)

            Spacer()

            Text(value)
                .font(.caption)
                .foregroundStyle(AppStyle.muted)
                .lineLimit(1)
        }
        .contentShape(Rectangle())
    }

    private func labeledValue(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(AppStyle.muted)
            Text(value)
                .font(.body.weight(.semibold))
                .foregroundStyle(AppStyle.ink)
        }
    }

    private func helperText(_ text: String) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(AppStyle.muted)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func hideKeyboard() {
        focusedField = nil
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil,
            from: nil,
            for: nil
        )
    }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

private final class PhotoLibrarySaver: NSObject {
    private var completion: ((Result<Void, Error>) -> Void)?

    static func save(_ image: UIImage, completion: @escaping (Result<Void, Error>) -> Void) {
        let saver = PhotoLibrarySaver()
        saver.completion = completion
        UIImageWriteToSavedPhotosAlbum(
            image,
            saver,
            #selector(PhotoLibrarySaver.image(_:didFinishSavingWithError:contextInfo:)),
            nil
        )
        PhotoLibrarySaverRetainer.shared.retain(saver)
    }

    @objc
    private func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeMutableRawPointer?) {
        defer { PhotoLibrarySaverRetainer.shared.release(self) }
        if let error {
            completion?(.failure(error))
        } else {
            completion?(.success(()))
        }
    }
}

private final class PhotoLibrarySaverRetainer {
    static let shared = PhotoLibrarySaverRetainer()
    private var activeSavers: [ObjectIdentifier: PhotoLibrarySaver] = [:]

    func retain(_ saver: PhotoLibrarySaver) {
        activeSavers[ObjectIdentifier(saver)] = saver
    }

    func release(_ saver: PhotoLibrarySaver) {
        activeSavers.removeValue(forKey: ObjectIdentifier(saver))
    }
}

#Preview {
    ContentView(viewModel: SDXLGeneratorViewModel())
}
