import SwiftUI

struct ContentView: View {
    @State private var seed: UInt64 = 1
    @State private var prompt = "cat"
    @State private var generatedImage = TinyImageGenerator().generate(prompt: "cat", seed: 1, size: 128)
    @State private var isGenerating = false
    @State private var status = "Core ML assets ready"
    @State private var selectedPreset = "cat"
    @State private var guidanceScale = 8.0
    @State private var selectedDecoderMode = CoreMLDecoderMode.compressed4Bit
    @State private var coreMLGenerator: LCMCoreMLImageGenerator?

    private let generator = TinyImageGenerator()

    var body: some View {
        NavigationStack {
            VStack(spacing: 18) {
                Image(decorative: cgImage(from: generatedImage), scale: 1)
                    .interpolation(.none)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .aspectRatio(1, contentMode: .fit)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                VStack(spacing: 12) {
                    TextField("Prompt", text: $prompt)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textFieldStyle(.roundedBorder)
                        .submitLabel(.done)
                        .onSubmit(generate)

                    Stepper(value: $seed, in: 0...999_999) {
                        Text("Seed \(seed)")
                            .monospacedDigit()
                    }

                    if let coreMLGenerator {
                        Picker("Preset", selection: $selectedPreset) {
                            ForEach(coreMLGenerator.presets) { preset in
                                Text(preset.title).tag(preset.key)
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: selectedPreset) {
                            prompt = selectedPreset
                        }

                        Picker("VAE", selection: $selectedDecoderMode) {
                            ForEach(coreMLGenerator.decoderModes) { mode in
                                Text(mode.title).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                    }

                    Stepper(value: $guidanceScale, in: 4...12, step: 1) {
                        Text("Guidance \(guidanceScale, specifier: "%.0f")")
                            .monospacedDigit()
                    }

                    Text(status)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    ProgressView()
                        .opacity(isGenerating ? 1 : 0)
                        .frame(height: 8)

                    Picker("Size", selection: .constant(128)) {
                        Text("128x128 Core ML").tag(128)
                    }
                    .pickerStyle(.segmented)

                    HStack(spacing: 12) {
                        Button {
                            seed &+= 1
                            generate()
                        } label: {
                            Label("Shuffle", systemImage: "shuffle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        Button(action: generate) {
                            Label("Generate", systemImage: "sparkles")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isGenerating || coreMLGenerator == nil)
                    }
                }
            }
            .padding()
            .navigationTitle("Tiny Image")
            .task {
                await loadCoreMLGenerator()
            }
        }
    }

    private func generate() {
        guard let coreMLGenerator, !isGenerating else { return }
        isGenerating = true
        status = "Generating \(selectedPreset), g\(Int(guidanceScale)), \(selectedDecoderMode.title)"
        let prompt = prompt
        let seed = seed
        let guidanceScale = Float(guidanceScale)
        let decoderMode = selectedDecoderMode

        Task.detached(priority: .userInitiated) {
            let result = Result {
                try coreMLGenerator.generate(
                    prompt: prompt,
                    seed: seed,
                    guidanceScale: guidanceScale,
                    decoderMode: decoderMode
                )
            }
            await MainActor.run {
                switch result {
                case .success(let image):
                    generatedImage = image
                    selectedPreset = coreMLGenerator.presets[coreMLGenerator.bestPresetIndex(for: prompt)].key
                    status = "Core ML 128x128 g\(Int(guidanceScale)) \(decoderMode.title)"
                case .failure(let error):
                    generatedImage = generator.generate(prompt: prompt, seed: seed, size: 128)
                    status = "Core ML failed: \(error.localizedDescription)"
                }
                isGenerating = false
            }
        }
    }

    private func loadCoreMLGenerator() async {
        let result = await Task.detached(priority: .userInitiated) {
            Result { try LCMCoreMLImageGenerator() }
        }.value

        switch result {
        case .success(let generator):
            coreMLGenerator = generator
            selectedPreset = generator.presets[generator.bestPresetIndex(for: prompt)].key
            if !generator.decoderModes.contains(selectedDecoderMode) {
                selectedDecoderMode = generator.decoderModes[0]
            }
            status = "Core ML assets ready, \(generator.stepCount) steps"
        case .failure(let error):
            status = "Core ML unavailable: \(error.localizedDescription)"
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
}
