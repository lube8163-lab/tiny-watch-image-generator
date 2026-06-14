import SwiftUI
import TinyWatchGenerator

struct ContentView: View {
    @State private var seed: UInt64 = 1
    @State private var prompt = "sunset"
    private let generator = TinyImageGenerator()

    var body: some View {
        VStack(spacing: 8) {
            Image(decorative: cgImage(from: generator.generate(prompt: prompt, seed: seed)), scale: 1)
                .interpolation(.none)
                .resizable()
                .scaledToFit()
                .frame(width: 128, height: 128)

            TextField("Prompt", text: $prompt)

            Button {
                seed &+= 1
            } label: {
                Image(systemName: "shuffle")
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
}
