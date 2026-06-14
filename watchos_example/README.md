# TinyImageWatchApp

This is the ready-to-open watchOS sample app for Tiny Watch Image Generator.

## Run In Xcode

```sh
open TinyImageWatchApp.xcodeproj
```

Select:

- Scheme: `TinyImageWatchApp`
- Destination: an Apple Watch Simulator

Then press Run.

The simulator path does not require a development team. For a physical Apple Watch, set your own Team in `Signing & Capabilities` and change the bundle identifier to one you own.

## Command Line Build

From the repository root:

```sh
xcodebuild \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme TinyImageWatchApp \
  -destination 'generic/platform=watchOS Simulator' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

To compile for physical watchOS architecture without signing:

```sh
xcodebuild \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme TinyImageWatchApp \
  -destination 'generic/platform=watchOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## Required Files

The target builds these files directly:

- `TinyImageWatchApp/ContentView.swift`
- `TinyImageWatchApp/TinyImageWatchApp.swift`
- `../Sources/TinyWatchGenerator/TinyImageGenerator.swift`
- `../Sources/TinyWatchGenerator/TinyWeights.swift`
- `TinyImageWatchApp/TinyWeights.bin`

`TinyWeights.bin` is intentionally tracked and copied into the app bundle. No model download is needed for the watchOS demo.
