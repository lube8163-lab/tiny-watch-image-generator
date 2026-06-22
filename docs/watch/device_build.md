# Apple Watch Device Build

This project currently provides a Swift Package. To run it on a physical Apple Watch, create a watchOS app target in Xcode and attach this package to it.

## Requirements

- Mac with Xcode installed.
- iPhone paired with the Apple Watch.
- iPhone connected to the Mac and trusted by both sides.
- Developer Mode enabled on both iPhone and Apple Watch.
- Apple Account added in Xcode.
- Automatic signing enabled for the watchOS app target.

## Create a Watch App

1. Open Xcode.
2. Choose `File > New > Project`.
3. Select `watchOS > App`.
4. Choose `Watch-only App` for the simplest local demo.
5. Set an organization identifier and bundle identifier.
6. In `Signing & Capabilities`, select your development team and keep automatic signing enabled.

## Add This Package

1. In the Xcode project, choose `File > Add Package Dependencies`.
2. Click `Add Local...`.
3. Select this repo folder:

   `/Users/tasuku/Documents/ちっちゃい画像生成モデル`

4. Add the `TinyWatchGenerator` library to the watchOS app target.
5. Replace the app's `ContentView.swift` with the example in:

   `watch_example/ContentView.swift`

## Pair the Watch in Xcode

1. Connect the paired iPhone to the Mac.
2. Open `Window > Devices and Simulators`.
3. Select the iPhone in the sidebar.
4. Confirm that the paired Apple Watch appears under `Paired Watches`.
5. If prompted, trust the Mac from the iPhone and Apple Watch.
6. Make sure Developer Mode is enabled on the Apple Watch, not just the iPhone.

## Run

1. In Xcode's run destination menu, select the Apple Watch paired with the iPhone.
2. Build and run with `Cmd-R`.
3. The first launch may take a while because Xcode prepares the watch for development.

## Troubleshooting

- If the watch does not appear, reconnect the iPhone and reopen `Window > Devices and Simulators`.
- Make sure the Watch app on iPhone shows the target watch as the active paired watch.
- Keep iPhone, Watch, and Mac on the same Wi-Fi network when possible.
- Unlock both the iPhone and Apple Watch during preparation.
- If signing fails, open `Signing & Capabilities` for the watchOS target and confirm the selected team.
- For older watches or older iOS/watchOS combinations, keeping the iPhone physically connected to the Mac may be required.

## Notes for ML Experiments

Start with `TinyWatchGenerator` before adding Core ML packages. Once the watch app launches reliably, add one model component at a time:

1. tiny Swift generator,
2. tiny decoder Core ML package,
3. denoiser Core ML package,
4. text conditioning.

This makes build, signing, and runtime memory problems easier to isolate.
