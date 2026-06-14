// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "StableDiffusionLocal",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "StableDiffusion",
            targets: ["StableDiffusion"]
        ),
    ],
    targets: [
        .target(
            name: "StableDiffusion",
            path: "Sources/StableDiffusion"
        ),
    ]
)
