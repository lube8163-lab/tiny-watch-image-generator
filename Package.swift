// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TinyWatchImageGenerator",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .watchOS(.v10)
    ],
    products: [
        .library(
            name: "TinyWatchGenerator",
            targets: ["TinyWatchGenerator"]
        )
    ],
    targets: [
        .target(name: "TinyWatchGenerator"),
        .executableTarget(
            name: "TinyPreview",
            dependencies: ["TinyWatchGenerator"]
        )
    ]
)
