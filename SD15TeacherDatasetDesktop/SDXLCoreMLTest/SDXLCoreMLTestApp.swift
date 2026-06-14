import SwiftUI

@main
struct SDXLCoreMLTestApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var viewModel = SDXLGeneratorViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .task {
                    viewModel.preloadResourcesIfNeeded()
                }
                .onChange(of: scenePhase) { _, newPhase in
                    switch newPhase {
                    case .active:
                        viewModel.preloadResourcesIfNeeded()
                    case .background:
                        viewModel.releaseCachedResources()
                    case .inactive:
                        break
                    @unknown default:
                        break
                    }
                }
        }
    }
}
