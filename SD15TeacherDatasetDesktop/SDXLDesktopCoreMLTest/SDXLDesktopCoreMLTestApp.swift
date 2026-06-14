import SwiftUI

@main
struct SDXLDesktopCoreMLTestApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var viewModel = SDXLGeneratorViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .onChange(of: scenePhase) { _, newPhase in
                    switch newPhase {
                    case .active:
                        viewModel.cancelScheduledResourceRelease()
                    case .background:
                        viewModel.scheduleCachedResourceRelease()
                    case .inactive:
                        break
                    @unknown default:
                        break
                    }
                }
        }
    }
}
