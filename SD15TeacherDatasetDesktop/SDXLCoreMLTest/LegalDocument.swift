import SwiftUI

enum AppSupport {
    static let email = "lube8163@gmail.com"
    static let emailURL = URL(string: "mailto:\(email)")!
}

enum LegalDocument: String, CaseIterable, Identifiable {
    case terms
    case privacy

    var id: String { rawValue }

    var title: String {
        switch self {
        case .terms:
            return "利用規約"
        case .privacy:
            return "プライバシーポリシー"
        }
    }

    var lastUpdatedText: String {
        "最終更新日: 2026年5月19日"
    }

    var sections: [LegalSection] {
        switch self {
        case .terms:
            return [
                .init(
                    title: "1. 適用",
                    body: "本規約は、ローカル画像生成（以下「本アプリ」）の利用条件を定めるものです。本アプリを利用することで、本規約に同意したものとみなします。"
                ),
                .init(
                    title: "2. 本アプリの内容",
                    body: "本アプリは、端末内に同梱された画像生成モデルを利用して、ユーザーが入力したプロンプトから画像を生成するツールです。生成結果はAIモデルの性質上、常に正確、適切、またはユーザーの意図どおりになるとは限りません。"
                ),
                .init(
                    title: "3. 禁止事項",
                    body: "ユーザーは、法令または公序良俗に反する目的、第三者の権利を侵害する目的、差別・嫌がらせ・脅迫・性的搾取・暴力を助長する目的、または本アプリや第三者に損害を与える目的で本アプリを利用してはなりません。"
                ),
                .init(
                    title: "4. 生成画像の取り扱い",
                    body: "生成画像の保存、共有、公開、商用利用その他の利用はユーザー自身の責任で行ってください。ユーザーは、生成画像の利用にあたり、適用される法令、権利処理、配布先プラットフォームの規約、および同梱モデルに関するライセンス条件を確認する責任を負います。"
                ),
                .init(
                    title: "5. 安全機能",
                    body: "本アプリには安全チェッカーに関する設定が含まれる場合がありますが、安全機能はすべての不適切な入力または出力を検出・防止するものではありません。ユーザーは生成結果を確認し、不適切な内容を保存、共有、公開しないよう注意してください。"
                ),
                .init(
                    title: "6. 免責",
                    body: "本アプリは現状有姿で提供されます。本アプリの利用、利用不能、生成結果、保存または共有されたコンテンツによりユーザーまたは第三者に生じた損害について、法令上認められる範囲で責任を負いません。"
                ),
                .init(
                    title: "7. 規約の変更",
                    body: "本規約は、機能追加、法令変更、ストア審査上の要請、その他必要に応じて変更されることがあります。変更後の規約は、本アプリ内または配布ページに掲載された時点で効力を生じます。"
                ),
                .init(
                    title: "8. お問い合わせ",
                    body: "本規約に関するお問い合わせは、\(AppSupport.email) までご連絡ください。"
                ),
            ]
        case .privacy:
            return [
                .init(
                    title: "1. 基本方針",
                    body: "本アプリは、ユーザーのプライバシーを尊重し、必要最小限のデータのみを端末上で扱います。本ポリシーは、本アプリにおける情報の取り扱いを説明するものです。"
                ),
                .init(
                    title: "2. 収集する情報",
                    body: "本アプリは、アカウント登録、氏名、メールアドレス、位置情報、連絡先、広告識別子、分析用識別子を収集しません。ユーザーが入力したプロンプト、ネガティブプロンプト、シード値、生成画像は、画像生成のために端末内で処理されます。"
                ),
                .init(
                    title: "3. 端末内での処理",
                    body: "画像生成は、アプリに同梱されたモデルを利用して端末内で実行されます。本アプリは、プロンプトや生成画像を開発者のサーバーへ送信しません。現在の実装では、広告SDK、アクセス解析SDK、外部トラッキングSDKも組み込んでいません。"
                ),
                .init(
                    title: "4. 写真ライブラリと共有",
                    body: "ユーザーが「写真に保存」を選んだ場合、生成画像を写真ライブラリへ追加するためにiOSの権限を利用します。ユーザーが共有機能を使った場合、共有先アプリやサービスに生成画像が渡されることがあります。その後の取り扱いは共有先の規約とプライバシーポリシーに従います。"
                ),
                .init(
                    title: "5. 保存と削除",
                    body: "生成画像は、ユーザーが明示的に写真ライブラリへ保存または共有しない限り、本アプリの画面表示のために一時的に保持されます。端末内には、初回利用時などにモデル実行用のキャッシュが作成されることがあります。アプリを削除すると、通常これらのアプリ内データも端末から削除されます。"
                ),
                .init(
                    title: "6. 第三者提供",
                    body: "本アプリは、開発者が収集した個人データを第三者へ販売または提供しません。ただし、ユーザー自身が共有機能を利用した場合、選択した共有先に画像などの情報が送信されることがあります。"
                ),
                .init(
                    title: "7. 子どものプライバシー",
                    body: "本アプリは、子どもから個人情報を意図的に収集することを目的としていません。未成年者が利用する場合は、保護者の管理と同意のもとで利用してください。"
                ),
                .init(
                    title: "8. ポリシーの変更",
                    body: "本ポリシーは、機能追加、法令変更、ストア審査上の要請、その他必要に応じて変更されることがあります。重要な変更がある場合は、本アプリ内または配布ページで通知します。"
                ),
                .init(
                    title: "9. お問い合わせ",
                    body: "プライバシーに関するお問い合わせ、同意の撤回、またはデータ削除に関する相談は、\(AppSupport.email) までご連絡ください。"
                ),
            ]
        }
    }
}

struct LegalSection: Identifiable {
    let id = UUID()
    let title: String
    let body: String
}

struct LegalDocumentView: View {
    let document: LegalDocument

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(document.title)
                        .font(.title2.weight(.bold))
                        .foregroundStyle(AppStyle.ink)
                    Text(document.lastUpdatedText)
                        .font(.footnote)
                        .foregroundStyle(AppStyle.muted)
                    Text("このページはストア提出に向けた実装用ドラフトです。公開前に、実際の配布形態、サポート窓口、モデルライセンス、対象地域に合わせて法務確認してください。")
                        .font(.footnote)
                        .foregroundStyle(AppStyle.muted)
                        .padding(.top, 4)
                }
                .padding(16)
                .background(AppStyle.surface)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(AppStyle.line, lineWidth: 1)
                )

                ForEach(document.sections) { section in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(section.title)
                            .font(.headline)
                            .foregroundStyle(AppStyle.ink)
                        Text(section.body)
                            .font(.body)
                            .foregroundStyle(AppStyle.slate)
                            .lineSpacing(4)
                            .textSelection(.enabled)
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(AppStyle.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(AppStyle.line, lineWidth: 1)
                    )
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(AppStyle.background.ignoresSafeArea())
        .navigationTitle(document.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
