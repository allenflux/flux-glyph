import UIKit
import CoreText
import CryptoKit

private let readyFile = "frame-ready.json"
private var activeFontAssets: [String: [String: Any]] = [:]
private var registeredAssetHashes: [String: String] = [:]

private func registerAssets(_ fonts: [[String: Any]]) -> [[String: Any]] {
    activeFontAssets = [:]
    var reports: [[String: Any]] = []
    for entry in fonts where (entry["kind"] as? String) == "asset" {
        let name = entry["path"] as? String ?? ""
        let expected = entry["sha256"] as? String ?? ""
        let postscript = entry["postscript"] as? String ?? entry["font_postscript"] as? String ?? ""
        var report: [String: Any] = ["kind": "asset", "path": name, "sha256": expected,
            "postscript": postscript, "family": entry["family"] as? String ?? "", "registration_succeeded": false,
            "sha256_verified": false]
        guard !name.isEmpty, URL(fileURLWithPath: name).lastPathComponent == name,
              !name.contains("\\"), !postscript.isEmpty, expected.count == 64 else {
            report["error"] = "invalid_asset_manifest"; reports.append(report); continue
        }
        let url = documentsURL("Fonts").appendingPathComponent(name)
        do {
            let bytes = try Data(contentsOf: url)
            let actual = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            guard actual == expected else { report["error"] = "font_asset_sha256_mismatch"; reports.append(report); continue }
            report["sha256_verified"] = true
            let descriptors = CTFontManagerCreateFontDescriptorsFromURL(url as CFURL) as? [CTFontDescriptor] ?? []
            let names = descriptors.compactMap { CTFontDescriptorCopyAttribute($0, kCTFontNameAttribute) as? String }
            report["descriptor_postscript_names"] = names
            guard names.contains(postscript) else { report["error"] = "asset_postscript_not_in_source"; reports.append(report); continue }
            var error: Unmanaged<CFError>?
            let registered = registeredAssetHashes[url.path] == expected || CTFontManagerRegisterFontsForURL(url as CFURL, .process, &error)
            guard registered else {
                report["error"] = error.map { String(describing: $0.takeRetainedValue()) } ?? "registration_failed"
                reports.append(report); continue
            }
            registeredAssetHashes[url.path] = expected
            report["registration_succeeded"] = true
            report["file_url"] = url.absoluteString
            activeFontAssets[postscript] = report
        } catch { report["error"] = String(describing: error) }
        reports.append(report)
    }
    return reports
}

private func documentsURL(_ name: String) -> URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(name)
}

private func writeJSON(_ value: [String: Any], name: String) {
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: documentsURL(name), options: .atomic)
    } catch {
        NSLog("Capture JSON write failed: %@", String(describing: error))
    }
}

private func rgba(_ value: Any?, fallback: UIColor) -> UIColor {
    guard let text = value as? String else { return fallback }
    let stripped = text.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
    guard (stripped.count == 6 || stripped.count == 8), let code = UInt64(stripped, radix: 16) else { return fallback }
    if stripped.count == 8 {
        return UIColor(red: CGFloat((code >> 24) & 255) / 255, green: CGFloat((code >> 16) & 255) / 255,
                       blue: CGFloat((code >> 8) & 255) / 255, alpha: CGFloat(code & 255) / 255)
    }
    return UIColor(red: CGFloat((code >> 16) & 255) / 255, green: CGFloat((code >> 8) & 255) / 255,
                   blue: CGFloat(code & 255) / 255, alpha: 1)
}

private func colorHex(_ color: UIColor) -> String {
    var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
    color.getRed(&r, green: &g, blue: &b, alpha: &a)
    return String(format: "#%02X%02X%02X%02X", Int((r * 255).rounded()), Int((g * 255).rounded()), Int((b * 255).rounded()), Int((a * 255).rounded()))
}

private func number(_ value: Any?, default fallback: Double) -> Double { (value as? NSNumber)?.doubleValue ?? fallback }

private func systemWeight(_ value: Any?) -> UIFont.Weight {
    if let raw = value as? NSNumber {
        let n = raw.doubleValue
        if (-1 ... 1).contains(n) { return UIFont.Weight(rawValue: CGFloat(n)) }
        switch n {
        case ..<350: return .light
        case ..<450: return .regular
        case ..<550: return .medium
        case ..<650: return .semibold
        case ..<750: return .bold
        default: return .heavy
        }
    }
    switch (value as? String)?.lowercased() {
    case "light": return .light
    case "medium": return .medium
    case "semibold", "semi-bold": return .semibold
    case "bold": return .bold
    case "heavy", "black": return .heavy
    default: return .regular
    }
}

private func canonicalFamily(postscript: String, family: String) -> String? {
    let lower = postscript.lowercased()
    // Regional PingFang names are distinct labels, never aliases of SC.
    for variant in ["SC", "TC", "HK"] {
        if lower.hasPrefix("pingfang\(variant.lowercased())-") && family == "PingFang \(variant)" {
            return "PingFang \(variant)"
        }
    }
    if lower.contains("helvetica") { return "Helvetica" }
    if lower.contains("sfui") || lower.contains("sfpro") || family.contains("SF Pro") || family.contains("SF UI") { return "SF Pro" }
    if let registered = activeFontAssets[postscript], registered["registration_succeeded"] as? Bool == true { return registered["family"] as? String }
    return nil
}

private func characterScript(_ scalar: UnicodeScalar) -> String? {
    if (0x4e00 ... 0x9fff).contains(scalar.value) { return "han" }
    if (0x30 ... 0x39).contains(scalar.value) || (0x41 ... 0x5a).contains(scalar.value) || (0x61 ... 0x7a).contains(scalar.value) { return "latin" }
    return nil
}

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    private var controller: CaptureController?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let controller = CaptureController()
        self.controller = controller
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = controller
        window.makeKeyAndVisible()
        self.window = window
        application.isIdleTimerDisabled = true
        if let url = launchOptions?[.url] as? URL { controller.open(url) }
        writeFontInventory()
        return true
    }

    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        controller?.open(url)
        return url.scheme == "fluxcapture"
    }

    func applicationDidBecomeActive(_ application: UIApplication) { controller?.becameActive() }

    fileprivate func writeFontInventory() {
        let named = UIFont.familyNames.sorted().map { family in
            ["family": family, "postscript_names": UIFont.fontNames(forFamilyName: family).sorted()] as [String: Any]
        }
        var system: [[String: Any]] = []
        for (weight, label) in [(UIFont.Weight.regular, "regular"), (.medium, "medium"), (.semibold, "semibold"), (.bold, "bold")] {
            for size in [16.0, 24.0, 36.0] {
                let font = UIFont.systemFont(ofSize: size, weight: weight)
                system.append(["weight": label, "size": size, "postscript_name": font.fontName, "family": font.familyName])
            }
        }
        writeJSON(["schema": "flux-glyph-ios-font-inventory-v1", "platform": "ios", "os_version": UIDevice.current.systemVersion,
                   "families": named, "system_fonts": system, "registered_font_assets": Array(activeFontAssets.values)], name: "font-inventory.json")
    }
}

private final class FrameSettler: NSObject {
    private var link: CADisplayLink?
    private var remaining = 6
    private let started = CACurrentMediaTime()
    private let canFinish: () -> Bool
    private let completion: () -> Void
    init(canFinish: @escaping () -> Bool, completion: @escaping () -> Void) {
        self.canFinish = canFinish
        self.completion = completion
        super.init()
        let link = CADisplayLink(target: self, selector: #selector(tick))
        self.link = link
        link.add(to: .main, forMode: .common)
    }
    func cancel() { link?.invalidate(); link = nil }
    @objc private func tick() {
        guard canFinish() else { remaining = 6; return }
        remaining -= 1
        if remaining <= 0 && CACurrentMediaTime() - started >= 0.15 { cancel(); completion() }
    }
    deinit { cancel() }
}

private final class CaptureController: UIViewController {
    private let canvas = CaptureView()
    private var pageIndex = 0
    private var generation = 0
    private var settler: FrameSettler?
    private var pendingReport: [String: Any]?
    private var commandTimer: Timer?
    private var requestID: String?
    private var activeSince = CACurrentMediaTime()
    override var prefersStatusBarHidden: Bool { true }
    override var prefersHomeIndicatorAutoHidden: Bool { true }
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .portrait }

    override func loadView() {
        view = canvas
        canvas.onRendered = { [weak self] report, version in
            DispatchQueue.main.async { self?.waitForDisplay(report, version: version) }
        }
    }
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        loadPage(pageIndex)
        if commandTimer == nil {
            let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in self?.pollCommand() }
            commandTimer = timer
            RunLoop.main.add(timer, forMode: .common)
        }
    }
    func becameActive() { activeSince = CACurrentMediaTime() }
    private func pollCommand() {
        guard UIApplication.shared.applicationState == .active,
              let data = try? Data(contentsOf: documentsURL("command.json")),
              let command = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let identifier = command["request_id"] as? String, !identifier.isEmpty, identifier != requestID,
              let index = command["page_index"] as? Int else { return }
        requestID = identifier
        pageIndex = index
        loadPage(index)
    }
    func open(_ url: URL) {
        guard url.scheme == "fluxcapture" else { return }
        if url.host == "page", let number = url.pathComponents.last.flatMap(Int.init) { pageIndex = number }
        else if url.host == "reload" { pageIndex = 0 }
        else { return }
        requestID = nil
        if isViewLoaded { loadPage(pageIndex) }
    }
    private func loadPage(_ index: Int) {
        generation += 1
        settler?.cancel(); settler = nil
        pendingReport = nil
        try? FileManager.default.removeItem(at: documentsURL(readyFile))
        do {
            let override = documentsURL("Scenes.json")
            let bundled = Bundle.main.url(forResource: "Scenes", withExtension: "json")
            guard let source = FileManager.default.fileExists(atPath: override.path) ? override : bundled else {
                throw NSError(domain: "FluxCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "Scenes.json is missing"])
            }
            let bytes = try Data(contentsOf: source)
            let payload = try JSONSerialization.jsonObject(with: bytes)
            let object = payload as? [String: Any]
            let assetReports = registerAssets(object?["fonts"] as? [[String: Any]] ?? [])
            (UIApplication.shared.delegate as? AppDelegate)?.writeFontInventory()
            let pages = object?["pages"] as? [[String: Any]] ?? payload as? [[String: Any]] ?? []
            guard pages.indices.contains(index) else {
                throw NSError(domain: "FluxCapture", code: 2, userInfo: [NSLocalizedDescriptionKey: "Page index \(index) outside 0..<\(pages.count)"])
            }
            let dimensions = object?["canvas_points"] as? [NSNumber] ?? [402, 874]
            let width = dimensions.count == 2 ? dimensions[0].doubleValue : 402
            let height = dimensions.count == 2 ? dimensions[1].doubleValue : 874
            guard width > 0, height > 0, width.isFinite, height.isFinite else {
                throw NSError(domain: "FluxCapture", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid canvas size"])
            }
            canvas.configure(page: pages[index], index: index, canvas: CGSize(width: width, height: height),
                             generation: generation, sourceSHA: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), assetReports: assetReports)
        } catch {
            writeJSON(["schema": "flux-glyph-ios-frame-v1", "status": "error", "page_index": index,
                       "error": String(describing: error), "generation": generation,
                       "request_id": requestID as Any? ?? NSNull()], name: readyFile)
        }
    }
    private func waitForDisplay(_ report: [String: Any], version: Int) {
        guard version == generation else { return }
        pendingReport = report
        guard settler == nil else { return }
        settler = FrameSettler(canFinish: { [weak self] in
            guard let self = self, let window = self.view.window else { return false }
            return version == self.generation && UIApplication.shared.applicationState == .active &&
                window.isKeyWindow && !window.isHidden && window.alpha == 1 &&
                CACurrentMediaTime() - self.activeSince >= 0.8
        }) { [weak self] in
            guard let self = self, version == self.generation, var ready = self.pendingReport else { return }
            ready["status"] = "ready"
            ready["request_id"] = self.requestID as Any? ?? NSNull()
            ready["application_state"] = "active"
            ready["window_is_key"] = self.view.window?.isKeyWindow ?? false
            let bounds = self.view.window?.bounds ?? .zero
            ready["window_bounds_points"] = [Double(bounds.minX), Double(bounds.minY), Double(bounds.maxX), Double(bounds.maxY)]
            ready["seconds_since_application_active"] = CACurrentMediaTime() - self.activeSince
            ready["ready_unix_seconds"] = Date().timeIntervalSince1970
            ready["display_link_frames_waited"] = 6
            ready["minimum_render_settle_seconds"] = 0.15
            writeJSON(ready, name: readyFile)
            self.settler = nil
        }
    }
}

private final class CaptureView: UIView {
    var onRendered: (([String: Any], Int) -> Void)?
    private var page: [String: Any] = [:]
    private var pageIndex = 0
    private var canvasSize = CGSize(width: 402, height: 874)
    private var generation = 0
    private var sourceSHA = ""
    private var assetReports: [[String: Any]] = []

    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = true
        backgroundColor = .white
        contentMode = .redraw
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    func configure(page: [String: Any], index: Int, canvas: CGSize, generation: Int, sourceSHA: String, assetReports: [[String: Any]]) {
        self.page = page; pageIndex = index; canvasSize = canvas; self.generation = generation; self.sourceSHA = sourceSHA
        self.assetReports = assetReports
        setNeedsLayout(); setNeedsDisplay()
    }
    override func layoutSubviews() { super.layoutSubviews(); setNeedsDisplay() }

    override func draw(_ rect: CGRect) {
        guard !page.isEmpty, let context = UIGraphicsGetCurrentContext() else { return }
        let background = rgba(page["background"], fallback: .white)
        background.setFill(); context.fill(bounds)
        let canvasScale = min(bounds.width / canvasSize.width, bounds.height / canvasSize.height)
        let offset = CGPoint(x: (bounds.width - canvasSize.width * canvasScale) / 2, y: (bounds.height - canvasSize.height * canvasScale) / 2)
        let screenScale = window?.screen.scale ?? UIScreen.main.scale
        context.saveGState()
        context.translateBy(x: offset.x, y: offset.y)
        context.scaleBy(x: canvasScale, y: canvasScale)
        // Quiet receipt framing; all text comes only from the annotated regions.
        let border = UIBezierPath(roundedRect: CGRect(x: 10, y: 82, width: canvasSize.width - 20, height: canvasSize.height - 112), cornerRadius: 18)
        UIColor.gray.withAlphaComponent(0.12).setStroke(); border.lineWidth = 0.5; border.stroke()
        let inputs = page["regions"] as? [[String: Any]] ?? []
        var results: [[String: Any]] = []
        for input in inputs {
            results.append(drawRegion(input, context: context, canvasScale: canvasScale, offset: offset, screenScale: screenScale))
        }
        context.restoreGState()
        let valid = results.filter { ($0["font_match_verified"] as? Bool) == true }.count
        let report: [String: Any] = [
            "schema": "flux-glyph-ios-frame-v1", "page_index": pageIndex, "page_id": page["id"] as? String ?? String(pageIndex),
            "split": page["split"] as? String ?? "unspecified", "generation": generation,
            "platform": "ios", "capture_source": "iOS Simulator UIKit/CoreText rendered app; screenshot supplied externally by simctl",
            "renderer": "UIView.draw + CTLineDraw", "os_version": UIDevice.current.systemVersion,
            "screen_scale": Double(screenScale), "pixel_size": [Int(round(bounds.width * screenScale)), Int(round(bounds.height * screenScale))],
            "source_screen_scale": Double(screenScale), "background_hex": colorHex(background),
            "view_points": [Double(bounds.width), Double(bounds.height)], "canvas_points": [Double(canvasSize.width), Double(canvasSize.height)],
            "canvas_scale": Double(canvasScale), "canvas_offset_points": [Double(offset.x), Double(offset.y)],
            "safe_area_points": [Double(safeAreaInsets.top), Double(safeAreaInsets.left), Double(safeAreaInsets.bottom), Double(safeAreaInsets.right)],
            "scene_manifest_sha256": sourceSHA, "background": page["background"] as? String ?? "#FFFFFF",
            "font_assets": assetReports,
            "verified_region_count": valid, "rejected_region_count": results.count - valid, "regions": results]
        onRendered?(report, generation)
    }

    private func pixelBox(_ box: CGRect, canvasScale: CGFloat, offset: CGPoint, screenScale: CGFloat) -> [Int] {
        [Int(floor((offset.x + box.minX * canvasScale) * screenScale)), Int(floor((offset.y + box.minY * canvasScale) * screenScale)),
         Int(ceil((offset.x + box.maxX * canvasScale) * screenScale)), Int(ceil((offset.y + box.maxY * canvasScale) * screenScale))]
    }

    private func drawRegion(_ input: [String: Any], context: CGContext, canvasScale: CGFloat, offset: CGPoint, screenScale: CGFloat) -> [String: Any] {
        let text = input["text"] as? String ?? ""
        let requested = input["font_postscript"] as? String ?? "-system"
        let expectedFamily = input["font_family"] as? String ?? ""
        let script = input["script"] as? String ?? ""
        let language = input["language"] as? String
        var result: [String: Any] = ["id": input["id"] as? String ?? "", "text": text, "script": script,
            "requested_font_postscript": requested, "requested_font_family": expectedFamily,
            "font_family": expectedFamily, "font_match_verified": false, "status": "rejected", "glyphs": []]
        func reject(_ reason: String) -> [String: Any] { result["reason"] = reason; return result }
        guard !text.isEmpty, !text.contains("\n"), ["han", "latin"].contains(script) else { return reject("invalid_text_or_script") }
        if let language = language {
            guard (script == "han" && ["zh-Hans", "zh-Hant", "zh-Hant-TW", "zh-Hant-HK"].contains(language)) ||
                  (script == "latin" && language == "en") else { return reject("invalid_capture_language") }
            result["requested_language"] = language
        }
        result["han_orthography"] = input["han_orthography"] ?? NSNull()
        let scalars = Array(text.unicodeScalars)
        guard scalars.allSatisfy({ $0.value <= 0xffff && !$0.properties.isJoinControl }),
              scalars.contains(where: { characterScript($0) == script }),
              !scalars.contains(where: { characterScript($0) != nil && characterScript($0) != script }) else { return reject("mixed_or_unsupported_script") }
        guard let points = input["bbox_points"] as? [NSNumber], points.count == 4 else { return reject("missing_bbox_points") }
        let box = CGRect(x: points[0].doubleValue, y: points[1].doubleValue,
                         width: points[2].doubleValue - points[0].doubleValue, height: points[3].doubleValue - points[1].doubleValue)
        guard box.width > 0, box.height > 0, box.minX >= 0, box.minY >= 0,
              box.maxX <= canvasSize.width, box.maxY <= canvasSize.height else { return reject("region_outside_canvas") }
        let pixel = pixelBox(box, canvasScale: canvasScale, offset: offset, screenScale: screenScale)
        result["bbox"] = pixel; result["bbox_pixels"] = pixel
        result["bbox_points"] = points.map { $0.doubleValue }
        let size = number(input["font_size"], default: 20)
        guard size.isFinite, size >= 6, size <= 120 else { return reject("invalid_font_size") }
        let font: UIFont?
        if requested == "-system" || requested == "SYSTEM" { font = UIFont.systemFont(ofSize: size, weight: systemWeight(input["weight"])) }
        else { font = UIFont(name: requested, size: size) }
        guard let font = font else { return reject("font_postscript_unavailable") }
        let ctFont = font as CTFont
        let actualPostscript = CTFontCopyPostScriptName(ctFont) as String
        let actualFamily = CTFontCopyFamilyName(ctFont) as String
        let family = canonicalFamily(postscript: actualPostscript, family: actualFamily)
        var fontSource: [String: Any] = ["kind": "ios_system", "os_version": UIDevice.current.systemVersion, "postscript": actualPostscript]
        if var asset = activeFontAssets[actualPostscript] {
            let resolvedURL = CTFontCopyAttribute(ctFont, kCTFontURLAttribute) as? URL
            let sourceURL = asset["file_url"] as? String
            let matchesURL = resolvedURL?.absoluteString == sourceURL
            asset["native_font_url"] = resolvedURL?.absoluteString ?? "unavailable"
            asset["native_font_url_verified"] = matchesURL
            fontSource = asset
            result["font_source"] = fontSource
            guard matchesURL else { return reject("resolved_font_asset_url_differs") }
        }
        result["font_source"] = fontSource
        result["actual_ui_font_postscript"] = font.fontName; result["actual_ui_font_family"] = font.familyName
        result["actual_font_postscript"] = actualPostscript; result["actual_font_family"] = actualFamily
        result["font_postscript"] = actualPostscript
        result["font_size_points"] = size * Double(canvasScale)
        result["ct_font_size"] = Double(CTFontGetSize(ctFont))
        let actualSizePoints = Double(CTFontGetSize(ctFont) * canvasScale)
        result["requested_font_size_points"] = size
        result["actual_font_size_points"] = actualSizePoints
        result["font_size_screen_px"] = actualSizePoints * Double(screenScale)
        result["source_screen_scale"] = Double(screenScale)
        result["font_traits"] = CTFontCopyTraits(ctFont) as NSDictionary
        guard family == expectedFamily else { result["fallback_detected"] = true; return reject("resolved_font_family_differs") }
        let color = rgba(input["color"], fallback: .black)
        let textColorHex = colorHex(color.resolvedColor(with: traitCollection))
        let backgroundHex = colorHex(rgba(page["background"], fallback: .white).resolvedColor(with: traitCollection))
        result["text_color_hex"] = textColorHex
        result["actual_text_color_hex"] = textColorHex
        result["background_hex"] = backgroundHex
        var attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): ctFont,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color.cgColor,
            NSAttributedString.Key(kCTLigatureAttributeName as String): 0]
        if let language = language { attributes[NSAttributedString.Key(kCTLanguageAttributeName as String)] = language }
        let attributed = NSAttributedString(string: text, attributes: attributes)
        let line = CTLineCreateWithAttributedString(attributed)
        var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
        let advance = CTLineGetTypographicBounds(line, &ascent, &descent, &leading)
        let baseline = CGPoint(x: box.minX, y: box.midY + (ascent - descent) / 2)
        let ink = CTLineGetBoundsWithOptions(line, [.useGlyphPathBounds])
        let inkBox = CGRect(x: baseline.x + ink.minX, y: baseline.y - ink.maxY, width: ink.width, height: ink.height)
        result["ink_bbox_pixels"] = pixelBox(inkBox, canvasScale: canvasScale, offset: offset, screenScale: screenScale)
        result["typographic_width_points"] = advance * Double(canvasScale)
        result["baseline_points"] = [Double(baseline.x * canvasScale + offset.x), Double(baseline.y * canvasScale + offset.y)]
        let utf16 = Array(text.utf16)
        var coverageGlyphs = [CGGlyph](repeating: 0, count: utf16.count)
        let coverageOK = CTFontGetGlyphsForCharacters(ctFont, utf16, &coverageGlyphs, utf16.count)
        var runs: [[String: Any]] = [], glyphRows: [[String: Any]] = []
        var fallback = false, clipped = false, zeroGlyphs = 0
        for item in CTLineGetGlyphRuns(line) as! [CTRun] {
            let attributes = CTRunGetAttributes(item) as NSDictionary
            guard let runFont = attributes[kCTFontAttributeName] else { fallback = true; continue }
            let actualRunFont = runFont as! CTFont
            let runPS = CTFontCopyPostScriptName(actualRunFont) as String
            let runFamily = CTFontCopyFamilyName(actualRunFont) as String
            let runCanonical = canonicalFamily(postscript: runPS, family: runFamily)
            if runPS != actualPostscript || runCanonical != expectedFamily { fallback = true }
            let count = CTRunGetGlyphCount(item)
            var glyphs = [CGGlyph](repeating: 0, count: count)
            var positions = [CGPoint](repeating: .zero, count: count)
            var indices = [CFIndex](repeating: 0, count: count)
            CTRunGetGlyphs(item, CFRange(location: 0, length: 0), &glyphs)
            CTRunGetPositions(item, CFRange(location: 0, length: 0), &positions)
            CTRunGetStringIndices(item, CFRange(location: 0, length: 0), &indices)
            zeroGlyphs += glyphs.filter { $0 == 0 }.count
            runs.append(["postscript_name": runPS, "family": runFamily, "glyph_count": count,
                         "language": attributes[kCTLanguageAttributeName] ?? NSNull(),
                         "glyph_ids": glyphs.map { Int($0) }, "string_indices_utf16": indices.map { Int($0) }])
            for i in 0 ..< count {
                var glyph = glyphs[i]
                let glyphBounds = CTFontGetBoundingRectsForGlyphs(actualRunFont, .horizontal, &glyph, nil, 1)
                let glyphBox = CGRect(x: baseline.x + positions[i].x + glyphBounds.minX,
                                      y: baseline.y - positions[i].y - glyphBounds.maxY,
                                      width: glyphBounds.width, height: glyphBounds.height)
                let index = indices[i]
                guard index >= 0, index < utf16.count, let scalar = UnicodeScalar(utf16[index]) else { fallback = true; continue }
                let character = String(scalar)
                let visible = glyphBounds.width > 0 && glyphBounds.height > 0
                let glyphPixel = pixelBox(glyphBox, canvasScale: canvasScale, offset: offset, screenScale: screenScale)
                let inside = !visible || (glyphPixel[0] >= 0 && glyphPixel[1] >= 0 && glyphPixel[2] <= Int(round(bounds.width * screenScale)) && glyphPixel[3] <= Int(round(bounds.height * screenScale)))
                if !inside { clipped = true }
                glyphRows.append(["character": character, "text_index": index, "glyph_id": Int(glyphs[i]),
                    "bbox": glyphPixel, "bbox_screen_px": glyphPixel, "visible": visible,
                    "font_classification_character": characterScript(scalar) == script,
                    "font_postscript": runPS, "font_family": runCanonical ?? runFamily,
                    "requested_font_size_points": size, "actual_font_size_points": actualSizePoints,
                    "font_size_screen_px": actualSizePoints * Double(screenScale),
                    "source_screen_scale": Double(screenScale), "text_color_hex": textColorHex,
                    "actual_text_color_hex": textColorHex, "background_hex": backgroundHex,
                    "font_source_kind": fontSource["kind"] as? String ?? "ios_system",
                    "font_source_sha256": fontSource["sha256"] as? String ?? NSNull(),
                    "actual_font_family": runFamily, "font_match_verified": runPS == actualPostscript && runCanonical == expectedFamily && glyphs[i] != 0 && inside])
            }
        }
        result["glyphs"] = glyphRows
        result["font_runs"] = runs
        result["glyph_coverage"] = ["utf16_count": utf16.count, "font_get_glyphs_succeeded": coverageOK,
                                    "requested_font_glyph_ids": coverageGlyphs.map { Int($0) }, "zero_run_glyph_count": zeroGlyphs]
        result["fallback_detected"] = fallback
        result["clipped"] = clipped
        guard coverageOK, !coverageGlyphs.contains(0), zeroGlyphs == 0 else { return reject("missing_glyph_coverage") }
        guard !fallback else { return reject("coretext_font_fallback") }
        guard !clipped, box.insetBy(dx: -0.5, dy: -0.5).contains(inkBox) else { return reject("glyph_ink_clipped_or_outside_region") }
        context.saveGState()
        context.translateBy(x: 0, y: canvasSize.height)
        context.scaleBy(x: 1, y: -1)
        context.textMatrix = .identity
        context.textPosition = CGPoint(x: baseline.x, y: canvasSize.height - baseline.y)
        CTLineDraw(line, context)
        context.restoreGState()
        result["font_match_verified"] = true; result["status"] = "ok"; result["font_family"] = expectedFamily
        result["reason"] = "native_coretext_font_and_glyph_coverage_verified"
        return result
    }
}
