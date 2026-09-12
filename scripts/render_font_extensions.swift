// Offline macOS renderer. No source font or Swift runtime is needed by serving.
import Foundation
import CoreText
import CoreGraphics
import ImageIO
import CryptoKit

struct Source: Decodable {
    let font_id: String
    let path: String
    let postscript_name: String
    let source_sha256: String
}
struct Request: Decodable {
    let fonts: [Source]
    let characters: [String]
}
struct RenderedSource: Encodable {
    let font_id: String
    let source_sha256: String
    let actual_url: String
    let actual_postscript_name: String
    let sizes: [Int]
    let glyph_count: Int
}
struct RenderReport: Encodable {
    let renderer: String
    let operating_system: String
    let fonts: [RenderedSource]
}
func failure(_ text: String) -> NSError {
    NSError(domain: "FluxGlyphRenderer", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
}
let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
let output = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
var registered = Set<URL>()
var renderedSources = [RenderedSource]()
for source in request.fonts {
    let url = URL(fileURLWithPath: source.path)
    let sha = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    guard sha == source.source_sha256 else { throw failure("Source SHA mismatch: \(source.font_id)") }
    if registered.insert(url.standardizedFileURL).inserted {
        // A URL/name descriptor alone can silently select Helvetica for an
        // unregistered downloaded face. Registration is scoped to this process.
        // System faces can already be registered; the exact identity checks
        // below remain mandatory regardless of the registration return value.
        var registrationError: Unmanaged<CFError>?
        _ = CTFontManagerRegisterFontsForURL(url as CFURL, .process, &registrationError)
        if let error = registrationError { _ = error.takeRetainedValue() }
    }
    for size in [20, 28, 44] {
        let descriptor = CTFontDescriptorCreateWithAttributes([
            kCTFontURLAttribute: url, kCTFontNameAttribute: source.postscript_name
        ] as CFDictionary)
        let font = CTFontCreateWithFontDescriptor(descriptor, CGFloat(size), nil)
        guard CTFontCopyPostScriptName(font) as String == source.postscript_name,
              let actualURL = CTFontCopyAttribute(font, kCTFontURLAttribute) as? URL,
              actualURL.standardizedFileURL == url.standardizedFileURL else {
            throw failure("Font fallback: \(source.font_id) / \(CTFontCopyPostScriptName(font))")
        }
        for character in request.characters {
            var utf16 = Array(character.utf16)
            var glyph: CGGlyph = 0
            guard utf16.count == 1, CTFontGetGlyphsForCharacters(font, &utf16, &glyph, 1), glyph != 0 else {
                throw failure("Missing glyph: \(source.font_id) / \(character)")
            }
            var box = CGRect.zero
            CTFontGetBoundingRectsForGlyphs(font, .horizontal, &glyph, &box, 1)
            let width = 96, height = 96
            guard box.width > 0, box.height > 0, box.width < 90, box.height < 90 else {
                throw failure("Invalid or clipped glyph bounds: \(source.font_id) / \(character)")
            }
            guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                throw failure("Cannot create bitmap")
            }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.setFillColor(CGColor(gray: 0, alpha: 1))
            var position = CGPoint(x: floor((96 - box.width) / 2 - box.minX), y: floor((96 - box.height) / 2 - box.minY))
            CTFontDrawGlyphs(font, &glyph, &position, 1, context)
            let name = "\(source.font_id)_\(size)_\(character.unicodeScalars.first!.value).png"
            guard let image = context.makeImage(),
                  let destination = CGImageDestinationCreateWithURL(output.appendingPathComponent(name) as CFURL,
                                                                    "public.png" as CFString, 1, nil) else {
                throw failure("Cannot create PNG")
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw failure("Cannot save PNG") }
        }
    }
    renderedSources.append(RenderedSource(font_id: source.font_id, source_sha256: sha,
                                           actual_url: url.standardizedFileURL.path,
                                           actual_postscript_name: source.postscript_name,
                                           sizes: [20, 28, 44], glyph_count: request.characters.count))
    print("Rendered \(source.font_id)")
}
let report = RenderReport(renderer: "CoreText exact source URL/PostScript; process registration",
                          operating_system: ProcessInfo.processInfo.operatingSystemVersionString,
                          fonts: renderedSources)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
try encoder.encode(report).write(to: output.appendingPathComponent("render-report.json"), options: .atomic)
