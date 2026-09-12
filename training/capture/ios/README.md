# iOS 原生字体截图采集 App

`FluxFontCapture.xcodeproj` / scheme `FluxFontCapture`，bundle ID `tech.fluxglyph.capture`。UIKit 的 `UIView.draw` 内使用 CoreText `CTLineDraw`，字体由 iOS `UIFont` 直接桥接，像素由模拟器实际渲染。App 不生成替代截图，不使用桌面 Pillow 渲染字形。

这是受控采集 App，不是支付宝 App，也不代表物理 iPhone 实拍。截图必须由外部 `xcrun simctl io <UDID> screenshot ...` 从实际模拟器画面取得。图片/字体标签来自这次实际绘制的记录。

## 场景与切页

启动时优先读取 `Documents/Scenes.json`，否则读取 App Bundle 内的 `Scenes.json`。批量采集通过原子写入 `Documents/command.json` 切页，例如 `{"page_index": 1, "request_id": "capture-run-page-1"}`。App 每 100ms 检查一次，只处理新 request_id，在 ready 中原样回显；每次处理都重新读取场景文件。这样切页留在 App 内，不触发 iOS URL 打开确认框或切入动画。URL `fluxcapture://page/<index>` 和 `fluxcapture://reload` 仅保留调试用途；未知 JSON 字段不会影响解析。

```json
{
  "schema": "flux-glyph-capture-scenes-v1",
  "platform": "ios",
  "canvas_points": [402, 874],
  "fonts": [],
  "pages": [{
    "id": "page-0",
    "split": "train",
    "background": "#FFFFFF",
    "regions": [{
      "id": "title",
      "text": "账单详情",
      "font_postscript": "PingFangSC-Semibold",
      "font_family": "PingFang SC",
      "script": "han",
      "font_size": 26,
      "weight": 600,
      "color": "#15191F",
      "bbox_points": [32, 112, 370, 162]
    }]
  }]
}
```

页面按 `min(viewWidth/canvasWidth, viewHeight/canvasHeight)` 等比居中绘制，字号与坐标使用同一变换。单行文字在给定区域内垂直居中、左对齐，不自动缩小字体。纯中文或纯数字/英文区域允许标点、空格；不支持混合脚本、换行及代理码位。超出区域的字形会拒绝该区域。

系统 SF 字体使用 `font_postscript: "SYSTEM"` 或 `"-system"`，`weight` 支持 CSS 300/400/500/600/700 或 light/regular/medium/semibold/bold。具名字体以 PostScript 名称为准，weight 不会擅自替换具名字体。直接桥接 UIFont，避免把私有 `.SFUI-*` 名称重新查找后错误回退到 Times。

## 外部字体

将原始 TTF/OTF 放入 `Documents/Fonts/<basename>`，并在 Scenes 的 fonts 中声明：

```json
{"kind":"asset","path":"AlipayNumber-Regular.ttf","postscript":"AlipayNumber-Regular","family":"Alipay Number","sha256":"源字体的64位小写SHA256"}
```

仅接受 basename。注册前核对 SHA256 与字体描述符 PostScript；采用 `CTFontManagerRegisterFontsForURL(..., .process)`。实际 UIFont/CTFont 必须使用相同源 URL，CoreText 每个 run 也必须保持相同 PostScript、有效 glyph ID 和完整覆盖。未注册、SHA 不符、字体回退、缺字或裁切都会拒绝该区域。类名映射只对本次 manifest 已验证的资产生效。

## 完成记录

每次切页先删除旧 `Documents/frame-ready.json`。绘制结束后等待至少 6 个 CADisplayLink 回调和 0.15 秒，且 App 必须处于 active 已满 0.8 秒、窗口可见并为 key window，再原子写入新 ready 文件。重复重绘会更新待写报告，不会无限重置等待器。驱动应核对 `request_id`、`page_index`、`page_id`、`scene_manifest_sha256`、`application_state == "active"` 和 `status == "ready"`，随后采集 PNG。初次启动仍应由驱动额外等待系统动画稳定；错误状态会直接写 `status: "error"`。

顶层记录包含真实 `pixel_size`、`screen_scale`、画布缩放/偏移、safe area、OS 版本、场景文件 SHA、注册资产结果及所有区域。

字号和颜色记录来自实际原生绘制：`requested_font_size_points` 是场景请求的画布字号，`ct_font_size` 是 CTFont 的原始字号，`actual_font_size_points = ct_font_size × canvas_scale` 是屏幕点大小，`font_size_screen_px = actual_font_size_points × source_screen_scale` 是源截图像素字号。region 和 glyph 都记录实际字号、`text_color_hex` / `actual_text_color_hex` 和 `background_hex`，颜色为已解析的 `#RRGGBBAA`。这些是采集真值，不能据此声称任意外部截图的真实 iOS pt 已知。

区域仅在 `status == "ok"` 且 `font_match_verified == true` 时可用于训练，还必须保留/检查：

- `actual_font_postscript`、`actual_font_family`、请求字体、实际字号。
- `fallback_detected == false`、`clipped == false`。
- `font_runs`：实际 CTRun PostScript/family、glyph IDs、UTF16 string indices。
- `glyph_coverage`：请求字体直接映射的字形 IDs、是否完整覆盖、实际 run 是否出现零 glyph。
- `font_source`：内置字体的 iOS/实际字体信息；资产字体另有 SHA 验证、注册结果、实际 URL 相等的证明。
- `glyphs`：每字 character/text_index/glyph_id、屏幕像素 bbox、实际字体与验证状态。坐标由 `CTFontGetBoundingRectsForGlyphs` 和 `CTRunGetPositions` 转换得出，未人为添加边框。

标点与空格仍保留绘制证据；训练裁字只应使用 `visible` 与 `font_classification_character` 都为 true 的字形。`text_index` 使用 UTF16；允许的中文/ASCII 都属于 BMP，因此与当前字符串字符下标一致。资产字形另有 `font_source_kind` 和 `font_source_sha256`，内置字体的 SHA 字段为 null，不虚构系统字体文件哈希。

`Documents/font-inventory.json` 包含当前 UIFont family/PostScript 清单、系统各字重实际字体名及已注册资产。安装或重装 App 后请重新通过 simctl 获取数据容器路径。

## 简体、繁体与 PingFang 原生区域名称

具名字体支持 inventory 中实际存在的 `PingFangSC-*`、`PingFangTC-*`、`PingFangHK-*`；必须同时匹配实际 CTFont family 和每个 CTRun，不会把 TC/HK 归为 SC。region 可提供 `language`：中文为 `zh-Hans`、`zh-Hant`、`zh-Hant-TW`、`zh-Hant-HK`，英文/数字为 `en`。该字段写入 CoreText shaping attribute，并记录 `requested_language` 与每个 run 的 `language`，用于离线来源核对。可选 `han_orthography` 仅描述受控文字词库，不参与推理。

SC/TC/HK 是原生来源标签，并不保证可以从任意截图像素辨别。实际同文本测试中 TC/HK 可以完全同像素，因此区域网络可以输出 PingFang 家族并保留原生细分真值，不能按简繁文字直接猜测地域字体。参见 `docs/ios-traditional-capture.md`；诊断场景不用于训练。
