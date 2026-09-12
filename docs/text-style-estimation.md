# 文字颜色与像素字号

R16 区域版复用本文的原图颜色测量；字号改由区域 CNN 的回归头预测，不再使用下面的逐字 `SizeMetrics`。R16 输出截图 px，并将未校准的 `font_size_px_interval` 留为 null，详见 [区域模型](ios-region-font.md)。下文的逐字字号统计及误差记录属于 R15。

`src/flux_glyph/text_style.py` 直接读取原始 RGB 截图及绝对坐标，不使用神经网络输入的 64×64 归一化字形来推断绝对大小。`estimate_text_style(image, glyphs, family=..., metrics=..., region_bbox=...)` 返回 `text_color_hex`、`font_size_px_estimate`、`font_size_px_interval` 和各自质量记录。缺乏可靠依据时返回 null。

颜色来自局部背景之外的稳定笔画核心像素，支持浅/深背景、彩色文字及抗锯齿；背景变化大、多种文字颜色、对比度过低会拒绝。文字无法分字或字体尚未覆盖时，仍可独立从 OCR 区域估计稳定颜色。返回的是截图中观察到的 RGB，不恢复透明度，也不保证等于压缩、混色前的应用源颜色。

字号使用原图实测笔画像素高度除以对应字体、字符的 `ink_height/em` 统计。`SizeMetrics` 仅从已核验的 iOS Simulator **train** 原 PNG、原生真实字号和字体标签建立；不直接用 CoreText 的包围框高度代替像素测量。每项至少 3 个字形、2 个来源截图；统计范围太宽会拒绝。未见英文/数字可使用同字体相近字形组，汉字因“一”和“国”等高度差异不使用全汉字平均值。字体身份来自已接受结果；普通逐字 `family_candidate` 不会自动用于字号，混合脚本可显式传已接受的 `style_family`。

输出单位是源截图像素 em 大小，包含截图自身缩放；不输出 iOS pt。区间是训练渲染差异与约 1px 测量误差形成的经验范围，不是统计置信区间。区域有明显混合字号时拒绝给出一个平均字号。该方法的误差还取决于上游字体分类和分字是否正确。

建立与独立评估：

```sh
.venv-dev/bin/python training/build_size_metrics.py \
  --data artifacts/ios-native-run/prepared \
  --output artifacts/ios-native-run/bundle/style/size_metrics.json
.venv-dev/bin/python training/evaluate_text_style.py \
  --data artifacts/ios-native-run/prepared \
  --metrics artifacts/ios-native-run/bundle/style/size_metrics.json \
  --split calibration --output artifacts/ios-native-run/style-calibration.json
```

评估脚本不修改统计或门槛；`--split test` 用于冻结后的测试集。报告同时列覆盖率、颜色通道误差、像素字号 MAE/相对误差和区间覆盖率。其输入使用原生真字体、真字形框，属于条件评估，不能替代完整 OCR 与字体识别流程的端到端评测。

验证包含 14 项单元用例：浅/深背景、等亮度彩色文字、JPEG、抗锯齿、未知字体、缺字、低对比度、混色、渐变背景、混合字号、重复字形、严格训练集限制和原图缩放。两页真实原生 smoke 的 24 个区域颜色均恢复为已知的 `#111111`；该 smoke 重用了训练样本，只用于坐标/测量流程核验，不能视作独立准确率。
