# 导入有字体名称标注的截图

`training/prepare_screenshots.py` 是后续真实截图微调的离线数据入口。本轮没有通过它加入真实字体真值截图；自动测试使用明确标记的源字体合成样本。

每行 JSONL 描述一张图片。`image` 可以是绝对路径或相对 JSONL 文件的路径；`source_id` 是原图/采集组的稳定标识。同一原图的导出、缩放和裁切派生物应沿用同一个 `source_id`，不能为进入不同 split 而改名。

```json
{"image":"screenshots/sample.png","source_id":"capture-session-001-page-01","split":"train","regions":[{"bbox":[20,100,260,150],"text":"Safari浏览器","font_family":"PingFang SC","script":"han"},{"bbox":[20,100,260,150],"text":"Safari浏览器","font_family":"SF Pro","script":"latin"}]}
```

以上是字段示例，不能当作实际图片的字体真值。字体名称需要由采集应用实际加载的字体文件、渲染日志或可靠标注提供；填写手机系统/品牌不等于提供字体标签。一个 `script` 内的字如果使用不同字体，应分成各自有明确字体标签的区域。

- `bbox` 必须是 `[left, top, right, bottom]` 四个整数，位于 EXIF 方向修正后的原图内，右/下边界不包含在区域中。文本周围应有原始背景余量。
- `text` 是完整区域的文字，用来核对 OCR；只允许忽略空白差异，不做模糊匹配或用标注文字覆盖错误 OCR。
- `font_family` 必须是 `training/network.py` 的已知家族，且属于该文件 `SCRIPTS[script]` 的支持范围。`ios`、`android`、缺失标签或不支持的字体/脚本组合会被拒绝。
- `script` 为 `han` 或 `latin`。Han 当前支持 U+4E00..U+9FFF；Latin 为 ASCII 字母和数字。混排中的另一类字符不会继承该标签，标点不导出为训练字形。
- `split` 为 `train`、`calibration` 或 `test`，必须在训练前确定。可选 `font_face` 记录已知具体字重/face；缺失时导出 `family:unspecified`，不会猜测字重。
- 可选 `source_kind` 用来明确标记 `source_font_synthetic_smoke` 等合成来源；导入器不会把标注本身认证成真实设备或字体来源证明。

在具备项目 OCR 依赖（包括 ONNX Runtime、OpenCV）和本地 `models/pp` 的环境执行：

```sh
python training/prepare_screenshots.py --input labels.jsonl --output prepared-real-fonts --model-dir models
```

OCR 读取区域、确认文字后，使用 CTC 顺序和原图墨迹间隔定位字形；仅接受可靠的字框，再调用与神经网络推理相同的 `extract_glyphs` 预处理。缺失边界、低 OCR 置信度或提取失败的字符保留拒绝记录，不生成均分字框。此步骤与 PyTorch 训练进程分开，训练进程不需要加载 OCR。

输出目录必须为空或不存在，包含 `train.npz`、`calibration.npz`、`test.npz` 和 `manifest.json`。NPZ 的 `x` 为 float32 `[N,64,64]`，`y` 为按 `FAMILIES` 顺序编码的 int64；另含 `chars`、`faces`、`sizes`、`scripts`、`source_ids`。`sizes` 是分字后的原图墨迹高度，不是字体字号。空 split 保留形状 `[0,64,64]`。

清单记录源文件与像素 SHA、标签、OCR 模型文件、每个字的原图坐标和拒绝原因。源 ID、源文件 SHA、解码原图像素或区域像素跨 split 重复时，在写数据前拒绝；这些检查不能发现所有缩放派生或近似重复，所以仍须提供稳定来源组 ID。人工标注不应由旧字体匹配器或状态栏模型自动补齐。

训练读取时使用 `training/real_data.py` 的 `PreparedScreenshotData` 再核验三份分区 SHA、字体顺序、字符/脚本和标签对应、来源 ID 及逐字清单绑定，不直接接受任意 NPZ。训练集为空会报错；空 calibration/test 需要调用者明确设置 `allow_empty_evaluation=True`，审计记录没有实图校准/测试结论。预检查会读取 test 的标签、来源等元数据与数组头来验证隔离性，只有选模完成后调用 `datasets('test')` 才加载其字形像素。加载器不认证标注真伪，也不接受测试中注入的模拟 OCR 结果作为训练输入。

官方 Roboto 数据源可以独立重建：

```sh
python training/download_sources.py
python training/download_sources.py --verify-only
```

默认输出位置与合成训练脚本一致，也支持 `--output DIRECTORY`。固定来源及许可证位于 `docs/sources/roboto/`；不包含字体二进制，也不会下载 AdobeVFR 研究许可数据。
