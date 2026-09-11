# 模型交付约定

模型与网站/API 独立版本化。服务器只加载当前选定的包，不在线扩字或训练。默认加载 `models/`，若存在 `models/ACTIVE.json` 则使用其安全相对路径 `path`，如 `releases/experiment-002`。

完整包至少包含：

- `MANIFEST.json`：`files` 数组，每项 `path`、`bytes`、`sha256`；可选 `version`。
- `pp/paddle_ocr_delivery.contract.json`、`pp/onnx/paddle_ocr_{det,rec,cls}.onnx`、`pp/charset/ppocr_keys_v1.txt`。
- `font/metadata.json`、`font/GATES.json`、metadata 指定的紧凑字形档案。

当前 PP 契约是 RGB 输入、DB 区域检测、PP REC 6625 类 CTC 输出。改变字典、预处理或输出形状时，必须一起更新相应运行时代码并验收；替换任意 ONNX 并不自动兼容。

当前字形档案 schema 为 `flux-glyph-r13-compact-v1`。每个汉字一个 NPY 成员，数组为 `[31,3,32,32] uint8`，对应 metadata 的 `font_ids` 顺序及字号 `[20,28,44]`。预处理为固定背景归一化、墨迹提取、64 像素字形、uint8、高斯模糊 1、缩放 32；读取后按原有 float64 归一化规则匹配。档案不是有损压缩的浮点权重。

运行时只读取受校验的模型和数组（NPY 禁止 pickle）；不执行模型包中的 Python 脚本。更换算法时应升级 `algorithm_version` 和兼容代码。压缩优化须证明得分和预期输出没有发生不受控变化。

## 精度验收

冻结版本与测试集后，再统计：字体名已知样本的总正确率、已输出字体结果的准确率、输出覆盖率、按字体/字号/背景的错误和待确认数。真实截图应有逐框字体标签；状态栏推测、旧模型输出或截图真假标签不是字体真值。

本地模型导出使用 `scripts/model_release.py export`。ZIP 只包含模型清单指定文件；导入先核对每个 SHA、文件大小及兼容性，然后激活新目录。`activate bundled` 可回退项目自带版本。导入和切换后需要重启 API，避免同一进程混用不同模型。
