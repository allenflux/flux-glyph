# iOS 区域字体网络（R16）

此文保留 R16 发布时的验收记录。当前版本见 [R17 简繁体、英文和数字验证](ios-traditional-results.md)。

本版保留 PP DB 文字区域检测，去掉文字识读、CTC 解码、逐字切分和参考字库匹配。字体族由训练后的卷积神经网络读取区域像素预测；同一个网络另有字号回归输出。颜色仍从原图像素测量。

本地已部署 `r16-ios-region-v1`。完整截图成绩、导出及下载验证和适用范围见 [验证结果](ios-region-results.md)。

完整流程为：原始截图 → 检测文字区域 → 区域图块 → 字体／字号 CNN → 分数门槛与结果。输入不包含文字内容、汉字／英文类别提示或原生字体标签。定位检测器来自已有 PP 模型，本轮训练的是字体／字号网络。

## 数据与训练

数据来自受控原生 UIKit/CoreText 页面在 iOS Simulator 中实际运行后，通过 `simctl` 保存的 PNG。页面内容和布局由采集程序生成；不属于第三方应用或真实手机采集。使用此前固定的 1,000 张截图和分组，未重新渲染成字体图片来替代截图。

| 分组 | 截图 | 原生文字区域 | 输入图块 |
| --- | ---: | ---: | ---: |
| 训练 | 800 | 8,807 | 22,545 |
| 校准 | 100 | 1,112 | 2,870 |
| 固定回归测试 | 100 | 1,097 | 2,773 |

八个字体族是 HarmonyOS Sans SC、MiSans、Noto Sans CJK SC、OPPO Sans、PingFang SC、SF Pro、Helvetica、Alipay Number。非系统字体通过采集应用加载到 iOS 中；这些样本仍是 iOS 截图。Alipay Number 样本只覆盖数字。

每个区域的字体和字号标签来自实际使用的原生 CTFont，而非旧模型预测。准备数据时核对截图 SHA、像素 SHA、字体文件、字体回退和原生样式；以截图、内容分组、文本及重复像素检查分组隔离。标签中的文字仅用于离线审计，模型的预处理和推理不读取这些文字。

网络从 R15 已训练的字形 CNN 权重初始化，改为读取宽区域图块，增加字号回归头，再执行 3,000 次 AdamW 更新，每批 64 个图块。训练按字体／区域采样，损失为八类交叉熵与字号对数比例的 Smooth-L1；没有按识读内容限制字体候选范围。

每 250 步只用校准组选择权重。温度、分数、前两名差值门槛也只用校准组确定；协议和权重选择先冻结，之后才运行测试评分。本轮测试复用了此前 R15 测试页面，因此报告为固定回归验证，不能称为新盲测。

训练脚本为 `training/train_regions.py`，输出目录为 `artifacts/ios-region-font-v1`，其中 `TRAINING_FREEZE.json`、`SELECTION_FREEZE.json` 和 `report.json` 记录来源、参数变化、选择和结果。预处理代码与训练代码都有 SHA 并保存冻结副本。数据校验与重训命令见 [训练复现说明](ios-region-font-training.md)。

## ONNX 与独立使用

页面提供的字体 ZIP 只包含字体／字号网络，不包含完整截图检测器。解压后可以独立运行：

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py text-region.png
```

Windows 的环境激活命令为 `.venv\Scripts\activate`。输入是一行或一个文字区域的裁图；直接使用整张截图时，应先检测并裁出文字区域。推理依赖 NumPy、Pillow、ONNX Runtime，无需 PyTorch 或 OCR 识读模型。

ONNX 输入 `tiles` 是 float32 `[N,1,64,256]`；输出为 `logits [N,8]` 和 `log_em_ratio [N]`。类别顺序、模型 SHA、温度和门槛都在 `metadata.json`。下载 ZIP 的 `SHA256.json` 覆盖其模型、代码和说明。

必须调用包内 `inference.py` 的预处理：估计背景、提取墨迹、保留宽高比例将墨迹高度归一化到 56，放入高 64 的画布，再取最多 8 个宽 256 的图块。不能将原图直接拉伸到输入大小。整行分数为各图块温度 softmax 的均值；字号为原图墨迹高度乘以 `exp(median(log_em_ratio))`。

`font.family` 为字体候选，证据不足则返回 null；候选分数不等同于真实准确率。`font_size_px_estimate` 是截图里的像素字号估计，不是 iOS pt。字号无已校准置信区间，`font_size_px_interval` 返回 null。颜色输出是原图可见颜色，透明度叠加后的颜色无法据此还原为应用源码中的颜色值。

## “待确认”的含义

待确认表示区域已经定位，但无法可靠输出字体名称。具体原因随结果返回：

- `below_score_gate`：模型最高分未达到校准门槛。
- `ambiguous_neural_families`：前两名太接近。
- `mixed_or_ambiguous_region`：各图块的判断不够一致；这不是已验证的混合字体检测器。
- `low_quality_region` / `nonuniform_region_background`：像素证据不足或背景不符合当前预处理条件。
- `region_too_long`：图块上限不足以连续覆盖整行。
- `too_many_regions`：超出本次处理数量上限。

候选首位只表示当前最像哪一类。未知字体即使高分也可能被分到已训练类别；未知字体、混合字体、第三方应用和真机截图的准确率需要额外标注验证。

## 本地服务与验证

完整 R16 模型包只声明 PP 检测器、检测契约及 `region_neural` 模型和元数据。推理启动时不创建文字识读器，也不加载字典、参考字库或旧单字 CNN。

```sh
PYTHONPATH=src .venv-dev/bin/python scripts/package_region.py \
  --region models/experiments/ios-region-v1 \
  --output artifacts/ios-region-font-v1/bundle-stable --version r16-ios-region-v1
docker compose -f compose.yaml -f compose.ios.yaml up -d --build
```

`GET /api/models/font` 返回模型版本、字体族、输入形状和 ZIP SHA；`GET /api/models/font/download` 下载独立包。启用令牌时，两者均需要网页 Cookie 或 Bearer 认证。

`scripts/evaluate_region_fonts.py` 用原始完整截图评估检测和字体／样式结果，以几何位置配对原生真值。验证时主动禁止文字识读、CTC 解码、逐字切分和旧字体分类器；任何调用都会失败。既报告正确字体名，也报告错误、待确认、漏检及字号／颜色覆盖率。
