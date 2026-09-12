# 模型交付约定

模型与网站/API 独立版本化。服务器加载当前选定的完整包，只执行推理。默认目录为 `models/`；若存在 `models/ACTIVE.json`，则读取其中安全相对路径 `path`，例如 `releases/r17-ios-hant-region-v1`。也可以通过 `FLUX_MODEL_DIR` 指定其他完整包目录。

区域模型（R16／R17）的默认流程为 PP 文字区域检测 → 区域字体 CNN／字号回归 → 原图颜色测量。保留文字位置检测，不进行文字识读；模型是否启用取决于所选完整包的清单。单独放入 ONNX 文件不会启用新算法。R17 数据来源和实际验证见 [简繁体训练](ios-traditional-training.md) 与 [验证结果](ios-traditional-results.md)；R16 架构历史见 [区域字体模型](ios-region-font.md)。

## 区域字体完整网站模型包

完整包包含以下受清单校验的文件：

| 文件 | 用途 |
|---|---|
| `MANIFEST.json` | 版本、模型方法及文件清单；每项含 `path`、`bytes`、`sha256` |
| `pp/onnx/paddle_ocr_det.onnx` | 从完整截图定位文字区域 |
| `pp/paddle_ocr_delivery.contract.json` | PP 检测输入与预处理配置 |
| `region_neural/metadata.json` | 字体类别顺序、区域模型契约、温度和门槛 |
| `region_neural/<model.path>` | 字体分类与字号回归双输出 ONNX |

`scripts/package_region.py` 仅复制检测模型、检测配置及区域模型／元数据，不复制 PP REC／CLS、识字字典、中文／Latin 参考档案或历史 `style/size_metrics.json`。颜色算法属于推理代码，随网站/API 一起部署。

模型 metadata 使用：

- `schema: flux-glyph-region-font-v1`、`algorithm: region-cnn64x256-v1`。
- `families` 指定输出类别顺序；类别数为 `C`，不能用操作系统默认字体猜标签。
- `model: {path, sha256}` 指向同目录单个 ONNX 文件。
- `temperature` 和 `gates.min_score`、`gates.min_margin`、`gates.min_patch_agreement` 控制候选门槛。
- `max_size_relative_spread` 限制不同图块字号估计的差异。
- `font_label_groups` 可显式声明 PingFang 字体族的 SC／TC／HK 原生成员；推理输出仍是模型实际训练的类别，不靠识字或名称猜测地区版本。旧模型可不提供该字段。
- `font_sources` 可记录每个类别的 `system`／`asset` 加载来源，用于页面及下载说明；Alipay Number 是应用字体资产，不是系统字体。这不是网络输入。

ONNX 输入 `tiles` 为 float32 `[N,1,64,256]`，批次数动态；输出顺序固定为 `logits [N,C]`、`log_em_ratio [N]`，均为 float32。输入是文字区域的图块，预处理保留字形比例、估计局部背景、提取墨迹并按高度缩放；长区域形成多个窗口。不能直接将原图任意拉伸至输入大小，也不传入 OCR 文本、字符下标或人工 script。

推理使用校验后的 ONNX 字节与单线程 CPU ONNX Runtime。区域分类先对每块应用温度 softmax，再聚合字体分数；只有分数、前两名差值和片段一致性达到门槛时，才输出 `candidate` 字体名。否则保留 `uncertain` 和具体原因，包括低分数、候选接近、片段不一致、过长、低质量或不均匀背景。

输出 `font_method`、`font.method` 为 `region_neural_network`，`font.scope` 为 `Detected text region`。`ocr_performed` 为 false，`text` 为 null／空字符串，`ocr_confidence` 为 null，`glyphs` 为空。候选使用 `{family, score}`，另有 `margin`、`patch_agreement`，不再输出参考距离。分数不是实际准确率，也不保证所有未知字体会被拒绝。

字号回归结合原图尺度输出 `text_style.font_size_px_estimate`，单位为截图像素，不是 iOS pt 或检测框高度。区域模型的 `font_size_px_interval` 为 null；文字颜色为原图中的可见 `#RRGGBB`。无可靠估计时相应字段为 null，`size`／`color` 给出状态原因。模型不恢复原始透明度，也不依据字体名推断设备或截图真伪。

## 重建、导入与回退

R17 字体权重和元数据保留目录为 `models/experiments/ios-hant-region-v1`；已验证完整包位于 `models/releases/r17-ios-hant-region-v1`。准备好权重后，在装有项目依赖的环境运行：

```sh
PYTHONPATH=src python scripts/package_region.py \
  --region models/experiments/ios-hant-region-v1 \
  --output artifacts/rebuilt-r17-bundle \
  --version r17-ios-hant-region-v1
```

`--base` 默认使用根目录 `models` 的当前活动包，从中提取检测模型。输出目录必须为新目录；打包时校验源模型及最终包，并做运行时兼容检查，不重新训练。

```sh
python scripts/model_release.py \
  --model-root models export \
  --output artifacts/flux-glyph-r17-ios-hant-region-v1.zip
python scripts/model_release.py install \
  artifacts/flux-glyph-r17-ios-hant-region-v1.zip --version r17-ios-hant-region-v1
# 安装/激活后重启；同时更新代码时重建镜像
docker compose up -d --build
# 回退根目录内保留的原始包
python scripts/model_release.py activate bundled
docker compose restart api
```

导出 ZIP 只包含 MANIFEST 和清单指定文件。导入先验证安全路径、SHA、文件大小及运行时兼容性，再激活新目录；旧目录保留。切换后须重启进程，防止同一进程混用不同版本。包中的 Python 文件不作为网站推理代码执行；算法契约变更须同时交付配套运行时代码。

`scripts/build_release.py --model-root models --output PATH_TO_NEW_ZIP` 生成包含配套网站源码的部署 ZIP，将当前活动模型展开为 ZIP 内的 `models/MANIFEST.json` 及其文件，不依赖原机器的 `ACTIVE.json` 路径或 `artifacts/`。部署 ZIP 与单独字体推理下载包用途不同。

`compose.yaml` 挂载根目录 `models/`，通过其中随源码提供的 `ACTIVE.json` 选择版本。`compose.ios.yaml` 只覆盖本地镜像名称，继承相同挂载，因此标准部署和本地部署均不依赖 `artifacts/`。

## 单独下载的字体推理工具包

`GET /api/models/font` 返回当前区域模型的信息，包括 `available`、`version`、`families`、`input_shape`、`download_url`、`bytes`、`sha256`、`ocr_required: false`、`usage`。未加载区域模型时 `available` 为 false，`GET /api/models/font/download` 返回 404。

下载 ZIP 包含字体 ONNX、`metadata.json`、`inference.py`、`text_style.py`、`predict.py`、`requirements.txt`、README 和 `SHA256.json`。它不含检测模型、用户上传图片、环境文件或训练数据，不能替代上面的完整网站包直接导入。

解压后，在其目录运行：

```sh
pip install -r requirements.txt
python predict.py text-region.png
```

输入一张文字区域裁图，脚本执行包内预处理并返回字体、字号和颜色；完整截图请先通过网站/API 定位。也可调用 `inference.RegionFontClassifier`，元数据与 ONNX 必须保持在同一目录。推理仅需 ONNX Runtime、NumPy、Pillow，不需要 PyTorch 或 OCR。

ZIP 包含逐文件 SHA256 清单；API 返回整个 ZIP 的 SHA256，下载响应同时带 `X-Model-SHA256`。首次生成工具包时再次核验模型字节，之后该服务实例缓存已核验的 ZIP；磁盘文件后续变化不会混入已有下载内容，模型切换须重启。

两路模型接口继承现有 API 的访问令牌保护：网页使用解锁 Cookie，程序使用 `Authorization: Bearer YOUR_TOKEN`，不支持将令牌放在下载 URL 中。

## 精度验收

冻结权重、门槛与测试集后，分别统计检测匹配、漏检、多余框，字体名正确／错误／待确认，以及字号、颜色的误差和可输出数量。字体真值须由原生绘制字体证据或可信人工标注支持；状态栏推测、旧模型输出和截图真假标签不是字体标签。

R16 不执行文字识读，评测不能把空 text 当作 OCR 失败再混入字体准确率。受控 iOS Simulator 截图的成绩应明确采集域，不能扩展为第三方 App、真机或未知字体的准确率。历史 R14／R15 的字形或识字依赖评测保留原口径，不归入 R16 成绩。

## 历史兼容：R14 参考字库

未声明 `region_neural/` 的旧包继续走历史加载逻辑。R14 完整包包括 PP DET／REC／CLS、字符字典、`font/metadata.json`、`font/GATES.json` 及紧凑字形档案；可选 `latin/metadata.json`、`latin/GATES.json` 和 `latin/references.zip`。这些文件须列入根 MANIFEST。

中文档案 schema 为 `flux-glyph-r13-compact-v1`，每个汉字 NPY 数组 `[31,3,32,32] uint8`，对应字体样式及字号 `[20,28,44]`。Latin schema 为 `flux-glyph-latin-candidates-v1`，逐字符数组 `[17,3,32,32] uint8`，字号 `[20,32,48]`。加载器检查成员、NPY 头、形状、覆盖和哈希，禁止 pickle。旧包返回 `{family, distance}`，适用旧苹方 `supported` 或其他字体候选规则。

旧 PP 契约包括 RGB 输入、DB 检测及 PP REC 6625 类 CTC 识字；替换字典、预处理或输出形状须同步更新对应历史运行时代码。参考库构建依赖本地源字体，但服务端不需要 macOS、CoreText、Swift 或 fontTools。

历史结果和方法见 [移动字体改进](mobile-font-improvements.md)、[中文精度报告](accuracy-report.md)。

## 历史兼容：R15 单字字体 CNN

旧 CNN 包声明 `neural/metadata.json` 及单个 ONNX，schema 为 `flux-glyph-neural-font-v1`、algorithm 为 `glyph-cnn64-v1`。输入 `glyphs` 为 float32 `[N,1,64,64]`，输出 `logits [N,C]`；元数据包含中文／Latin 分类范围、温度、分数与差值门槛。

该旧流程需要 OCR 与原像素分字，同字重复先平均，再对不同字符等权平均；分字不完整、低识字置信度或低模型分数会待确认。结果方法为 `neural_network`。旧截图训练包还可声明 `style/size_metrics.json`，由训练字形建立字号比例。

这些规则仅描述历史兼容，不用于 R16 的区域 CNN。历史来源和成绩见 [iOS 单字截图训练](ios-screenshot-training.md)、[历史截图评测](ios-screenshot-results.md)、[更早合成实验](neural-font-training.md)。
