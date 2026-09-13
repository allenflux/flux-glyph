# Flux Glyph

上传 iOS 或 Android 截图，定位文字区域，使用**一个联合训练的字体／字号 CNN**识别字体，并测量文字颜色。网页不需要选择手机系统；字体识别使用一个 ONNX、一套共同评分规则。区域检测器只负责定位，不进行 OCR 识字。网络只输入区域图像，不需要文字内容或字符切分。

网站和 API 使用 **9000** 端口。模型下载放在上传区上方，使用说明统一为 `/docs#font-model`。

**本地实验模型 `r21-unified-font-v1-preview` 已完成 6,000 步联合训练和单 ONNX 导出，一致性验证通过，稳定验收未通过。** CAL 命名精度 95.07%、已知正确覆盖 63.34%；这些是选模型用的校准结果，不是盲测准确率。开发回归已完成，命名精度 96.49%、已知正确覆盖 51.66%，稳定条件仍未通过；这些旧 TRAIN 留出的图片可能被初始化网络见过，不是盲测。当前实际可用版本以 `/api/health` 和 `/api/models/font` 为准；服务器只推理。

后续模型修复在本地进行。加宽网络、字重均衡和类别间距辅助训练均已完成，分别通过 41、43、45／46 项保留验收，尚未替换 R21。下一轮正在本机调整已知字体的教师约束，尚无验收结果。见[容量与字重修复记录](docs/unified-font-capacity-repair.md)。

## 下载字体模型并单独使用

```sh
curl http://localhost:9000/api/models/font
curl -o flux-glyph-font-onnx.zip http://localhost:9000/api/models/font/download
```

解压 ZIP 后，在其目录安装依赖并输入一个文字区域裁图：

```sh
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python predict.py text-region.png
```

联合模型包包含一个字体／字号 `model.onnx`、`metadata.json`、预处理与推理代码、颜色测量代码、`predict.py`、依赖说明和 SHA256 清单。Python 3.10+，推理依赖 ONNX Runtime、NumPy、Pillow，不需要 PyTorch。一个模型同时处理两端截图，不附带第二个字体编码器、独立拒识网络或复核网络。下载包不包含用户上传图片和训练字体源文件。

输入应是一行或一个文字区域。`predict.py` 保留原始比例，生成 float32 `[N,1,64,256]` 图块，再按元数据中的完整类别顺序、温度和共同门槛聚合。ONNX 输出 `logits [N,C]` 与 `log_em_ratio [N]`；联合训练有 24 个具名字体和一个未知类别。不要把整张截图直接拉伸为单个图块。完整截图可交给网站或异步 API：

```sh
curl -F 'file=@screenshot.png' http://localhost:9000/api/jobs
curl http://localhost:9000/api/jobs/TASK_ID
```

请求不需要平台参数。模型信息接口包含 `available`、`version`、`families`、`download_url`、`bytes`、`sha256` 和 `usage`，具体类别与状态以当前文件为准。版本更新后重新识别，历史任务不会自动重算。启用访问令牌时，网页下载使用解锁后的 Cookie；命令行附带 `Authorization: Bearer YOUR_TOKEN`，不把令牌写入 URL。

## 联合字体范围和证据

具名范围包含 PingFang、SF Pro、Helvetica、HarmonyOS Sans SC、MiSans、OPPO Sans、Alipay Number、Noto／Source Han 黑体与宋体组、Roboto、LXGW WenKai、WenQuanYi Micro Hei、三款 ZCOOL 字体、Ma Shan Zheng，以及 FandolHei／Kai／Song、Long Cang、Zhi Mang Xing、Kaiti SC、Songti SC、HanziPen SC。完整名称顺序和采集域见 [联合来源](docs/sources/unified/README.md)。

PingFang 合并 SC／TC／HK 字体族，不从相同字形强行区分地区版本。FandolKai 与 Kaiti SC、FandolSong 与 Songti SC 保留不同类别。字体名称不能确定设备系统。Alipay Number 是应用数字字体；Kaiti SC、Songti SC、HanziPen SC 在本批采集中也是应用加载资产，不代表 iOS 系统内置。

数据来自经过实际字体核验的 iOS Simulator 和 Android Emulator 原生页面。联合 TRAIN 为 70,128 个视图／19,803 个原生区域；CAL 为 18,672／5,424；开发保留集为 3,504／1,092。缩放和 JPEG 版本属于相关视图，不是独立采集。没有将用户截图中的字体猜测用作训练真值。

新增具名类别的开发分区由旧 TRAIN 的整页／文本连接组件预先划分，历史初始化网络可能已经见过这些图片，不能称为盲测。第二批 Android TEST 继续封存，也没有覆盖全部联合类别。未知类帮助保留未覆盖情况，不保证拒绝或识别所有未知字体。95% 命名精度是校准选择偏好，稳定通过条件另外记录；模型分数不是实际准确率。

CAL 中 iOS 原生来源的命名精度／正确覆盖为 98.97%／75.26%，Android 为 90.25%／52.13%。部分新类别覆盖仍低，不能承诺截图中的所有手写字体、金额或时间都已改善。

完整数据、单模型训练、实际校准和导出结果、固定选择条件及限制见 [联合训练说明](docs/unified-font-training.md) 与 [精简数据审计](docs/sources/unified/DATA_AUDIT.json)。旧模型的评测不会作为联合模型成绩。

## Docker Compose 部署

先在本地训练并验证模型，将代码、完整模型包和 `models/ACTIVE.json` 一起提交到 `main` 并推送。服务器只通过 Git 同步已验证的提交，然后重建推理容器；不在服务器训练。`compose.yaml` 默认只读挂载 `./models`，无需复制本地训练的 `artifacts/` 目录。

```sh
cp .env.example .env
# 可编辑 .env 设置 FLUX_API_TOKEN；留空时无需令牌。
docker compose up -d --build
docker compose logs -f api
```

已有部署更新时，先确认工作区干净，再执行 `git pull --ff-only origin main` 和 `docker compose up -d --build`。如果远端包含未经验证的其他提交，应先核对提交号。更新后检查 `/api/health` 的版本和 `/api/models/font` 的下载信息。

公开访问地址为 `http://allenflux.tech:9000/`，服务与上传均使用 9000 端口。健康检查为 `/api/health`，API 文档为 `/docs`。设置令牌后，网页可输入相同令牌解锁；程序请求使用 `Authorization: Bearer <令牌>`。

本地 `compose.ios.yaml` 只设置镜像名称，继承相同的 `./models` 挂载和活动模型选择：

```sh
docker compose -f compose.yaml -f compose.ios.yaml up -d --build
```

默认资源配置面向 **2 核 / 2 GB** 主机：一个推理任务运行、最多八个等待；单线程 ONNX 会话与 BLAS；容器最多 2 核、1536 MiB 内存，为主机预留空间。限制单图 8 MB / 1200 万像素；最多处理 200 个文字区域，超出区域仍保留框并标明未处理。此处是部署配置，不代表峰值资源验收成绩。

上传数据保存在 `glyph-data` 命名卷中，每次上传有独立任务目录，包含上传文件、原图、区域裁图、标注 PNG 和 JSON。默认最多保留 **7 天（168 小时）**，并设 **100 个已结束任务**的数量上限；超过上限时提前删除最旧的结果。

服务启动、任务结束及每分钟检查一次过期数据，整组删除过期任务；排队和处理中的任务保留，无需额外设置 cron。`.env` 中的 `FLUX_RETENTION_HOURS`、`FLUX_MAX_SAVED_JOBS` 可调整保留策略，修改后执行 `docker compose up -d`。清理后原结果、裁图和下载链接返回 404。

`docker compose down` 不会删除数据卷；只有确实要清空结果时才使用 `down -v`。清理仅针对任务数据，模型保持只读。

## 页面、排队与字体结论

结果上方展示原图与区域缩略图列表。点击框或 `R001` 等区域编号，展开对应裁图、字体候选、字号与颜色，支持收起；切换区域不会自动滚动整页。JSON 保留原始返回内容。页面不展示识字全文、识字置信度或单字证据。

等待时显示真实队列位置和前方任务数。准备与检测使用不定进度动画；逐区域处理后按已完成步骤推进，结果保存成功才到 100%，不预测剩余秒数。动画支持系统的减少动态效果设置。进度变绿只表示任务完成。

区域模型的蓝色 `candidate` 表示通过模型分数、前两名差值及区域片段一致性门槛的字体候选；琥珀色 `uncertain` 保留具体原因：

- 模型分数未达门槛，或前两名字体评分过于接近。
- 区域内不同图像片段的字体判断不一致。
- 区域过长、图像质量不足或背景颜色不均匀。
- 区域超出本次处理数量上限。

区域列表和详情展示当前字体候选及模型评分，包括尚未通过确认门槛的区域。`candidate` 是模型候选，评分不是正确概率。`uncertain` 保留原因，API 的 `font.family` 为 null，候选可从 `font.candidates` 读取。模型的未知类别胜出时显示“未知字体”，不输出具名字体或字号；这不等于识别出了未知字体的真实名称。

联合模型使用包含未知类的完整 softmax，不去掉未知类后放大其他分数。没有第二个字体网络或平台路由。页面不改写原始或复制的 JSON。相近字体、混合字体、裁剪偏差和图像退化仍可能导致误判；字体结果不能证明设备系统或截图真伪。

`text_style.font_size_px_estimate` 是输入截图中的像素字号估计，**不是 iOS pt，也不是检测框高度**。区域模型的 `font_size_px_interval` 为 null，不将图块差异伪装成置信区间。`text_color_hex` 是可见 `#RRGGBB` 颜色；缺少可靠结果时字段为 null，`size`、`color` 说明状态和原因。缩放截图会改变像素字号；颜色测量不恢复原始透明度。

队列满时返回 429 和 `Retry-After: 2`；网页保留所选图片。拿到任务 ID 后，查询中断只重试查询，不重复上传。等待任务只保存文件和任务信息，不为每个用户加载模型。

`GET /api/jobs/{id}` 附带 `queue_position`（等待位置从 1 开始，运行时为 0，结束后为 null）、`queue_ahead`、`queue_total` 和 `progress`。`progress.stage_code` 表示阶段，`percent` 可为 null；`current` / `total` 在提供时表示当前阶段的区域计数。进度代表完成步骤比例，不代表耗时比例。

## 图片识别 API

同步识别在 120 秒内完成时直接返回结果，超时返回 202 与任务 ID：

```sh
curl -F "file=@./alipay.png" http://allenflux.tech:9000 -o result.json
curl -F "file=@./alipay.png" http://allenflux.tech:9000/upload
curl -T ./alipay.png http://allenflux.tech:9000/upload/alipay.png
```

`curl -F` 自动附带 multipart 格式和文件名，无需手写 `Content-Type` 或 `X-Filename`。这些上传地址加 `?wait=false` 可先返回任务 ID，原有 `/api/jobs`、`/api/predict` 继续可用。

```sh
# 有令牌时
curl -H 'Authorization: Bearer YOUR_TOKEN' \
  -F "file=@./alipay.png" http://allenflux.tech:9000

# 异步提交、查询、下载
curl -F "file=@./alipay.png" http://allenflux.tech:9000/api/jobs
curl http://allenflux.tech:9000/api/jobs/TASK_ID
curl -o annotated.png http://allenflux.tech:9000/api/jobs/TASK_ID/image
curl -o result.json http://allenflux.tech:9000/api/jobs/TASK_ID/json
```

结果包含模型版本、原图尺寸、区域 `id`、`quad`、`detector_bbox`、`source_bbox`、`crop_url`、`font`、`text_style` 和下载地址。框坐标对应 EXIF 方向校正后的原图，标注 PNG 与其尺寸一致；原上传文件不被标注覆盖。

区域模型返回 `ocr_performed: false`，`text` 为 null 或空字符串，`ocr_confidence` 为 null，`glyphs` 为空。`font.method` 为 `region_neural_network`，`font.scope` 为 `Detected text region`；`candidates` 使用 `{family, score}`，并提供 `score`、`margin` 和 `patch_agreement`。区域图像直接进入字体网络，检测模型仅负责找位置。

## 本地开发与模型版本管理

Python 3.11 开发环境：

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  FLUX_MODEL_DIR=models \
  .venv/bin/uvicorn flux_glyph.api:app --host 127.0.0.1 --port 9000
```

默认加载 `models/`：存在 `ACTIVE.json` 时选择对应完整包，否则加载根目录完整包。清单记录文件大小和 SHA256；进程启动时验证模型，版本写入每个结果。只把 ONNX 放在磁盘上不会启用它。服务端只推理，不训练。

导出当前完整包，也可安装其他经过验证的版本到默认 `models/` 下；历史包保留以便回退：

```sh
.venv/bin/python scripts/model_release.py \
  --model-root models export \
  --output artifacts/flux-glyph-current-bundle.zip
.venv/bin/python scripts/model_release.py install \
  artifacts/flux-glyph-current-bundle.zip --version RELEASE_VERSION
# 同步完整 models/ 及配套代码后重启/重建
docker compose up -d --build
# 回退根目录随项目提供的历史包
.venv/bin/python scripts/model_release.py activate bundled
docker compose restart api
```

模型管理命令在装有项目依赖的本地环境运行。安装目录为 `models/releases/版本号/`，`ACTIVE.json` 记录所选路径，旧模型保留；服务器同步完整目录后重启即可，不需要宿主机 Python 依赖。更改算法或 ONNX 输入输出契约时，应同步更新推理代码并验收。完整格式及历史兼容见 [模型交付说明](docs/model-bundles.md)。独立字体下载 ZIP 是裁图推理工具包，不是这些命令使用的完整网站模型包。

开发测试依赖：`pip install pytest httpx fonttools`。运行 `PYTHONPATH=src python -m pytest -q` 和 `node tests/ui_qa.js`；`node tests/region_browser_qa.js` 使用隔离的本地页面与模拟 API 验证区域 UI，不运行字体精度评测。`tests/browser_qa.js` 和 `tests/integration_api.py` 面向已运行的服务。测试依赖和样本不参与容器推理。

## 历史记录与项目结构

历史 iOS R20 使用三个神经网络组成复核流程，见 [R20 结果](docs/font-sans-results.md)和[原训练说明](docs/font-sans-verification.md)。历史 Android 独立实验版 `android-open-fonts-v1-preview` 的 TEST 命名精度为 92.60%，正确覆盖为 68.47%，原稳定验收失败，见 [原 Android 训练](docs/android-font-training.md)和[逐轮记录](docs/android-font-results.md)。这些数字不是联合模型准确率；历史报告和权重保持原样。

R17 简繁体数据与结果见 [采集证据](docs/ios-traditional-capture.md)、[验证报告](docs/ios-traditional-results.md)。更早版本见 [移动字体改进](docs/mobile-font-improvements.md)、[iOS 单字截图训练](docs/ios-screenshot-training.md)、[神经网络历史](docs/neural-font-training.md)。历史推理兼容代码与保存任务不改变各自原始版本。

- `src/flux_glyph/`：区域检测、字体／字号 CNN、颜色测量、标注导出与 API；另保留历史推理兼容代码。
- `web/`：区域裁图交互、中英切换、模型下载与 API 说明。
- `models/`：默认／历史模型、版本清单和保存的实验权重。
- `assets/`：用于中文标注的字体子集与许可证。
- `scripts/`、`training/`：模型打包、版本管理、数据采集和本地训练工具。
- `tests/`、`docs/`：功能测试、各版本评测和操作文档。

## Git 部署时的模型校验

模型清单验证原始字节，包括换行。`.gitattributes` 的 `models/** -text` 禁止 Git 改写模型文件，部署时应保留属性文件和对应 MANIFEST。

启动报 `Model file checksum mismatch` 时，先同步完整版本；不要依据服务器上的异常文件重新生成清单。`./models` 挂载会覆盖镜像内模型，仅重建镜像不能修复宿主机模型不一致。日志给出实际目录、预期／实际字节数及 SHA256；切换过版本时同时检查 `models/ACTIVE.json`。
