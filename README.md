# Flux Glyph

上传截图，定位文字区域，直接识别区域字体并估计字号、文字颜色。页面用 `R001` 等编号和原图裁图定位区域，提供字体详情、格式化 JSON、复制 JSON 和原尺寸标注 PNG 下载，支持中文与 English。

网站和 API 使用 **9000** 端口。保留 PP 的文字区域检测模型用于定位；字体 CNN 直接处理区域图像，不读取文字内容，不需要字符切分或人工选择中文／英文。独立字体模型也可以下载到本地使用。

## R17 简繁体、英文与数字字体模型

当前流程是：**原图 → 文字区域检测 → 区域字体 CNN 与字号回归 → 原图颜色测量**。字体判断使用训练后的网络权重。候选包含模型分数；`uncertain` 表示当前结果未通过门槛。

本地 Docker 已切换到 `r17-ios-hant-region-v1`，页面为 `http://localhost:9000/`，模型下载位于上传区上方。以 `/api/health` 的 `model_version`、结果中的 `font_method: region_neural_network` 及所选模型包为准。本次本地验收不代表线上服务器已更新。

R17 使用新采集的 1,000 张原生 iOS Simulator 截图继续训练，包含简体、繁体、英文和数字。八个字体输出中，PingFang 表示苹方字体族，覆盖原生 SC／TC／HK，不宣称能从相同字形区分地区版本。SF Pro、PingFang 和 Helvetica 来自系统；Alipay Number 及其余类别由采集应用加载，**Alipay Number 不是 iOS 系统字体**。来源写入元数据和下载说明。

新固定测试的 100 张完整截图、1,104 个区域中，1,013 个字体名正确、1 个错误、90 个待确认；没有漏检或额外框。已输出名称的准确率为 99.90%，返回名称的覆盖率为 91.85%。这些是受控模拟器成绩，不能推广到任意 App、真机或未知字体。详见 [R17 验证报告](docs/ios-traditional-results.md)、[采集证据](docs/ios-traditional-capture.md) 和 [训练流程](docs/ios-traditional-training.md)。

默认模型由随源码提供的 `models/ACTIVE.json` 选择，所需完整包在 `models/releases/`，因此标准 Docker Compose 部署也能提供字体模型下载。根目录历史 R14 文件保留用于回退。

R17 完整包已保留在 `models/releases/r17-ios-hant-region-v1`，字体权重和元数据另存于 `models/experiments/ios-hant-region-v1`。需要重建完整包时：

```sh
PYTHONPATH=src .venv/bin/python scripts/package_region.py \
  --region models/experiments/ios-hant-region-v1 \
  --output artifacts/rebuilt-r17-bundle \
  --version r17-ios-hant-region-v1
```

输出目录必须尚不存在；打包工具会验证字体模型，并从当前基础包复制 PP 检测模型和检测配置。区域模型完整包不需要 PP 识字模型、字符字典、逐字参考库或旧字号统计文件。

## 下载字体模型并单独使用

网页顶部的“字体模型”卡片显示当前可下载模型的版本、文件大小和下载按钮；“本地使用方式”包含字体来源及运行命令。下载包包含字体 ONNX、`metadata.json`、`inference.py`、`text_style.py`、`predict.py`、`requirements.txt`、README 和 SHA256 清单，不包含用户上传图片。服务更新后可点“刷新状态”重新获取下载信息。

解压 ZIP 后，在其目录安装依赖并运行：

```sh
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python predict.py text-region.png
```

输入应是一行或一个文字区域的裁图。脚本返回字体候选、像素字号估计及可见颜色，不要求文字内容；完整截图可交给网站/API 自动定位区域。Python 3.10+，推理依赖 ONNX Runtime、NumPy、Pillow，不需要 PyTorch。

模型输入 `tiles` 为 float32 `[N,1,64,256]`；两个输出是 `logits [N,C]` 和 `log_em_ratio [N]`。必须使用包内 `preprocess_region` 保留比例、归一化背景并生成图块，再按元数据里的类别顺序、温度和门槛聚合；不要把整张截图直接拉伸成模型输入。`predict.py` 已完成这些步骤。

```sh
curl http://localhost:9000/api/models/font
curl -o flux-glyph-font-onnx.zip http://localhost:9000/api/models/font/download
```

信息接口包含 `available`、`version`、`families`、`input_shape`、`download_url`、`bytes`、`sha256`、`ocr_required: false` 和 `usage`。未加载区域模型时返回 `available: false`，下载接口返回 404。启用令牌时，两路接口与图片识别使用同一访问保护；网页下载使用解锁后的 Cookie，命令行添加 `Authorization: Bearer YOUR_TOKEN`，不把令牌写入 URL。

## Docker Compose 部署

将整个项目及经过验证、已选定的完整模型包复制到服务器。`compose.yaml` 默认只读挂载 `./models`，由随源码提供的 `ACTIVE.json` 选择已验证的完整包；无需复制本地训练的 `artifacts/` 目录。

```sh
cp .env.example .env
# 可编辑 .env 设置 FLUX_API_TOKEN；留空时无需令牌。
docker compose up -d --build
docker compose logs -f api
```

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

未通过门槛时 `font.family` 为 null，存在的候选分数仍可查看。分数是已知类别间的模型评分，不是实际准确率；未知字体也可能误判。字体名不能用于确定手机系统、设备或截图真伪。绿色 `supported` 及灰色 `out_of_scope` 保留用于历史结果兼容，区域模型不沿用旧苹方参考匹配规则。

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
  --output artifacts/flux-glyph-r17-ios-hant-region-v1.zip
.venv/bin/python scripts/model_release.py install \
  artifacts/flux-glyph-r17-ios-hant-region-v1.zip --version r17-ios-hant-region-v1
# 同步完整 models/ 及配套代码后重启/重建
docker compose up -d --build
# 回退根目录随项目提供的历史包
.venv/bin/python scripts/model_release.py activate bundled
docker compose restart api
```

模型管理命令在装有项目依赖的本地环境运行。安装目录为 `models/releases/版本号/`，`ACTIVE.json` 记录所选路径，旧模型保留；服务器同步完整目录后重启即可，不需要宿主机 Python 依赖。更改算法或 ONNX 输入输出契约时，应同步更新推理代码并验收。完整格式及历史兼容见 [模型交付说明](docs/model-bundles.md)。独立字体下载 ZIP 是裁图推理工具包，不是这些命令使用的完整网站模型包。

开发测试依赖：`pip install pytest httpx fonttools`。运行 `PYTHONPATH=src python -m pytest -q` 和 `node tests/ui_qa.js`；`node tests/region_browser_qa.js` 使用隔离的本地页面与模拟 API 验证区域 UI，不运行字体精度评测。`tests/browser_qa.js` 和 `tests/integration_api.py` 面向已运行的服务。测试依赖和样本不参与容器推理。

## 历史记录与项目结构

R14 参考匹配记录见 [移动字体改进](docs/mobile-font-improvements.md)、[中文精度报告](docs/accuracy-report.md)；R15 单字 CNN 记录见 [iOS 截图训练](docs/ios-screenshot-training.md)、[历史截图评测](docs/ios-screenshot-results.md)。更早的实验见 [神经网络训练历史](docs/neural-font-training.md)，原部署资源记录见 [交付验证](docs/delivery-validation.md)。这些记录保留各自版本、输入和评分口径，不是 R16 验证结果。

- `src/flux_glyph/`：区域检测、字体／字号 CNN、颜色测量、标注导出与 API；另保留历史推理兼容代码。
- `web/`：区域裁图交互、中英切换、模型下载与 API 说明。
- `models/`：默认／历史模型、版本清单和保存的实验权重。
- `assets/`：用于中文标注的字体子集与许可证。
- `scripts/`、`training/`：模型打包、版本管理、数据采集和本地训练工具。
- `tests/`、`docs/`：功能测试、各版本评测和操作文档。

## Git 部署时的模型校验

模型清单验证原始字节，包括换行。`.gitattributes` 的 `models/** -text` 禁止 Git 改写模型文件，部署时应保留属性文件和对应 MANIFEST。

启动报 `Model file checksum mismatch` 时，先同步完整版本；不要依据服务器上的异常文件重新生成清单。`./models` 挂载会覆盖镜像内模型，仅重建镜像不能修复宿主机模型不一致。日志给出实际目录、预期／实际字节数及 SHA256；切换过版本时同时检查 `models/ACTIVE.json`。
