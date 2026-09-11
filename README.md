# Flux Glyph

网页保留 `Flux Glyph` 项目名，以浅灰色弱化 `Flux`，使用红底 `fg` 小图标，无作者署名；右上角提供 `中文 / EN` 切换按钮。

上传支付宝截图，自动框出文字区域，点击查看中文字体结果与原图裁字。结果旁直接显示格式化 JSON，支持一键复制；可下载在原图尺寸上画框、标注字体的 PNG。界面和 API 说明支持中文、English 切换。

网站和 API 使用 **9000** 端口。项目自带 PP-OCR 模型和紧凑字体参考包；运行时不依赖旧仓库、移动硬盘、macOS Vision 或在线模型下载。

## Docker Compose 部署

将整个项目（包括 `models/`）复制到服务器：

```sh
cp .env.example .env
# 可编辑 .env 设置 FLUX_API_TOKEN；留空时无需令牌。
docker compose up -d --build
docker compose logs -f api
```

公开访问地址为 `http://allenflux.tech:9000/`，服务与上传均使用 9000 端口。健康检查为 `/api/health`，API 文档为 `/docs`。设置令牌后，网页可输入相同令牌解锁；程序请求使用 `Authorization: Bearer <令牌>`。

默认面向 **2 核 / 2 GB** 主机：一个推理任务运行、最多八个等待；单线程 ONNX 会话与 BLAS；字体缓存仅保留 32 个汉字；容器最多 2 核、1536 MiB 内存，为主机预留空间。限制单图 8 MB / 1200 万像素；最多识别 200 个文字框，超出框仍可见并标明未处理。

上传数据保存在 `glyph-data` 命名卷中，完成任务按 24 小时或最多 100 个任务清理；启动、任务完成及每分钟检查一次，正在处理的任务保留到完成。`docker compose down` 不会删除数据卷；不要使用 `down -v`，除非确实要清空结果。模型以只读目录挂载。

## 排队与页面交互

等待时显示真实队列位置和前方任务数，处理时显示阶段和已完成区域数。准备与检测使用不定进度动画；逐区域处理后按已完成步骤推进，结果保存成功后才到 100%，不预测剩余秒数。动画支持系统的减少动态效果设置。

队列满时返回 429 和 `Retry-After: 2`；网页保留所选图片以便重试。拿到任务 ID 后，查询中断不会自动重复上传图片。等待任务只存文件和任务信息，不为每个用户加载一套模型。

浏览器负责文字框绘制、界面缩放、中英切换、JSON 格式化和复制；服务端负责原图检测、识字、字体匹配和标注 PNG。原图不会为了页面展示而压缩后再送入模型。网页不再显示 JSON 下载按钮，原 API 下载接口继续兼容。

`GET /api/jobs/{id}` 附带 `queue_position`（等待位置从 1 开始，运行时为 0，结束后为 null）、`queue_ahead`、`queue_total` 和 `progress`。`progress.stage_code` 是处理阶段，`percent` 可为 null，`current` / `total` 是当前阶段的区域计数。进度代表完成步骤比例，不代表耗时比例。

## API

同步识别（120 秒内完成则直接返回结果，超时返回 202 与任务 ID）：

```sh
curl -F "file=@./alipay.png" http://allenflux.tech:9000 -o result.json
```

无需手写 `Content-Type` 或 `X-Filename`，`curl -F` 会自动带上 multipart 格式与文件名。也支持：

```sh
curl -F "file=@./alipay.png" http://allenflux.tech:9000/upload
curl -T ./alipay.png http://allenflux.tech:9000/upload/alipay.png
```

这些上传地址默认在完成后返回 JSON；加 `?wait=false` 则先返回任务 ID。原有 `/api/jobs`、`/api/predict` 接口继续可用。

有令牌时：

```sh
curl -H 'Authorization: Bearer YOUR_TOKEN' \
  -F "file=@./alipay.png" http://allenflux.tech:9000
```

异步任务与网页相同：

```sh
curl -F "file=@./alipay.png" http://allenflux.tech:9000/api/jobs
curl http://allenflux.tech:9000/api/jobs/TASK_ID
curl -o annotated.png http://allenflux.tech:9000/api/jobs/TASK_ID/image
curl -o result.json http://allenflux.tech:9000/api/jobs/TASK_ID/json
```

结果包含 `model_version`、`summary`、原图尺寸、每框的 `quad`、文字、字体标签、候选距离、逐字原图裁图，以及下载地址。框坐标对应 EXIF 方向校正后的原图；导出的标注 PNG 与其像素尺寸一致。原始上传文件保留，不会被标注覆盖。

结果状态：

- `supported`：中文字形通过当前苹方规则，逐字候选一致。
- `candidate`：其他字体在三种图像检查中排名一致，仍为字体候选。
- `uncertain`：分字不足、缺参考字、字体不一致或证据不足。
- `out_of_scope`：纯数字、英文等当前中文字体模型范围外。

当前字体包覆盖 11 个字体族、31 个字体样式、986 个汉字。混合文字框的字体结论仅指中文部分，不能推断数字或英文使用相同字体。字体结果不代表真实设备或截图真伪。

**90% 是验收目标，不是已经得到真实支付宝截图验证的准确率。** 验证须同时报告输出字体名称后的准确率、输出覆盖率、待确认数量。受控渲染图片的真字体实验不能替代真实截图的逐框字体标注。

本次固定 36 张有字体真值的受控测试：28 张输出字体名，27 张名称正确（96.43%），覆盖率 77.78%，全部样本名称正确率 75.00%；若同时要求裁字证据完整，已输出结果的严格正确率为 92.86%。数字分母不同，不能把 96.43% 写成所有支付宝截图的识别率。详见 [精度报告](docs/accuracy-report.md)。

实际原图测试和资源记录见 [交付验证](docs/delivery-validation.md)。100 张原图是无字体真值的功能、覆盖与资源测试，白图和蓝图使用同一条推理流程。

## 本地开发与模型迭代

Python 3.11：

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/uvicorn flux_glyph.api:app --host 127.0.0.1 --port 9000
```

`models/MANIFEST.json` 为模型文件清单及 SHA256。推理进程启动时验证模型；模型版本写入每个结果。当前字体方法是固定字形特征匹配，不是新训练的 Yuzu 字体神经网络；PP 检测和识字使用 ONNX 神经网络。字体参考包先在本地预处理，服务端按需读取，避免重复渲染和存储浮点大数组。

本地训练/生成新模型后，按相同契约打成完整模型包。当前格式、导入与回退见 [模型交付说明](docs/model-bundles.md)。服务端仅执行推理，不执行训练。

```sh
# 导出当前自包含模型包
.venv/bin/python scripts/model_release.py export --output artifacts/flux-glyph-model-v1.zip
# 导入本地新版本；会校验 SHA 和运行时格式
.venv/bin/python scripts/model_release.py install new-model.zip --version experiment-002
# 重启加载已选版本
docker compose restart api
# 回退最初随项目提供的模型
.venv/bin/python scripts/model_release.py activate bundled
docker compose restart api
```

建议在本地验收新版本后再同步到服务器。新版本放入 `models/releases/版本号/`，`ACTIVE.json` 仅切换指向；旧模型文件保留，便于回退。修改代码或依赖后需要重新 `docker compose up -d --build`。

模型管理命令在装有本项目依赖的本地开发环境中运行。将完整 `models/` 目录同步到服务器后重启容器即可；服务器无需安装宿主机 Python 依赖。导入检查包括全部参考数组、字体表、门槛及 PP ONNX 的完整性与运行时格式。训练算法或输出契约发生变化时，需要同时更新相应推理代码。

开发验证依赖为 `pip install pytest httpx fonttools`。运行 `PYTHONPATH=src python -m pytest -q`；固定精度复测为 `python tests/accuracy_eval.py --evaluate`。复测直接读取项目内已冻结的 36 张图片；只有重新生成输入或离线迁移旧字库才需要本地源字体/旧实验目录。实际 HTTP 检查为 `python tests/integration_api.py`。这些依赖和测试样本不参与容器推理。

## 项目结构

- `src/flux_glyph/`：跨平台检测、识字、原像素分字、字体匹配、标注导出、API。
- `web/`：参考 FluxDrop 风格的单页网站。
- `models/`：PP ONNX、字体参考包、版本清单。
- `assets/`：用于画中文标注的开源字体子集与许可证。
- `scripts/`：离线模型打包、导入和回退工具。
- `tests/`、`docs/`：功能测试、精度和资源验证记录。
