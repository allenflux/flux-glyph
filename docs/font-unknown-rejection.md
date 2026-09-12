# 未覆盖字体的神经网络拒识

原区域模型在八个已知类别中给出相对分数。训练范围之外的楷体、手写体或其他字体，也可能被分到 PingFang、SF Pro 等类别，并得到接近 1 的分数。提高原分类分数门槛不能解决这个问题。

本轮增加独立的二分类神经网络，输入仍是区域原图生成的 `float32 [N,1,64,256]` 图块，输出 `known_logits [N,2]`，顺序为 `unknown, known`。它从 R17 图像编码器初始化一份独立网络，端到端训练编码器及 `128 → 64 → 2` 判断头；图像处理参数封装进 ONNX。R17 字体名称和字号模型的权重文件保持不变。推理不输入文字内容、脚本类型、手机品牌或系统标签，不做参考字形匹配。

通过覆盖检查后，继续执行原字体分类和字号回归；拒识时返回 `status: out_of_scope`、`reason_code: unknown_font_rejected`，字体名称、命名候选、分数和字号均为空，原图颜色仍可测量。页面显示“未知字体”。覆盖检查失败或输出非法时停止命名，不能绕过拒识模型。普通已知类别分数与覆盖检查分数都不等于实测准确率。

## 数据与冻结规则

采集工具在独立 iOS Simulator 中实际运行原生应用，再用 `simctl screenshot` 获取图像。内容是受控生成页面，不是第三方应用或真实安卓设备截图。每行以 CoreText 的实际字体、字体运行区间、文件来源、SHA 和字形覆盖验证标签。Courier 和 Seravek 的资产仅重命名字体注册标识以避免系统同名冲突，原字体各 face 的字形、字符映射、度量表逐字节保持不变，原文件及派生文件 SHA 均记录在来源清单。

360 页包含 4,320 行。每页同时出现已知和未知字体；两类的文字类型、字号、颜色及背景配置匹配。

| 分区 | 页数 | 已知区域 | 未知区域 | 未知字体族 |
|---|---:|---:|---:|---|
| train | 240 | 1,440 | 1,440 | Kaiti SC、Songti SC、HanziPen SC、Times、Courier |
| calibration | 60 | 360 | 360 | Baoli SC、Weibei SC、Lucida Grande、Comic Sans MS |
| test | 60 | 360 | 360 | Wawati SC、Libian SC、Seravek、Chalkboard |

另采集 60 张纯训练截图、720 个区域，新增 Apple Chancery、Bradley Hand、Noteworthy 各 120 个未知区域，与 360 个已知区域成对。补充数据不含校准或测试字体；原 360 页及其清单保持不变。英文手写数字统一添加前置 0，避免部分字体首字负侧边距使笔画被采集框裁切；两类成对采用相同文本，修正后重新验证全部来源。

另外保留 R17 的 8,744 个训练区域与 1,093 个校准区域作为已知字体样本；其 1,104 个测试区域只作最终回归。新数据与原 R17 场景的文字内容互斥，源图与区域像素不跨分区。文字标注仅验证采集及隔离，不进入网络。未知字体族在训练、校准、测试中互斥。用户给出的页面缩略图只作冻结后的定性检查，不用于训练、校准或准确率统计。

旧 R17 prepared manifest 保持原 SHA。新增 `preprocessing_snapshot` 验证历史源码快照的完整 SHA，并检查预处理函数、相关常量和依赖 AST 与当前代码相同；不能通过改写原清单来放行不同的预处理。

训练前冻结策略：独立网络本地训练 4,000 步，每 500 步只用 calibration 评估。门槛要求每个已知数据来源误拒不超过 1%，每个已知字体族不超过 2%，系统字体合计不超过 1%；在这些约束内选择未知字体召回率最高的检查点，最低要求 80%。在原选中网络上，用额外训练样本继续 2,000 步，每 500 步只按相同 calibration 策略选择；两轮候选也按同一校准目标比较，不用 test 决定版本。分数采用图块二分类 softmax 的均值，温度固定为 1，门槛保留 `2e-5` 数值余量。冻结检查点、门槛和逐行 CAL 判定后，才运行 test；test 要求未知字体召回至少 80%，每个已知数据来源与系统字体误拒各不超过 2%。

ONNX 导出保留网络参数，GroupNorm 使用中心化方差和 float64 中间归约。原 PyTorch GroupNorm 是独立数值参考，同时检查不同 batch、两个校准来源及训练时冻结的逐行判定。测试不参与导出选择或门槛调整。

## 本地运行

生成与采集脚本：`training/capture/generate_unknown_scenes.py`、`training/capture/capture_ios.py`；准备及来源复核：`training/capture/prepare_unknown_regions.py`。完整本机来源、场景和截图清单保存在 `artifacts/font-unknown-gate-v1/`。训练命令示例，`$TORCH_PYTHON` 指向本地有 PyTorch 的 Python：

```sh
"$TORCH_PYTHON" training/train_rejection_e2e.py \
  --data artifacts/font-unknown-gate-v1/data-v1 \
  --anchor artifacts/ios-hant-region-v1/data \
  --checkpoint artifacts/ios-hant-region-v1/region/model.pth \
  --snapshot artifacts/font-unknown-gate-v1/provenance/region_font_r17.py \
  --output artifacts/font-unknown-gate-v1/run-e2e-v1 --device mps
```

新增手写数据续训入口为 `training/train_rejection_augmented.py`，增加 `--supplement` 与 `--resume-rejection`，并保留补充清单及父拒识检查点 SHA。`training/export_rejection.py` 导出并验证，`training/evaluate_rejection.py` 在模型和策略冻结后评估。每次输出必须使用新目录，失败结果保留；不覆盖测试报告。

## 推理与下载兼容性

新元数据算法为 `region-cnn64x256-rejection-v2`，显式绑定两个 ONNX 的 SHA、已知类别顺序和拒识门槛。旧推理实现不能把它当作不带拒识的 v1 模型运行；新版实现仍兼容原 R17 v1。

下载 ZIP 同时包含字体/字号 ONNX、拒识 ONNX、元数据、独立推理脚本、颜色估计、使用说明、依赖清单与 SHA256。只需 ONNX Runtime、NumPy、Pillow；`predict.py` 自动执行覆盖检查，仍然不需要 PyTorch 或 OCR。服务器通过 Git 获取本地验证后的代码与模型，不在服务器训练。
