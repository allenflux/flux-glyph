# 无 OCR 的 iOS 区域字体与字号模型

该版本把整段文字区域的图像直接输入 CNN，预测字体名称与像素字号。网络不接收文字识别结果、字符、脚本类型或分字框。区域位置可由检测器提供；离线分类评估使用原生采集程序记录的区域框，因此检测器漏检和框偏移要另做端到端评估。

## 数据和监督证据

数据来自 1,000 张实际 `xcrun simctl io ... screenshot` PNG：800 张训练、100 张校准、100 张固定回归测试。界面是原生 iOS 应用绘制的受控生成场景，不能等同于第三方应用或实体手机的随机截图。

每张图片对应原生 CTFont/CTRun 字体证据。内置字体记录实际解析的 PostScript 名称；应用注册的字体另核文件 SHA、注册字体和实际字体来源。字号真值是实际 CTFont point size 乘屏幕 scale。文字内容仅参与离线原生请求核对、预先固定的内容分组隔离，不进入预处理或网络。

| 分区 | 截图 | 完整区域 | 64×256 图块 |
|---|---:|---:|---:|
| train | 800 | 8,807 | 22,545 |
| calibration | 100 | 1,112 | 2,870 |
| test | 100 | 1,097 | 2,773 |

11,016 个区域全部通过原生证据和图像预处理，未发生区域拒绝。类别按已采集的训练数据取八类：HarmonyOS Sans SC、MiSans、Noto Sans CJK SC、OPPO Sans、PingFang SC、SF Pro、Helvetica、Alipay Number。所有类别都实际通过 iOS 原生渲染；这不代表它们都是系统自带字体。

原图文件、解码后像素、source/page/content group、区域像素和规范化区域文本的跨分区交集为零。字符允许跨分区重复。测试页已用于此前单字 CNN 的评估，本轮属于复用固定回归集，不是新的盲测。

完整数据清单位于本地 `artifacts/ios-region-font-v1/data/MANIFEST.json`；SHA-256：`ed7979b1950ff7098d01208451a50ac5abe20c0d4e143d7847f35cdc5c133c2a`。源截图和字体文件不随源码提交。

## 训练与运行计算

训练和运行共享 `flux_glyph.region_font.preprocess_region`：从原始 RGB 图像估计背景与墨迹包围框，保留纵横比，将墨迹高度归一化为 56，补白至高 64。宽图像均匀提取最多八个宽 256 的窗口；墨迹为 1、背景为 0。无法覆盖完整宽度或背景、对比度不满足要求时拒绝。

模型沿用已训练单字 CNN 的卷积主干和 128 维特征层，用 adaptive pooling 兼容 64×256 输入。八类分类头按字体名称从父模型映射；新增字号头预测 `log(actual_font_size_screen_px / ink_height_px)`。这是从真实行图像继续训练，字号头也实际更新。输入没有 script mask。

每步按字体均衡抽取 64 个区域，再从每个区域随机选取一个窗口；每一类的区域队列遍历完才重新洗牌。损失是八类交叉熵加 0.5 倍 log 比例的 smooth-L1。初始化来源、种子、采样规则、3,000 步预算和选择准则在读取校准预测前写入 `TRAINING_FREEZE.json`。

完整区域的字体概率为各窗口 softmax 概率的均值。字号为 `ink_height_px × exp(median(log_em_ratio))`；字号窗口差异为 `(p90(exp(log_ratio)) − p10(exp(log_ratio))) / exp(median(log_ratio))`。任何窗口的 log 比例绝对值超过 3 时拒绝该区域；不通过字体门槛的区域不输出字号。

每 250 步用完整校准集选 checkpoint：字体宏平均准确率减去 0.1 倍截断到 1 的平均相对字号误差。随后仅在校准集上选择温度和字体分数/间隔门槛，要求至少 20 个接受区域且经验精度不低于 97%。窗口字体一致率固定为 0.7；字号差异门槛固定为 0.2。此精度目标不构成未知字体或实体手机的保证。

所有模型、温度和门槛选择写入 `SELECTION_FREEZE.json` 后，才打开 test 图块计算最终固定回归结果。`report.json` 同时记录全体区域字体准确率、接受精度/覆盖、字号误差、源截图数量及权重变化。

## 复现

离线预处理使用具备运行时图像依赖的环境；训练环境需要 `requirements-train.txt`。本地已保存的数据和源图片都有 SHA 检查。

```bash
python training/train_regions.py --captures artifacts/mobile-font-capture/ios-capture-v1/labels.jsonl --data artifacts/ios-region-font-v1/data --prepare-only
python training/train_regions.py --data artifacts/ios-region-font-v1/data --validate-only
python training/train_regions.py --data artifacts/ios-region-font-v1/data --output artifacts/ios-region-font-v1 --device mps --steps 3000 --batch-size 64 --eval-every 250
```

输出目录必须为新目录，或仅包含本次准备的 `data` 与可丢弃速度探测 `benchmark`。`--benchmark-steps 20 --device mps` 只访问 train 图块，不保存模型、不读取校准或测试图块。

本机 batch 64 速度探测：CPU 约 0.547 秒/步，MPS 约 0.0804 秒/步。3,000 步的 MPS 纯更新预计约四分钟，实际总时间还包括校准、SHA 检查和导出。

## 本次实际结果

本次已执行全部 3,000 次梯度更新，训练及校准约 251 秒，校准规则选中第 3,000 步。全部 8,807 个训练区域都被采样，每个区域至少 13 次。卷积主干、分类头和字号头参数均发生变化。

| 指标 | 校准集 1,112 区域 | 复用测试集 1,097 区域 |
|---|---:|---:|
| 字体 top-1 准确率，不经过门槛 | 93.88% | 93.89% |
| 接受结果精度 | 98.23%（888/904） | 97.92%（895/914） |
| 接受覆盖率 | 81.29% | 83.32% |
| 全体区域字号 MAE | 1.001 px | 1.039 px |
| 全体区域字号平均相对误差 | 1.74% | 1.81% |

校准冻结温度 0.5、最低分数 0.5、最低间隔 0.01、窗口一致率 0.7、字号窗口差异上限 0.2。测试接受结果中有 19 个错误字体、183 个区域未被接受。以上使用原生区域框，不能直接作为整张截图检测加分类的准确率。测试集已被前版使用，本次仍明确作为固定回归集报告。

训练首次 ONNX 导出 SHA-256 为 `da9866834921b43886d86d662d3f95056b3affe119c31b6c7abc2190554672de`，它未通过预先规定的分类数值容差，保留为历史产物。训练审计结果见 [ios-region-font-training-results.json](ios-region-font-training-results.json)。以上原生框表格反映冻结 PyTorch 模型，端到端检测结果另行评估。

## 交付前的稳定 ONNX 导出

实际校准图块定位出首次导出在 GroupNorm 的长维度 float32 归约上累积误差；关闭 ORT 优化或普通 float32 中心方差展开均未消除。`training/export_region_stable.py` 将四个 GroupNorm 展开为分组、减均值、中心平方方差、除标准差与仿射变换，并仅在该归约中使用 float64，之后恢复 float32。CPU ONNX Runtime 支持这些中间 double 运算，网络输入、分类输出和字号输出仍是 float32。冻结权重、训练步骤、温度和门槛全部不变。

稳定导出使用原始 PyTorch GroupNorm 的相同 64 个 calibration 区域、187 个窗口作为比较对象，容差未改。batch 1/7/32/128 全部通过分类/字号两个输出、区域聚合、类别与接收决定检查。分类数值最大差降为 3.624e-5，log 字号比例最大差 2.086e-7。原始失败导出与报告、float32 展开失败对照均保留。细节见 [region-onnx-parity.md](region-onnx-parity.md)。

交付必须使用稳定导出，SHA-256 为 `9fde0988dd6e0fe52fb8cacbdd96d1456eabf01ce3a4b431de9ebfd982f3d304`。模型与 metadata 在 `models/experiments/ios-region-v1/`；完整部署 bundle 为 `artifacts/ios-region-font-v1/bundle-stable/`。`artifacts/ios-region-font-v1/region/` 和原 `bundle/` 只保留首次导出的历史证据。

重新导出与复核时使用新的输出目录，保留训练参考：

```bash
python training/export_region_stable.py \
  --run artifacts/ios-region-font-v1 \
  --output artifacts/ios-region-font-v1/export-diagnostics/stable-double \
  --float64-reduction
python training/region_parity.py reference \
  --run artifacts/ios-region-font-v1 --data artifacts/ios-region-font-v1/data \
  --output artifacts/ios-region-font-v1/parity-reference
python training/stage_region_parity_export.py \
  --reference artifacts/ios-region-font-v1/parity-reference \
  --export artifacts/ios-region-font-v1/export-diagnostics/stable-double \
  --output artifacts/ios-region-font-v1/export-diagnostics/double-parity-reference
python training/region_parity.py check \
  --reference artifacts/ios-region-font-v1/export-diagnostics/double-parity-reference \
  --output artifacts/ios-region-font-v1/export-diagnostics/double-parity.json
```

前两步使用具备 PyTorch/ONNX 的训练环境，最后一步使用 ONNX Runtime 环境。中间绑定步骤逐项核对训练 metadata、门槛和 checkpoint SHA 不变，并复制完全相同的 PyTorch 输出，不重新生成更有利于导出模型的参考。

模型输入为 `tiles: float32[N,1,64,256]`，输出为 `logits: float32[N,8]` 与 `log_em_ratio: float32[N]`。`metadata.json` 携带字体顺序、模型 SHA、温度、门槛、训练和稳定导出证据。字号输出单位是截图像素里的字体 em 大小，不能直接当成未知设备的 point/sp。
