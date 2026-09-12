# iOS 截图字体训练

这版使用实际运行在 iOS Simulator 中的原生 App 截图训练字体 CNN，并在识别 API／网页中返回估计字号与文字颜色。截图中的文字和页面由采集 App 生成；它们不是独立第三方 App 或真实手机采集，不能把该测试结果写成所有 iPhone 截图的准确率。

## 数据与真实标签

采集设备为 iPhone 17 Pro、iOS 26.5、1206×2622 像素、3×屏幕缩放。`capture_ios.py` 通过 `xcrun simctl io ... screenshot` 保存原 PNG；UIKit/CoreText 原生 App 输出逐行及逐字实际 CTRun 字体、字形 ID、原图坐标、有效字号和实际颜色。外部字体同时验证源文件 SHA、注册路径和实际使用的 PostScript 字体名；发现替代字体或裁切会拒绝。

共 1,000 张截图、11,016 行：按预先声明的页面／内容分组分为训练 800、校准 100、测试 100 张。来源图像、页面、完整文字组合和区域像素均检查跨集合重叠；不同组合中允许出现相同字符。字体、文字、字号、颜色分别分配，覆盖浅／深背景、13 种文字颜色及 13–27 pt。

8 个字体家族、23 种样式：PingFang SC、SF Pro、Helvetica、HarmonyOS Sans SC、MiSans、Noto Sans CJK SC、OPPO Sans、Alipay Number。后五类在 iOS 采集 App 中作为自定义字体加载；这不代表采集了安卓截图。Alipay Number 只有数字样本。

预处理从原截图提取 119,447 个字形：训练 95,559、校准 12,056、测试 11,832。4,000 个空白／标点等非分类字符跳过，10 个字形因预处理质量不足拒绝。字体标签取自原生渲染记录，未用旧分类器的预测作为标签。

## 训练与推理

网络输入为原图单字经相同预处理后的 64×64 灰度笔画，卷积层学习字形特征，分类头输出字体家族。以前一版合成实验的网络初始化，并按实际采集到的 8 个字体家族选取分类头；本轮优化只使用原生截图像素，无教师预测损失。

训练预算为 20 轮×250 次 AdamW 更新、batch 128，中文／Latin 与家族均衡采样。中文和数字／英文分别计算分类交叉熵。只用校准集选权重，温度按完整文字行的负对数似然选择；输出门槛要求至少 20 个完整校准区域、已输出字体名准确率达到 97%。这是校准目标，测试结果另行报告，不能当作真实准确率保证。

同一行内同字先平均，再对不同字等权平均。测试在权重、温度、门槛冻结后运行。另用整个测试 PNG 运行真正的 PP 检测→OCR→分字→CNN，原生框只用于事后评分，不送入预测。颜色从原始 RGB 笔画核心估计；字号用原图笔画高度及只在 train 建立的尺度统计估计，单位为截图 px，见 [字号与颜色](text-style-estimation.md)。

## 本地复现

采集 App 工程及原生协议见 [iOS 采集说明](../training/capture/ios/README.md)。先在模拟器中构建、安装 App，并准备带源 SHA 的字体清单，再执行：

```sh
PYTHONPATH=src .venv-dev/bin/python training/capture/capture_ios.py \
  --simulator SIMULATOR_UDID \
  --scenes artifacts/mobile-font-capture/ios-scenes-colors-v2.json \
  --assets artifacts/mobile-font-capture/assets \
  --output artifacts/mobile-font-capture/ios-capture-v1
PYTHONPATH=src .venv-dev/bin/python training/capture/prepare_captured.py \
  --input artifacts/mobile-font-capture/ios-capture-v1/labels.jsonl \
  --output artifacts/mobile-font-capture/ios-prepared-v1
.venv-train/bin/python training/train_screenshots.py \
  --data artifacts/mobile-font-capture/ios-prepared-v1 \
  --warm-start artifacts/neural-font-v1/neural/model.pth \
  --output artifacts/ios-font-screenshots-v1 \
  --device mps --epochs 20 --steps-per-epoch 250 --batch-size 128
.venv-dev/bin/python training/build_size_metrics.py \
  --data artifacts/mobile-font-capture/ios-prepared-v1 \
  --output artifacts/mobile-font-capture/ios-size-metrics-v1.json
.venv-dev/bin/python scripts/package_neural.py \
  --neural artifacts/ios-font-screenshots-v1/neural \
  --size-metrics artifacts/mobile-font-capture/ios-size-metrics-v1.json \
  --output artifacts/ios-font-screenshots-v1/bundle \
  --version r15-ios-screenshots-v1
```

训练环境安装 `requirements-train.txt`，无 MPS 的机器可以选 `--device cpu`。原图、源字体、warm-start checkpoint 和训练产物保存在本地 artifacts 中，不随普通源码包传输；缺少这些文件会明确失败，不会替换为桌面渲染或伪标签。采集与训练输出要求新目录，原始固定实验不覆盖。

`TRAINING_FREEZE.json`、`SELECTION_FREEZE.json`、`training-history.json` 和 `report.json` 记录样本来源、分区、代码及权重 SHA、实际更新次数与选中轮次；线上只需导出的 ONNX 和 JSON，不加载 PyTorch checkpoint。`scripts/check_neural_parity.py` 使用固定 128 个校准字形校验 PyTorch 与实际 CPU ONNX 的输入、分数和门槛判断一致。

完整测试与边界见 [iOS 验证结果](ios-screenshot-results.md)。
