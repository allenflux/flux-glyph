# 简繁体、英文和数字区域模型的训练流程

本轮使用第二批原生 iOS Simulator 截图，在 R16 区域模型权重上继续训练字体分类与像素字号回归。采集方式、实际字体证明及类别分组原因见 [采集记录](ios-traditional-capture.md)。运行时仍只向字体模型输入区域图像；文字检测负责找位置，不进行 OCR 识字。

## 数据与标签

固定场景包含 1,000 页、10,941 个区域：800 页训练、100 页校准、100 页测试。每页包含两个英文和两个数字区域，其余为简体或繁体。新旧文字组合以及本批三个分区的文字组合按 NFKC、casefold 和去空白后检查隔离。原始 PNG、解码像素、区域像素、场景及原生字体证明也参与校验。

八个输出类别为 HarmonyOS Sans SC、MiSans、Noto Sans CJK SC、OPPO Sans、PingFang、SF Pro、Helvetica、Alipay Number。PingFang 合并原生 SC／TC／HK 字体族，原生名称继续保留用于审计；该分组在采集和训练前确定。不同地区版本存在相同像素，不能把同一图像强制分成三个互斥类别。

`font_sources` 记录实际加载来源：PingFang、SF Pro、Helvetica 为系统内置；其余为采集应用加载的字体资产。**Alipay Number 是应用自带数字字体，不是 iOS 系统字体。** 字体来源用于说明和验证，不作为网络输入或按系统推断字体的规则。

## 冻结的训练方式

- 从 R16 的 PyTorch 区域检查点初始化，保留字号回归头；新 PingFang 类的起始权重来自旧 SC 类，随后由本批 SC／TC／HK 原生截图共同训练。
- 4,000 次 AdamW 参数更新，batch 64，初始学习率 0.0001，seed `2026091293`。
- 每 250 步只用 calibration 选择检查点；温度、字体分数、候选间距及片段一致性门槛也只用 calibration。
- 选择结果与代码 SHA 冻结后，才打开 test 数组。测试来自新内容和新截图，但仍与父模型属于相同受控模拟器渲染域。
- 字号目标为 `log(截图像素字号 / 图像墨迹高度)`；推理将预测比例还原为输入截图像素字号。颜色从原图测量。

## 复现命令

需先完成 [原生采集](ios-traditional-capture.md)，并准备 `requirements-train.txt` 中的离线训练依赖及 R16 检查点。以下目录必须为新的输出目录；服务容器只推理，不负责训练。

```sh
python scripts/audit_native_capture.py \
  --captures artifacts/mobile-font-capture/ios-hant-capture-v1 \
  --exclude-scenes artifacts/mobile-font-capture/ios-capture-v1/Scenes.json \
  --output artifacts/mobile-font-capture/ios-hant-capture-v1/SOURCE_AUDIT.json

python training/train_regions.py \
  --captures artifacts/mobile-font-capture/ios-hant-capture-v1/labels.jsonl \
  --data artifacts/ios-hant-region-v1/data \
  --prepare-only --group-pingfang --test-history fresh

python training/train_regions.py \
  --data artifacts/ios-hant-region-v1/data --validate-only

python training/train_regions.py \
  --data artifacts/ios-hant-region-v1/data \
  --output artifacts/ios-hant-region-v1 \
  --warm-start artifacts/ios-region-font-v1/region/model.pth \
  --allow-new-traditional-families --preserve-size-head \
  --device mps --steps 4000 --batch-size 64 --eval-every 250 \
  --learning-rate 0.0001 --seed 2026091293
```

ONNX 导出后需要独立比较原 PyTorch GroupNorm 的两个输出及区域聚合结果。稳定导出只调整 GroupNorm 的中间归约精度，不修改冻结权重、门槛或数值容差。然后对完整原始截图运行 [区域评测](region-font-evaluation.md)，分别报告简体、繁体、英文、数字的正确、错误、待确认和漏检数量。

原生受控页面测试不能证明第三方 App、物理手机或未知字体的准确率。实际训练与最终部署结果以对应版本的验证报告为准。
