# iOS 系统字体优先训练

本轮重点是 PingFang、SF Pro 和 Helvetica。模型仍以区域图像为输入，同时预测八类字体和像素字号；检测模型仅定位，不读取文字。其他五类字体继续参与训练，防止把外观相近的应用字体都归到系统字体。

训练只在本地 Mac 的 PyTorch／MPS 环境执行。通过验证后，才将代码和模型提交到 Git `main`，由服务器拉取对应提交并重建推理容器。原始训练截图、临时检查点和训练依赖不需要部署到服务器。

## 固定训练方案

使用 R17 检查点继续训练，保留原字号回归头。沿用已经核验来源的 8,744 个训练区域和 1,093 个校准区域，不把测试集加入训练。

- PingFang、SF Pro、Helvetica 的字体采样权重各为 2；其他五类各为 1。系统字体合计占采样计划的 `6/11`，各字体内部仍使用打乱且不放回的区域队列。
- 执行 4,000 次 AdamW 更新，batch 64，学习率 `0.00003`，seed `2026091294`；每 250 步评估校准集。
- 沿用父模型的温度 `0.5`，分数门槛 `0.5`、分差门槛 `0.01`、图块一致率门槛 `0.7`，字号相对离散度上限 `0.2`。
- 只通过校准集选择检查点：系统字体正确命名必须增加，同时约束错误数量和已有正确结果。没有合格检查点时，保存失败证据，不导出或发布替代模型。

在本地训练环境中运行；输出目录必须为新目录：

```sh
python training/train_regions.py \
  --data artifacts/ios-hant-region-v1/data \
  --output artifacts/ios-system-finetune-v1/run \
  --warm-start artifacts/ios-hant-region-v1/region/model.pth \
  --parent-metadata artifacts/ios-hant-region-v1/bundle-stable/region_neural/metadata.json \
  --system-focused --system-family-weight 2 --preserve-size-head \
  --steps 4000 --batch-size 64 --learning-rate 3e-5 \
  --eval-every 250 --seed 2026091294 --device mps --test-history reused \
  --regression-protocol artifacts/ios-system-finetune-v1/REGRESSION_PROTOCOL.json
```

`--system-focused` 对应上述固定实验，检查预算、父模型参数哈希与判定规则。普通训练不启用该选项时，默认仍按各字体等权采样。`--test-history reused` 只记录本轮评估历史，不修改原始数据清单。

## 验证与发布条件

校准集选择唯一检查点后，冻结权重、代码和判定规则，再验证 ONNX 与原 PyTorch 的两个输出及最终区域判定是否一致。完整截图验证复用 R17 的 100 张原生页面和 67 个历史字体样例；这些属于固定回归集，不是新盲测。

发布要求系统字体正确命名增加、整体及系统字体错误不增加、原有正确命名保留、未知字体误认不增加。不能把检测数量不符当作未知字体成功拒识，也不能根据回归结果改门槛或换另一个检查点。

先前仅放宽系统字体图块一致率的候选已被拒绝：原生回归中系统字体正确命名由 518 增至 539，但整体错误由 1 增至 2；历史样例也新增 Noto Sans 被误认成 PingFang。该候选未上线，线上判定继续使用原 R17。上述失败是改为实际本地微调的原因。

训练与验证记录保存在 `artifacts/ios-system-finetune-v1/`；运行前的 `REGRESSION_PROTOCOL.json` 固定数据哈希、父模型、训练方案和验收规则。最终结果以同目录报告为准。受控模拟器回归结果不能代表所有真机、第三方应用或未知字体的识别率。
