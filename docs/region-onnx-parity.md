# 整行模型 ONNX 数值一致性

`training/region_parity.py` 校验 `[N,1,64,256]` 整行窗口模型的两个输出：`logits[N,C]` 与 `log_em_ratio[N]`。它与旧单字 64×64 模型的验证器相互独立。

第一阶段由 PyTorch 环境从固定 calibration 分区选择最多 64 个区域，按字体与窗口数量轮流取样，覆盖单窗口和多窗口。每个区域保留全部窗口及原始笔画像素高度。源数据、选中 checkpoint、冻结记录、网络代码与 ONNX 的 SHA 必须匹配；不打开 test 张量，也不改变权重或门槛。

第二阶段由 ONNX Runtime 环境以 batch 1、7、32、128 运行相同窗口，分别比较两个原始输出，并比较完整区域的平均 softmax 概率、字体首选和接收门槛、字号接收门槛与像素字号。字号公式是 `ink_height × exp(median(log_em_ratio))`，偶数个窗口时不能替换成 `median(exp(log_em_ratio))`。

容差在实际模型验证前写入源码和 reference-source.json：

| 比较 | 绝对容差 | 相对容差 |
|---|---:|---:|
| logits | 0.0002 | 0.0002 |
| log_em_ratio | 0.00002 | 0.0001 |
| 区域概率 | 0.00002 | 0.0001 |
| 像素字号 | 0.02px | 0.0002 |

字体首选与各门槛决定必须完全一致。任何数值或决定失败都会记录 `passed: false`，CLI 返回非零退出码，不会因为整体结论相似就把失败重写成通过。报告还核对独立聚合计算与部署 runtime 的聚合函数一致。

本机分为两个 Python 环境运行，以免改变已有训练环境依赖：

```sh
/Users/allenflux/PycharmProjects/alipay-ai-inference/runs/font-family-yuzu-trial-20260908/.venv/bin/python \
  training/region_parity.py reference \
  --run artifacts/ios-region-font-v1 --data artifacts/ios-region-font-v1/data \
  --output artifacts/ios-region-font-v1/parity-reference
.venv-dev/bin/python training/region_parity.py check \
  --reference artifacts/ios-region-font-v1/parity-reference \
  --output artifacts/ios-region-font-v1/onnx-parity.json
```

`tests/test_region_parity.py` 验证两个输出分别出错、非有限数、批量拆分、错误聚合公式、类别窗口覆盖和失败报告持久化。测试中的 session double 只验证检查器行为；正式模型仍须实际运行以上两阶段，不能用测试替代真实 ONNX 数值验证。

## 2026-09-12 正式验证与导出修正

冻结第 3000 步的整行模型，在 64 个 calibration 区域、187 个窗口上完成实际验证。原始导出失败：logits 最大差 0.00177956，52/1496 个元素超容差；11/64 个区域概率超容差。字号回归通过，字体首选和接收决定均一致。`artifacts/ios-region-font-v1/onnx-parity.json` 保留原始失败。

关闭 ONNX Runtime 图优化或仅启用 basic/extended 优化没有改变误差。逐层对照发现首个卷积误差仅 4.17e-7，首个 GroupNorm 后升至 2.11e-4；普通 float32 中心方差展开也未解决长维度归约累加误差。

`training/export_region_stable.py --float64-reduction` 仅将 GroupNorm 的均值、中心方差与归一化中间计算使用 float64，随后转回 float32；所有卷积/线性层权重和门槛保持冻结值。导出器逐项核对参数键和张量完全相等。原始 PyTorch GroupNorm 输出和所有容差保持不变，修正版通过全部 4 个 batch 的两个原始输出、区域概率、字号及门槛检查：logits 最大差 3.624e-5，log ratio 最大差 2.086e-7，超容差元素为 0。报告为 `export-diagnostics/double-parity.json`，修正版模型为 `export-diagnostics/stable-double/model.onnx`；普通 float32 展开失败报告也保留。

修正模型 SHA256：`9fde0988dd6e0fe52fb8cacbdd96d1456eabf01ce3a4b431de9ebfd982f3d304`。同一 CPU 上 3 个窗口、10 次测量的中位运行时间为原版 16.58ms、修正版 16.95ms（2 次预热未计入）；这只是局部耗时诊断，不是完整端到端性能基准。
