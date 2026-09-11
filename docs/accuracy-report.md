# Flux Glyph 字体精度验证

评测时间：2026-09-11T13:35:05.551078+00:00

> 结论边界：这是由已校验原始字体文件生成的受控合成端到端评测，不是真实支付宝截图评测，不能据此声称真实截图字体识别率达到 90%。当前没有独立标注的真实字体真值集。

## 严格口径结果

固定样本共 36 张。字体名 accepted precision = 0.964286，strict evidence-complete precision = 0.928571，coverage = 0.777778，total name-correct rate = 0.75，strict total correct rate = 0.722222。uncertain/待确认一律计为未正确。

OCR exact rate = 1.0；分字完整率 = 0.944444；真值字框直送 matcher 的 oracle 正确率 = 0.722222。

| 字体 | name precision | strict precision | coverage | total name correct | strict total correct | OCR exact | 分字状态ok | 分字几何完整 | oracle matcher | name/strict/accepted/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HarmonyOS Sans SC | 1.0 | 1.0 | 0.833333 | 0.833333 | 0.833333 | 1.0 | 1.0 | 1.0 | 0.5 | 5/5/5/6 |
| MiSans | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 0.666667 | 3/3/3/6 |
| Noto Sans CJK SC | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6/6/6/6 |
| OPPO Sans | 0.666667 | 0.666667 | 0.5 | 0.333333 | 0.333333 | 1.0 | 1.0 | 1.0 | 0.166667 | 2/2/3/6 |
| PingFang SC | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6/6/6/6 |
| Songti SC | 1.0 | 0.8 | 0.833333 | 0.833333 | 0.666667 | 1.0 | 0.833333 | 0.666667 | 1.0 | 5/4/5/6 |

## 分字修复前后

| 指标 | 修复前 | 修复后 | delta |
|---|---:|---:|---:|
| accepted_precision | 0.904762 | 0.964286 | +0.059524 |
| strict_evidence_complete_precision | 0.666667 | 0.928571 | +0.261904 |
| coverage | 0.583333 | 0.777778 | +0.194445 |
| total_name_correct_rate | 0.527778 | 0.75 | +0.222222 |
| total_correct_rate | 0.388889 | 0.722222 | +0.333333 |
| ocr_exact_rate | 1.0 | 1.0 | +0.000000 |
| segmentation_status_ok_rate | 0.805556 | 0.972222 | +0.166666 |
| segmentation_complete_rate | 0.583333 | 0.944444 | +0.361111 |

## 失败归因

检测遗漏 0；识字/区域选择失败 0；分字失败 2；字体拒识 7；字体误判 1。

非苹方样本误接受为苹方：0/30 (0.0)。

oracle matcher 与端到端之间的差值用于定位 detector/OCR/CTC+墨迹分字带来的退化；它不代表生产上可获得的准确率。每条样本的文字、分字状态、预测和前三候选见 JSON 报告。

## 固定协议

协议哈希 `2e99954e033b207784ee652b530e62bb1db738cb6c63a892ca8b78cf764cc7af`；输入清单哈希 `70b6a15d913905478211e6062bb29505696de002700b050d8c190c17cdc4c094`。共 6 个字体家族 × 6 个预先冻结的字号/背景/压缩组合。TTC/OTC 用 fontTools 按 exact PostScript name 解出 face index，字体文件 SHA-256 与旧仓字体清单逐项一致；没有字体 fallback。

## 真实截图缺口

状态栏、页面标签或旧模型输出都没有被当成字体真值。本报告只能证明当前受控条件表现。要验证用户要求的真实支付宝截图 90% 目标，需要另建独立人工标注集，并按同一 accepted precision / coverage / total correct 口径盲测。

代表失败样本可视化：`docs/accuracy-failures.png`。
