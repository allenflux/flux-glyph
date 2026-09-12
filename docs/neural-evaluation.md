# CNN 固定样本端到端评测

这是固定字体渲染样本的检测、OCR、分字、神经网络分类评测；没有独立标注的真实手机截图真值。
iOS／Android 分组仅表示参考字体家族，不是图片拍摄设备或系统标签。

模型：`r15-neural-font-v1`；权重 SHA-256：`5b59de4f9061196d3a9f81ed431c5afc303ca3e49b41f96a8bf568c693415ab7`。

| 分组 | 总数 | 正确字体名 | 错误字体名 | 命名数 | 严格正确 | 弃权 | 正确拒判 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| by_script/han | 36 | 0 | 0 | 0 | 0 | 36 | 0 |
| by_script/latin | 31 | 0 | 0 | 0 | 8 | 23 | 8 |
| by_cohort/android_font_references | 36 | 0 | 0 | 0 | 0 | 36 | 0 |
| by_cohort/application_font_reference | 3 | 0 | 0 | 0 | 0 | 3 | 0 |
| by_cohort/ios_font_references | 14 | 0 | 0 | 0 | 0 | 14 | 0 |
| by_cohort/other_or_negative_font_references | 14 | 0 | 0 | 0 | 8 | 6 | 8 |
| by_positive_negative/negative | 14 | 0 | 0 | 0 | 8 | 6 | 8 |
| by_positive_negative/positive | 53 | 0 | 0 | 0 | 0 | 53 | 0 |
| by_family/Alipay Number | 3 | 0 | 0 | 0 | 0 | 3 | 0 |
| by_family/HarmonyOS Sans SC | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| by_family/Helvetica | 4 | 0 | 0 | 0 | 0 | 4 | 0 |
| by_family/MiSans | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| by_family/Noto Sans CJK SC | 6 | 0 | 0 | 0 | 0 | 6 | 0 |
| by_family/OPPO Sans | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| by_family/PingFang SC | 6 | 0 | 0 | 0 | 0 | 6 | 0 |
| by_family/SF Pro | 4 | 0 | 0 | 0 | 0 | 4 | 0 |
| by_family/Songti SC | 6 | 0 | 0 | 0 | 0 | 6 | 0 |
| by_family/unknown_negative | 8 | 0 | 0 | 0 | 8 | 0 | 8 |

中文沿用原严格口径：OCR 完全一致、全部分字状态正常、字框覆盖真实墨迹至少 90%、水平字框容差 ±2px，且字体候选正确。
拉丁沿用原口径：OCR 忽略空白后一致，所有检测区域给出正确候选；未知字体没有候选才是正确拒判。
中文宋体 negative 是针对苹方的已知字体反例，仍应识别为 Songti SC；它与 Latin 未知字体拒判样本含义不同。
字体名正确但裁字几何不完整的中文样例，仅未通过严格正确口径，不记为错误字体名。
正例待确认计为弃权，不计正确；阈值在评测前后均校验未修改。
