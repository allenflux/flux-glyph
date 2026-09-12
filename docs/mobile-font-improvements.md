# iOS / Android 字体覆盖改进

版本：`r14-mobile-fonts-v1`，2026-09-12。本次修改已在本地代码、模型、API 和网页中完成；未发布到远程服务器。

## 改动

- 汉字参考从 986 增至 1194，补充 208 个已在功能样本中发现的缺字，包括「浏、览」。31 个字体样式均通过源 SHA、实际 CoreText URL 和 PostScript 检查；原 986 个 NPY 成员逐个验证字节不变。[验证记录](font-extension-validation.json)。
- 中文候选匹配增加半像素对齐，降低不同字号和重采样的影响；苹方 supported 的原校准距离与门槛保留。
- 新增独立数字／英文字库：SF Pro、Helvetica、MiSans、HarmonyOS Sans SC、OPPO Sans、Alipay Number；62 个字母数字、17 个样式。Alipay Number 仅数字。
- 时间、金额、英文不再直接归类为未覆盖。混排 `font.components` 分别返回中文和数字／英文，不能相互推断。原像素分字支持空格与标点邻居、真实的 1px Latin 字间隙；图标混入、黏连、低置信度或参考过于接近时保留待确认。
- 新模型包含严格校验的 Latin 文件，旧版中文模型仍可加载；推理不需要本机字体、Swift 或外部下载。

## 验证

| 受控中文端到端样本 | 修改前正确 | 修改后正确 |
|---|---:|---:|
| 安卓四个家族合计 | 16/24 | 19/24 |
| MiSans | 3/6 | 5/6 |
| OPPO Sans | 2/6 | 3/6 |
| HarmonyOS Sans SC | 5/6 | 5/6 |
| Noto Sans CJK SC | 6/6 | 6/6 |
| PingFang SC | 6/6 | 6/6 |

固定 36 张完整中文流程：30 个正确名称、0 个错误名称、6 个未输出名称，覆盖率 83.33%；29/30 个输出结果同时满足严格裁字证据。安卓错误名称从 1 降至 0。[完整中文报告](accuracy-report.md)。

额外 432 组渲染字框检查中，安卓正确从 214/288（74.3%）升至 252/288（87.5%），错误命名从 7 降至 1。MiSans 在该额外集由 64/72 降至 62/72；部分此前正确的输入仍会转为待确认，不能把总体改善理解为每张图都改善。[安卓报告](android-alignment-validation.md)。

数字／英文端到端固定 31 例，均使用冻结图片原像素贴图并运行真实检测、OCR、分字与匹配：23 个阳性中正确候选 19、错误 0、待确认 4；8 个 Times/Courier 负例全部拒识。OCR 忽略空白后 31/31 正确。[全流程记录](mobile-latin-e2e-validation.json)。另有 134 个已知字体渲染字框检查：输出候选 66 个且全部正确；16 个未知字体负例全部拒识。[字框匹配记录](latin-accuracy-validation.json)。

另用合成页面进行了本地 HTTP 与真实浏览器检查：22:43 返回 SF Pro 候选，-100.00 返回 Alipay Number 候选，Safari浏览器分别返回中文 PingFang SC 与英文 SF Pro；日期长串因局部分字证据不足仍待确认。1440px 中文和 390px 英文页面的 8 项字体详情检查通过，混排显示正确且无页面横向溢出。

自动回归：84 tests passed，另含 3 subtests；UI 静态契约检查通过。新测试覆盖证据隔离、旋转裁图坐标、低 OCR 抑制、未知字体拒识、缓存上限、模型依赖闭包及篡改拒绝。

以上均为受控输入与功能验证，没有独立真机字体真值，不能声明真实支付宝截图达到上述正确率。用户提供的结果截图已经缩放并画框，未用于字体精度统计。

## 部署与复测

同步整个项目和 `models/`，执行 `docker compose up -d --build`。代码和模型应一起更新；仅重启旧镜像无法加载本次新增能力。

```sh
PYTHONPATH=src python -m pytest -q
python tests/accuracy_eval.py --evaluate
python tests/mobile_latin_eval.py
node tests/ui_qa.js
```
