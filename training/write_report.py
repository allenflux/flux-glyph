"""Write the evidence-based training note after the frozen evaluation finishes."""
from __future__ import annotations
import json
from pathlib import Path
from data import ROOT

out=ROOT/'artifacts/neural-font-v1'
r=json.loads((out/'report.json').read_text());s=json.loads((out/'SELECTION_FREEZE.json').read_text())
h=json.loads((out/'training-history.json').read_text());freeze=json.loads((out/'TRAINING_FREEZE.json').read_text())
rows=[]
for d in r['datasets']:
    label='R10 原生中文小字' if d['name'].endswith('20260911') else ('FreeType 中文' if d['script']=='han' else '数字/英文')
    rows.append(f"| {label} | {d['after']['rows']:,} | {100*d['before']['accuracy']:.2f}% | {100*d['after']['accuracy']:.2f}% | {100*d['after']['macro_accuracy']:.2f}% |")
eligible=s['selected']['eligible']
lines=[
'# 字体神经网络 v1 训练结果',
'',
'已完成真正的字体 CNN 训练和 ONNX 导出。此轮数据是有字体文件来源的合成字形；没有使用已核验字体标签的真实手机截图。',
'',
'**本轮未达到替换现有识别算法的标准，只交付可选择的神经网络实验后端。** 原生中文小字留出测试下降，数字/英文分类正确率仍较低；中文与数字/英文均未在校准集找到符合预定精确率要求的接受门槛。',
'选中检查点'+('通过' if eligible else '未通过')+'旧 R8 中文校准保护条件。该条件与超过当前参考字形算法是不同的验证目标。',
'',
'## 已执行训练',
'',
f"- 15 类轻量 CNN，64×64 单字输入，中文 11 族、数字/英文 7 族按脚本分别判断。卷积层和中文分类权重从已训练的 R8 模型初始化。",
f"- CPU 4 线程，PyTorch {freeze['torch']}，{len(h)} 轮 × {freeze['steps_per_epoch']} 步，合计 {r['optimizer_steps']:,} 次 AdamW 更新，每批 96 个字形。",
f"- 完整训练与校准选型耗时 {r['training_seconds']:.1f} 秒；第一轮平均 loss {h[0]['loss']:.4f}，末轮 {h[-1]['loss']:.4f}。选中第 {r['selected_epoch']} 轮，即累计 {r['selected_epoch']*freeze['steps_per_epoch']} 步时的权重。",
f"- 训练数据合计 {sum(v['rows'] for v in freeze['train_sources']):,} 字形；校准数据 {sum(v['rows'] for v in freeze['calibration_sources']):,} 字形。旧数组与原冻结 SHA 逐一比对。",
'- 新增 FreeType 汉字：23/31 个具体字体/字重可解析，8 个 Apple 内部来源明确保留未覆盖；训练 8,832、校准 2,944、测试 2,944 个字形。',
'- 数字/英文训练 7,582、校准 920、测试 768 个字形；训练期 78 个细窄字形未通过与运行时相同的裁字质量门槛，已在数据清单记录。',
'- Roboto 来自 Google Fonts 官方固定提交，包含 OFL 许可、下载 URL、字体版本、变体轴与 SHA 验证记录。其他字体沿用项目已有的可核验源文件。',
'',
'## 留出测试',
'',
'训练、校准、测试按字符隔离；同字的多个字号和退化视图不跨集合。只用校准集选择检查点、温度与门槛，再读取测试数组。R10 测试是历史使用过的基准，不是首次使用的新研究测试。',
'',
'| 条件 | 单字数 | 神经网络初始化 | 训练后 | 训练后字体宏平均 |',
'|---|---:|---:|---:|---:|',
*rows,
'',
'“初始化”比较对象是旧中文神经网络和新增随机数字/英文分类层，不是当前参考字形匹配器。数字/英文前后变化说明模型学到了训练数据的特征，不能据此宣称比现有参考匹配更准。',
'',
'上表是来源已知的合成字体测试，不能当作 iOS/安卓真实截图准确率。高分候选也不表示物理设备来源、截图真实性或全部未知字体可被拒绝。逐字预测、2/4 个不同字的概率聚合结果和接受/误接受数在 JSON 报告中保留。',
'两种脚本均未找到接受结果精确率至少 97% 且接受数至少 20 的校准门槛。导出的 0.99 分数 / 0.4 分差只是保守备用值，没有通过上述目标验证。',
'',
'## 可核验证据',
'',
f"- 训练前权重 SHA：`{r['weights_before_sha256']}`。",
f"- 训练后权重 SHA：`{r['weights_after_sha256']}`。",
f"- 首层卷积权重变化 L2：{r['parameter_l2_change']['trunk.0.weight']:.8f}；分类层变化 L2：{r['parameter_l2_change']['family_head.weight']:.8f}。",
f"- ONNX：{r['onnx_bytes']:,} 字节，SHA `{r['onnx_sha256']}`。",
'- `artifacts/neural-font-v1/TRAINING_FREEZE.json`：预算、类别、来源、训练前权重与执行代码 SHA。',
'- `artifacts/neural-font-v1/training-history.json`：每轮真实优化器步数、损失和校准结果。',
'- `artifacts/neural-font-v1/SELECTION_FREEZE.json`：测试前固定的检查点与判定参数。',
'- `artifacts/neural-font-v1/neural/`：ONNX、PyTorch 权重与运行时元数据。',
'- `docs/neural-training-report.json`：全部留出评估结果及具体字体混淆。',
'- `training/prepare_screenshots.py`：后续导入人工确认字体名称的真实截图入口；本轮未使用。',
'',
'训练结束后补强了 `--real-data` 的分区 SHA、明确字体标签、类别顺序、字符脚本和来源隔离验证；真实截图按单字评估，不跨图片拼接虚构文字区域。这些后续输入校验修改没有改变本轮权重或测试结果，实际执行代码仍由冻结快照和 SHA 保留。',
]
(ROOT/'docs/neural-training-report.md').write_text('\n'.join(lines)+'\n')
