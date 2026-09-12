# 字体神经网络训练

这里训练的是字体名称分类 CNN，与 OCR 模型分开。输入为原图裁出的单字，输出 15 个字体家族的原始 logits；运行时按中文/数字英文的类别集合计算候选。参考字形匹配不是这个网络的预测步骤。

当前模型从旧 R8 中文 CNN 初始化，保留卷积特征提取层，扩展字体分类层。训练数据来自有字体文件来源的渲染字形，包含 CoreText、FreeType、多字号、JPEG 与轻微采样扰动。Roboto 来自 Google Fonts 官方仓库的固定提交。当前没有获得可核验真实字体标签的手机截图，因此本轮是合成数据训练，不应表述为“已经用大量真实 iOS/安卓截图训练完成”。

## 文件

- `network.py`：64×64 灰度输入，四组卷积、128 维特征和 15 类分类层。
- `data.py`：校验旧训练数组的冻结 SHA，按字符划分并生成数字/英文数据。
- `build_han_freetype.py`：独立 FreeType 中文渲染器；记录不能渲染的内部字体来源。
- `train.py`：实际 AdamW 训练、仅校准集选权重/门槛、随后测试与 ONNX 导出。
- `prepare_screenshots.py`：可选的真实截图人工字体标签导入流程；本轮没有使用真实标签数据。

仅 NumPy/Pillow 的参考库构建不算训练。本训练器明确记录反向传播后的权重 SHA、优化器步数、每轮 loss、选中检查点和源数组 SHA；产物位于 `artifacts/neural-font-v1/`。

## 复现

使用 Python 3.10+、PyTorch、NumPy、Pillow、fontTools 与 ONNX。训练环境不需要加载线上 OCR。旧数据根目录需要包含 R8/R10 的冻结训练、校准、测试数组及字体文件；来源字体不能随意换成同名字体。运行前先按 `downloads/roboto-google-fonts/SOURCE_MANIFEST.json` 取得相同的官方 Roboto 字体。

```sh
python training/build_han_freetype.py --old-root /absolute/path/to/alipay-ai-inference --output /absolute/path/to/new-run/data/han-freetype
python training/train.py --old-root /absolute/path/to/alipay-ai-inference --output /absolute/path/to/new-run --device cpu --epochs 12 --steps-per-epoch 100
```

训练输出目录一旦包含 `TRAINING_FREEZE.json` 就不能覆盖。需重新训练时使用另一个 `--output`，保留此前结果。`--device mps` 需要 PyTorch 可以使用 Apple GPU；`auto` 在可用时选择 MPS。

实际本轮使用仓库内已有的 PyTorch 2.8 / Python 3.10 环境，以及 `artifacts/neural-font-v1/dependencies` 中的导出依赖。`frozen-training/` 保留执行时的训练代码快照。

## 验证范围

合成数据按字符隔离训练、校准、测试，同一字符的尺寸及增强不会跨集合。中文测试沿用既有 R10 测试字符，是本轮训练和父模型未训练的字，仍属于历史使用过的基准，不能称为首次公开的新研究测试。

模型选型依据校准集，优先保留旧 R8 中文宏平均正确率不下降超过 3 个百分点的检查点。所有模型若都不满足该条件，只能导出研究候选并明确报告。门槛对单字和 2/4 个不同字的概率聚合校准；重复字符先平均，避免把重复同字当成额外字体证据。

报告中的“before”是神经网络初始化，不是线上参考字形算法。数字/英文新增类别的初始化层没有受过字体训练，因此这一项训练前后差值只能证明网络发生学习，不能证明它超过现有参考字形匹配。库外字体拒识、真实安卓/iOS截图准确率，需要另行用真实字体来源标签验证。
