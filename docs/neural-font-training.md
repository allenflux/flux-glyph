# 字体名称神经网络

后续已实现 iOS 原生模拟器截图采集及专门的截图训练入口，见 [iOS 截图训练](ios-screenshot-training.md)。以下保留第一版合成字形实验及其复现方式；其中的 15 类、175,134 个字形和精度结论均属于历史实验，不是当前 iOS 截图训练集。

本轮新增真正训练的字体分类 CNN。OCR 先定位和识读文字，再从原图提取单字；CNN 将 64×64 字形输入卷积层，输出各字体族的 logits。训练时用已知源字体名称计算分类损失并反向更新网络权重。推理时不查参考字库，也不计算截图与参考字形的距离。

支持 15 个字体族：中文 11 类；数字／英文 7 类，其中 MiSans、HarmonyOS Sans SC、OPPO Sans 同时覆盖两种文字。另有 SF Pro、Helvetica、Alipay Number、Roboto。Alipay Number 的训练源只有数字。中文和数字／英文分别预测，不能从中文结论推断同框数字的字体。

输出为 `candidate` 或 `uncertain`。模型分数是已知类别之间的相对评分，不是真实准确率，也不能保证未知字体会被拒绝。“待确认”表示 OCR／裁字证据不足，或模型分数、第一与第二候选差值未过门槛；一个字也可预测，但其可区分信息可能较少。

## 本轮训练与评估

训练源包括经过原始 SHA 核验的 R8、R10 字形数组、新增 FreeType 中文渲染，以及数字／英文渲染。新下载的 Roboto 来自 Google Fonts 官方仓库，原始来源、许可证与 SHA 见 [数据审计](neural-data-audit.md) 和 `sources/roboto/`。

本轮训练集有 175,134 个字形，校准集 35,608 个。它们共享部分字符与源字体，包含字号和压缩派生，不能当作 175,134 张独立手机截图。训练／校准／测试按字符划分；同一字符的合成变体留在同一集合。R10 测试是历史固定回归集，并非首次开启的新盲测。

在旧 R8 CNN 的卷积／特征层基础上继续训练并扩展分类头。执行 12 轮、每轮 100 次 AdamW 更新，batch 96；按字体均衡采样，训练中加入缩放、轻微模糊、像素位移与噪声。损失为分类交叉熵和旧中文教师的保护项。只用校准数据选择权重、温度及输出门槛，再运行固定测试。

**本轮没有经来源验证的真实手机截图字体名称监督标签。** 因此交付的是合成数据训练的第一版神经网络，不能宣称已用大量 iOS／安卓真机截图完成训练，或保证优于现有参考匹配器。训练前后结果、权重 SHA 与参数变化记录在 [训练报告](neural-training-report.json)；完整流程评估另见 `neural-evaluation.json`（生成后提供）。

## 复现训练

训练依赖与线上 CPU ONNX 推理依赖分开。原始训练数组、旧 checkpoint 和部分本地源字体来自相邻实验仓库，不包含在部署包内；它们缺失时会报错，不会使用伪标签代替。

```sh
python3.11 -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt
.venv-train/bin/python training/download_sources.py
.venv-train/bin/python training/build_han_freetype.py \
  --old-root /Users/allenflux/PycharmProjects/alipay-ai-inference \
  --output artifacts/neural-font-v1/data/han-freetype
.venv-train/bin/python training/train.py \
  --old-root /Users/allenflux/PycharmProjects/alipay-ai-inference \
  --output artifacts/neural-font-v1 --device cpu --epochs 12 --steps-per-epoch 100
```

已冻结的训练输出不覆盖；重复实验应换一个 `--output`，重新构建该目录的 FreeType 数据。完整流程验证需要生产依赖，而训练环境只需上述训练依赖。checkpoint 保存在训练目录，线上仅使用导出的 ONNX 与 metadata，不加载 PyTorch pickle。

## 导入已知字体的真实截图

`training/prepare_screenshots.py` 读取逐框字体标注 JSONL。例如下面记录中 `font_family` 必须来自采集应用的实际字体配置或可靠标注，不能根据 iOS／Android 型号猜测：

```json
{"image":"capture.png","source_id":"capture-session-01","split":"train","regions":[{"bbox":[20,40,260,100],"text":"账单详情","font_family":"PingFang SC","script":"han"}]}
```

划分使用 `train`、`calibration`、`test`。同一来源及其压缩／缩放派生应使用相同 `source_id` 并放在同一集合。混排可对同一框分别标注 `han` 与 `latin`。导入器校验 OCR 文本、分字、重复源图／区域像素与跨集合来源冲突；未通过样本保留拒绝原因。OCR 只核验文字内容，不能证明填写的字体名称正确。

```sh
PYTHONPATH=src .venv-dev/bin/python training/prepare_screenshots.py \
  --input labels.jsonl --output artifacts/real-font-labels
.venv-train/bin/python training/train.py \
  --old-root /Users/allenflux/PycharmProjects/alipay-ai-inference \
  --output artifacts/neural-font-v2 --real-data artifacts/real-font-labels
```

## 生成独立模型包

新后端由完整包 MANIFEST 中的 `neural/metadata.json` 启用。未声明 neural 的旧包继续使用原字体匹配方法。

```sh
PYTHONPATH=src .venv-dev/bin/python scripts/package_neural.py \
  --neural artifacts/neural-font-v1/neural \
  --output artifacts/neural-font-v1/bundle --version r15-neural-font-v1
PYTHONPATH=src .venv-dev/bin/python scripts/model_release.py \
  --model-root artifacts/neural-font-v1/bundle export \
  --output artifacts/flux-glyph-neural-font-v1-models.zip
```

打包不会切换现有模型。可以用 `FLUX_MODEL_DIR=artifacts/neural-font-v1/bundle` 启动单独的本地 API 进行评估；正式切换仍使用项目现有的版本导入机制。神经网络模式保留旧参考文件以兼容完整包格式，但不调用旧匹配结果补充或覆盖 CNN 输出。
