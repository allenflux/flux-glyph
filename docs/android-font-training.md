# 独立安卓字体模型（实验候选）

当前安卓版本为 **`android-open-fonts-v1-preview`**，使用第一轮在本地训练并冻结的区域字体 CNN，提供主动试用和下载。该模型通过校准和 ONNX 一致性验证，但独立 TEST 未通过原稳定版验收；实验开放不更改验收策略或失败报告。

用户在页面选择 iOS 或 Android 实验候选；该选择只决定使用哪个模型，不鉴定截图来自什么手机。默认 iOS 保留 R20 的分类、拒识和复核网络。安卓模式不调用 iOS 模型兜底。

## 当前下载版的数据与训练

首批已知类别为 Noto Sans CJK SC、Noto Serif CJK SC、LXGW WenKai、WenQuanYi Micro Hei、ZCOOL KuaiLe、ZCOOL XiaoWei、ZCOOL QingKe HuangYou、Ma Shan Zheng 和 Roboto。Roboto 仅取英数；其余类别同时取简体中文和英数。Noto 与思源对应字体按同源中文视觉组展示，不能根据相同字形确认具体发行项目。

[当前下载版来源和许可证](sources/android/README.md) 记录第一轮 35 个静态字体实例、下载版本、文件 SHA、实际名称和 cmap。训练使用其中 33 个：九个已知类别的 26 个实例和七个未知实例。校准另留 Liu Jian Mao Cao，测试另留 Smiley Sans，各一个实例。采集应用显式加载字体资产，不把它们都称为安卓系统默认字体，也不声称覆盖所有手机厂商或主题商店的字体。

原始截图由本地 Android 15 / API 35 Emulator 内实际运行的采集应用生成，第一轮采集 600 张训练、100 张校准和 100 张测试截图，每张 12 个区域，共 9,600 个区域；12 种字号为 16–84 px。字号、颜色、背景和文字布局配对，区域位置打乱，图中不绘制字体标签。训练使用 7,200 个原生区域，得到 28,792 个可用派生视图；校准、测试各有 1,200 个原生区域和 4,800 个相关视图。

首版在本地 MPS 实际执行 6,000 个 AdamW 更新，选中第 6,000 步；已有本地网络只提供初始化，不提供字体标签或教师预测。全部校准视图的原始 PyTorch、缓存 MPS 与 ONNX 字体命名及字号可用性一致。

每个区域通过 Android TextRunShaper 核对实际 glyph 使用的字体文件 SHA、TTC index、PostScript 名称和字形覆盖，并检查是否裁切。仅记录请求的 Typeface 名称不足以作为真值。字形的负边距通过原生位置调整保留完整笔画。

文字、原始页面与内容身份按分区隔离。首版未知类训练来源为 Long Cang、Zhi Mang Xing、Fandol 黑体／宋体／楷体，完整家族与字体文件不跨训练、校准和测试分区。该分区说明不替代初始化模型全部历史来源的独立审计。用户截图不作为具体字体真值，也不上传用于训练。

## 训练与验收方法

原生区域派生四种视图：原尺寸、半尺寸、75% 尺寸加 JPEG 75、原尺寸 JPEG 75。它们是相关视图，不是四倍独立样本。模型只读取图像图块，不使用文字内容、平台标签或参考字形匹配；文字区域检测器可以用于完整截图定位。

训练裁图采用原生字形边界，向外保留 `max(2, ceil(字号 * .15))` 像素并限制在原布局框内。第一轮最初使用整行布局槽时，过量空白让 5,889 / 28,800 个训练视图未通过固定对比度检查；紧裁后仅 8 个缩小的未知字体视图因墨迹高度不足 6 px 被拒绝。此次修正只依据 TRAIN 数据，不更改预处理器或拒绝阈值。采集原图、原生证明和失败审计均保留。给定原生边界的区域评估不能代表整张截图的检测性能。

区域网络接受 float32 `tiles [N,1,64,256]`，输出十类 `logits [N,10]` 和 `log_em_ratio [N]`。九类用于字体命名，额外一类 `__unknown__` 用于拒识。训练对编码器、字体头和字号头执行真实梯度更新；已有本地网络只提供初始化，不提供字体标签或教师预测。

字号目标来自实际原生像素字号和视图缩放比例。颜色由区域原图测量。字号输出是输入截图中的像素值，不是 Android sp、dp 或 iOS pt。

训练前固定的稳定版验收策略为：命名精度至少 98%、已知字体正确命名覆盖率至少 70%、每类至少 50%、原尺寸已知字体首选准确率至少 90%、留出未知字体不被错误命名至少 80%；正确命名结果的字号中位相对误差不超过 10%、P90 不超过 25%、字号输出覆盖率至少 70%。预处理拒绝项计入弃权分母。这些条件保持原样，实验候选开放不将其改成通过。

检查点及温度、分数、差值和图块一致性门槛仅按校准集选择。选定后验证原始 PyTorch、缓存训练输出与 ONNX 的校准集一致性，冻结模型、门槛和评估脚本，最后才准备和推理测试图块。未知类不被命名包括显式拒识、不确定和预处理拒绝，不等于模型正确辨认了未知字体的名字。

这些验收针对受控 Android 原生采集和指定留出字体。通过后也不能把成绩推广为所有真机、App、字体文件版本、主题字体或混合字体的准确率。

首版 TEST 的整体命名精度为 92.60%（2,465 / 2,662），已知字体正确命名覆盖率为 68.47%（2,465 / 3,600）。唯一未知测试家族 Smiley Sans 有 14.5%（174 / 1,200）视图被错误命名。文泉驿每类覆盖仅 46%，因此整体精度、总覆盖和每类覆盖三项未通过。100 张测试页面上的四种视图相互关联，这些数字不代表所有真机和未知字体。

实际通过与失败结果见[逐轮验证记录](android-font-results.md)。模型分数不是正确概率；不能用来源字体的汉字覆盖数量或训练步数代替识别准确率。

## 后续未发布实验

[第二轮来源登记和原始许可证](sources/android-v2/README.md) 扩充到 63 个静态实例，采集 1,000 张训练、150 张校准、150 张封存测试截图，共 15,600 个区域，字号细分为 19 档。字重按字体家族、字号和脚本独立轮换；第二轮文字排除第一轮全部分区的 1,218 个规范化文字。原生字体证明与预处理计数见 [CAPTURE_VERIFICATION.json](sources/android-v2/CAPTURE_VERIFICATION.json)。

扩充实验的未知训练来源加入首轮使用过的 Liu Jian Mao Cao、Smiley Sans，以及 Lato 和 Open Sans；校准为 M PLUS 1p、Yusei Magic 和 Ubuntu，新的封存测试为 IBM Plex Sans SC、Yuji Syuku 和 Liberation Sans。旧校准、测试家族改作训练后，不再称为本轮未见来源。

第二轮 10,000 步训练、第三轮 6,000 步均衡采样／OE 微调、OVR 4,000 步独立神经头训练均未通过 CAL，没有读取新测试进行预处理或推理，也没有发布。63 个来源实例及这些后续训练步骤**不属于 `android-open-fonts-v1-preview`**，不能作为当前下载版已改进的证据。

## 使用和发布

网页顶部选择“Android 实验候选模型”后，同时切换识别和模型下载；提交后任务绑定当时的模型，切换页面模式不会改变已提交任务。安卓模型缺失或损坏时该模式返回 503，不会改用 iOS 网络。

```sh
# 安卓模型信息与独立 ONNX 工具包
curl 'http://localhost:9000/api/models/font?mode=android'
curl -o android-font-onnx.zip 'http://localhost:9000/api/models/font/download?mode=android'

# 完整图片：检测文字区域后运行安卓字体网络
curl -F 'file=@screenshot.png' 'http://localhost:9000/api/jobs?mode=android'

# 下载包解压后，直接识别一个文字区域
pip install -r requirements.txt
python predict.py text-region.png
```

结果包含 `font_mode`；`device_inference_performed` 为 false。安卓模型信息接口和 ZIP 内元数据还声明 `release_tier: experimental`、`test_passed: false`、`stable_validation_passed: false`，`validation` 包含实际评测与适用范围；预测 API 与 CLI JSON 使用 `model_release_tier: experimental` 和 `stable_validation_passed: false`。未指定模式的请求继续使用 iOS。两种模式共用一个有界任务队列和单个执行线程。

安卓模型独立存放在 `models/android/`，由该目录的 `ACTIVE.json` 选择；可用 `FLUX_ANDROID_MODEL_DIR` 改写根目录。原 `models/ACTIVE.json` 继续控制默认 iOS 模型。训练及采集均在本地执行，代码、ONNX 和清单通过 Git `main` 更新，服务器只执行推理。首版实验候选保留已验证的导出一致性和失败 TEST 状态，完整披露后提供试用，不标成稳定版验收通过。
