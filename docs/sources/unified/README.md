# 联合字体模型的数据来源

本目录为本地现成原生截图数据的公开精简索引。路径均相对于仓库根目录，不包含用户名、设备 UUID、用户上传图片、训练文本或原始字体文件。完整本地清单由 SHA 绑定；精简索引不替代逐区域原生字体证明。

- [SOURCE.json](SOURCE.json)：四个输入数据根目录、24 个具名字体的实际采集来源、标签分组、唯一初始化检查点与来源清单 SHA。
- [DATA_AUDIT.json](DATA_AUDIT.json)：联合分区的页面／区域／视图／图块计数、来源组件划分、旧测试暴露、代码及清单 SHA。
- [训练协议](../../unified-font-training.md)：单 CNN 联合训练、共同校准门槛、字号真值与实验状态。

## 原始记录

iOS 受控页面通过 UIKit／CoreText 渲染，用 `simctl` 截图；实际字体、字形回退、注册来源和实际字号被采集程序核验。原始说明见 [轻字重与 Roboto](../../font-sans-data.md)、[简繁体原生采集](../../ios-traditional-capture.md)、[未知字体训练](../../font-unknown-rejection.md)。Android 使用真实运行的模拟器 Canvas／TextRunShaper，逐字形验证实际字体文件、TTC 索引和 PostScript 名称，见 [第二批来源](../android-v2/README.md)。这是原生模拟器受控场景，不是实体手机或第三方 App 随机截图。

PingFang／SF Pro／Helvetica 在这批 iOS 采集中使用系统来源；其余具名字体均由采集应用加载资产。尤其 Kaiti SC、Songti SC、HanziPen SC 是已登记本地 TTC 资产，Alipay Number 是应用数字字体；不能据此标成 iOS 内置。HarmonyOS Sans SC、MiSans、OPPO Sans 的来源登记也不把“可下载”改写成开源许可证。Android 开放来源各自保留 OFL、文泉驿双许可证或 Fandol GPL 字体例外等原文，不能一概标作 OFL。

## 标签和派生规则

PingFang SC／TC／HK 合并成 PingFang；Noto Sans CJK SC／Source Han Sans SC、Noto Serif CJK SC／Source Han Serif SC 分别为视觉组。原生文件、字重和地区身份继续保留在数据证明中。Roboto 的两个宽度、五个字重由同一可追溯可变字体实例化，Noto 与 Roboto 的共同文件在两端具有相同 SHA；它们不按渲染平台重复分为不同类别。

FandolKai 与 Kaiti SC、FandolSong 与 Songti SC 不合并，字体风格名称不是同源证明。Klee One 未作未知负例，因为它与 LXGW WenKai 存在派生关系；WenKai Mono 等同家族变体也不作为相反标签。注册用 PostScript 改名、TTC 字体实例及可变字体实例化的来源、变换与 SHA 保留在原始来源清单；仅改名不会生成一个新的视觉类别。

本轮把原 TRAIN 中 FandolHei、FandolKai、FandolSong、Long Cang、Zhi Mang Xing、Kaiti SC、Songti SC、HanziPen SC 的可信真实字体标签提升为具名类别。其他未覆盖来源保持 `__unknown__`，同时记录真实来源家族用于采样和评估，不把它们笼统命名为“安卓字体”。

## 分区与评估限制

按页面、原图像素身份、原生裁图身份和规范化文本建立全来源连接组件；同组件的区域和全部衍生视图共同分区。原 CAL 保持不变，选定旧 TRAIN 开发来源再按组件分到 TRAIN／CAL／development_holdout；巨组件不拆开。联合分区没有采用旧 TEST 像素，第二批 Android TEST 继续封存。

开发保留集可能被历史初始化网络见过，必须称为开发回归，不能称盲测。原 iOS 及 Android 历史测试结果保持各自版本和范围；Smiley Sans、Liu Jian Mao Cao 的旧留出角色已经在第二批来源登记中变更为 TRAIN。新封存 Android TEST 仍只有原九类具名字体，不能单独证明全部联合类别和 iOS 表现。

联合实验模型 `r21-unified-font-v1-preview` 已完成本地训练及单 ONNX 导出，一致性验证通过，稳定验收未通过。CAL 命名精度／正确覆盖为 95.07%／63.34%；冻结开发回归为 96.49%／51.66%，两者都不是盲测。开发保留图片和未知字体来源来自旧 TRAIN，初始化可能见过部分图片；数据来源索引本身不构成准确率证明。截图或模型评分本身不能证明字体来源、设备系统或截图真实性。
