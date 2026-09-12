# 字体神经网络训练数据审计

审计日期：2026-09-12。范围为本项目 `tests/fixtures` 与相邻 `alipay-ai-inference/runs` 中已定位的字体实验；不是对全部磁盘或全部历史实验的穷尽认证。审计阶段未执行训练、修改历史数据或给截图补字体标签；后续模型训练由独立流程记录。

## 结论

本地已有可用于字体分类网络的、由明确源字体生成的合成训练数据，以及相关历史神经网络实验。已核验的截图清单只有人工平台判断、状态栏教师伪标签或无标签，**没有找到经来源验证的真实截图字体名称真值**。因此可以立即做合成数据预训练和受控验证；现有资料不足以声称完成大量真机截图监督训练或报告真机字体准确率。

## 源字体合成数据

以下实验路径均位于 `/Users/allenflux/PycharmProjects/alipay-ai-inference/runs/`。各实验存在复用、派生、增强关系，数量不能直接相加当作独立样本数。

| 数据入口 | 数量与划分 | 字体/渲染覆盖 | 可用性与限制 |
|---|---|---|---|
| `alipay-font-generalization-v8-20260911/data/arrays/{train,calibration,test}` | 63,488 / 7,936 / 7,936 个单字；1024 / 128 / 128 个不同汉字 | 31 faces、11 families，CoreText 精确源字体，字号 32/48 | 本次核验 train/cal 的元数据和 NPY SHA 全通过，全部引用 PNG 存在；test 仅检查文件存在和冻结协议，不读取像素或评分 |
| `alipay-font-smalltext-v10-20260911/data/arrays/{train,calibration,test}` | 95,232 / 23,808 / 23,808 个单字；1024 / 128 / 128 个不同汉字 | 同一 31 faces，训练字号 14/20/28；cal/test 为 16/24/40、identity/jpeg75 | train/cal 数组为 `[N,64,64]`；本次核验元数据和 NPY SHA 全通过，全部引用 PNG 存在；test 仅检查存在 |
| `apple-font-finetune-v1-20260910/data/{train,calibration}.jsonl` | 23,990 / 5,272 行，对应 15,750 / 2,956 个源图片路径 | 6 个命名字体类＋OTHER；CoreText/FreeType；clean/degraded 派生 | 本次重新统计行数、确认全部路径存在；旧 `source-audit.json` 记录 28 个源字体文件和全部图片 SHA 验证。没有在本次重新逐图片哈希 |
| `alipay-font-open-set-v11-20260911/data/SPLIT.json` | 新生成计划：known-test 11,904；unknown-train 9,216；unknown-cal 1,920；unknown-test 2,688 张源 PNG | 非已知字体训练/校准负例；新字体 Lantinghei SC、Wawati SC 仅 test | 这里数量来自冻结 split 的生成计划，未在本次逐项验收生成结果。不可把未知字体测试集改作训练输入后仍声称保留盲测 |

R8/R10 的字体清单为 `alipay-font-generalization-v8-20260911/data/fonts.json`，SHA-256 为 `8cf0910c19d77e04897d7bd3fc5289a82ef99aa59bc554b078a24eb653f351b1`。本次对清单中 **31/31 个源字体条目重新计算 SHA，全部符合声明**。含同文件多个 face 的情形，31 条不表示 31 个独立字体文件。

| 字体家族 | face 数 |
|---|---:|
| PingFang SC | 12 |
| HarmonyOS Sans SC | 3 |
| Noto Sans CJK SC | 3 |
| OPPO Sans | 3 |
| MiSans | 2 |
| Heiti SC | 2 |
| Hiragino Sans GB | 2 |
| Songti SC、Baoli SC、Kaiti SC、Yuanti SC | 各 1 |

这些是字体家族标签。把安卓字体文件放在 CoreText 中渲染，不等于收集到了 Android 实机渲染；PingFang 字体标签也不等于截图设备标签。

## 划分与防泄漏

- 本次重新计算 R8、R10、R11 `data/SPLIT.json` 中各自 train/cal/test 字符集合交集，均为 0。
- R10 train/cal 继承 R8 的字符划分；R10 新 test 排除了 R8 train/cal/test 和 R7 字符。R11 再使用新的 test 字符；其冻结协议还排除 R10 test。
- R8 的训练字符明确包含“账单详情”等常用标题字。因此本项目 36 张受控标题图可用于完整流程回归，不能当作 R8/R10 未见字符测试。
- 这些 test 已用于历史实验报告；在新实验中仍可作为固定回归集，但不能称为从未见过的新盲测集。新的模型选择应先冻结协议，训练及调参只读取允许的 train/cal。
- Apple finetune 的旧源审计报告 train/cal 文本交集为 0、源图片 SHA 交集为 0；源字体 SHA 跨集合相同是已知字体分类的预期。clean/degraded 视图共享源图片身份，应在同一 split。

本次重新验证的数组 SHA：

| 数组 | SHA-256 |
|---|---|
| R8 train | `ddfb45fa997921d50a29fb2dcf936e74afc6027dbe6508de77c36c91412e582b` |
| R8 calibration | `eac6fdae4d81e5c3f659acfb86e56c194f0905e56d3494096a671accdfce5865` |
| R10 train | `ced7c5e2790dff5b7c94a88fc26075fe1a1a39688e6a92422c14267c234de45b` |
| R10 calibration | `12cf0b51e3ea301cd09d94db8a92d5f5df30d08cf30a1b83ccd2dcf6dff9780a` |

## 原始截图与标签性质

这里的“原始截图”指清单中的源图片文件，不代表已验证拍摄设备、内容真实性或字体来源。

| 已定位清单 | 数量 | 标签性质与可用范围 |
|---|---:|---|
| `font-device-human-repair-train-v1-20260908/data/records.jsonl` | 123 个来源；人工 ios 51/android 50/uncertain 2；20 未标注 | `label-source-manifest.json` 明确 `font_identity_labels=0`、`device_provenance_verified=false`。人工目测平台判断不能转换为具体字体家族监督标签 |
| `apple-font-finetune-v1-20260910/device-data/image-manifest-dedup.jsonl` | 1000 原图；1411 标题裁图；eligible 511 | 标签来自旧状态栏教师，去重后 eligible train/cal/test=357/77/77。分组控制派生像素泄漏，不证明字体或硬件来源 |
| `alipay-font-device-expanded-validation-v1-20260910/data/selected-images.jsonl` | 2000 原图，blue/white 各1000 | 原始选样无字体/设备标签；后续教师输出 ios541/android1442/uncertain17，均为伪标签。历史提取报告 2819 标题裁图，1910 图有标题 |
| `alipay-font-real-fields-v9-20260911/data/selected-sources.jsonl` | 132 原图、158 标题裁图 | 60 人工标记 ios 来源＋72 状态栏导出 ios 来源；`font_name_ground_truth=null`，只宜提取、诊断或后续标注 |
| `alipay-font-originals-v14-20260911/data/selected.jsonl` | 100 原图、2786 中文区域 | 源文件抽样未按字体/设备预测筛选；`font_ground_truth` 未提供。区域可能重叠，2786 不是独立截图数 |

上述数量中有重叠，不能相加。旧教师的高置信度、旧参考匹配器候选、人工“像 iOS”判断、文件夹 blue/white 或手机品牌，都不构成字体名称真值。旧教师伪标签集不应进入新字体分类器的监督训练目标。

R9 的 `discovery/data-audit.json` 也明确记录检查过的实图来源 `verified_font_identity_records_found=0`。R14 `REPORT.md` 明确原始截图没有源字体文件、渲染日志或可信字体标注。

## 本项目 fixtures 与 Latin

- `tests/fixtures/font_accuracy/generated_manifest.json`：36 张合成页面，6 个家族各6张（PingFang、Noto、HarmonyOS、MiSans、OPPO，及 Songti 负例）。每张包含源字体 SHA、PostScript、图片 SHA、文字/字框。**本次图片 SHA 36/36 通过**。
- `tests/fixtures/latin_accuracy/cases.json`：31 张 Pillow 合成字框，38px；SF Pro、Helvetica、HarmonyOS、MiSans、OPPO 各4张，Alipay Number 3张，Times/Courier 负例8张。**本次图片 SHA 31/31 通过**。`expected_family` 来自生成源；同文件的 `family/status/top_ranked_family` 是匹配器输出，不能用作标签。
- Latin 固定内容为短时间、金额、日期和 Safari，不能代表大规模 Latin 字符泛化。源字体发现与校验函数在本项目 `scripts/build_latin_references.py`：SFNS.ttf 含 optical 17/48 × weight 400/600/700 的6实例；Helvetica.ttc regular/bold 2 faces；Android 8 faces；Alipay Number 只有 regular 且仅数字。共17 styles、6 families。
- Latin 现有源哈希：SFNS.ttf `2bfd40dc72e6759e248f82a52a40d551338979fffc9b5c070e685b4b7ad19e66`；Helvetica.ttc `25eceb458d4baf628ee0b6a135a9f8ac5ec7b2826646720834bd2bf00dfc825a`；AlipayNumber-Regular.ttf `6074082d8cb92e175184b177e28335e0171f3ddfb2b5818d7853295f7fb0fada`。这是构建脚本的固定来源声明，Latin 源字体未在本次重新逐个哈希。
- `tests/fixtures/ui_title_billing_details.png` 与相邻 Vision JSON 用于文字/分字回归，未发现独立字体来源真值，不纳入有字体标签的实图数量。

## 真机监督训练仍缺的资料

需要按文字区域记录可验证的 `font_family`、具体字体文件/face/版本及 SHA、原始截图 SHA、裁图坐标、系统/渲染环境与采集会话。来源证明可来自控制字体的采集应用和其实际加载日志；不能从型号猜字体。混排中文、数字和英文需要分别标注字体。

训练/校准/测试应按源截图、采集会话和同源派生组隔离，并检查原图 SHA、裁图/归一化像素 SHA；同一截图的缩放、压缩或重复裁图不得跨集合。需要保留字体未知、低清晰度和字形不可区分的样本，以单独验证拒识能力。现有合成集适合启动模型训练，实际截图收益仍须由这类独立标签验证。
