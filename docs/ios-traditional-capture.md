# iOS 繁体字体覆盖与原生采集

## 原有覆盖缺口

旧场景 `artifacts/mobile-font-capture/ios-capture-v1/Scenes.json` 有 7,016 个中文区域、194 个不同中文字符。词库是简体写法；常见繁体字“帳、單、詳、餘、轉、網、絡、醫、療、關、閉、語、設”均未出现。旧区域网络的八个分类名称只包含 PingFang SC，没有 TC/HK，也没有“PingFang 族”的输出。这既有文字形状覆盖不足，也有输出标签粒度的问题。

字体网络依然只接收区域图像。这里的中文文本、language、orthography 都是受控采集时的渲染请求及离线审计信息，不是识字结果，不进入推理特征。

## 本地原生证据

当前可用环境：Xcode `/Applications/Xcode.app/Contents/Developer`，已启动 iPhone 17 Pro 模拟器，iOS 26.5，UDID `CC398960-5392-4E41-9672-DA2AB6641BFF`。保存的 UIFont inventory 列出 PingFang SC、TC、HK 各六种字重；本次采集选 Light、Regular、Medium、Semibold 四种。

采集 App 已扩展严格 SC/TC/HK 名称核验：实际 PostScript 前缀与实际 CTFont family 必须同时匹配；每个 CTRun 的 PostScript、family、字形覆盖和可选 shaping language 也要匹配。TC/HK 不会改名成 SC。`CTFontGetGlyphsForCharacters` 失败、零 glyph、字体回退或裁切都会拒绝。

已实际编译、安装 App，并用 `simctl screenshot` 采集 12 页诊断图，共 144 个区域。每个原生 family 各 48 个区域，覆盖四字重、六组相同文本、`zh-Hant` 和 `zh-Hant-HK` 两种明确语言请求；全部原生验证通过，没有回退、缺字或裁切。

| 同文字、字重、字号和语言的比较 | 比较数 | 原始区域像素完全相同 | 归一化网络输入完全相同 |
|---|---:|---:|---:|
| PingFang TC / HK | 48 | 48 | 48 |
| PingFang SC / TC | 48 | 36 | 36 |
| PingFang SC / HK | 48 | 36 | 36 |

对照包含“賬單詳情餘額轉帳”“臺灣香港網絡設定”“骨直令雨言衣辶示”等文本，不只是共同的简单字符。其余 SC/TC 差异也不能据此推断任意外部截图的地域字体可辨识性。

因此，新区域模型以 **PingFang 族**为字体输出，原生 SC/TC/HK 名称继续保留为采集真值。相同像素不能强制训练成三个互斥类别。以后若做细分，需要另建能证明字形区分度的数据与拒绝规则；不能凭文字是繁体就认定 TC/HK，也不能把它们直接显示为 SC。

完整诊断证据位于 `artifacts/mobile-font-capture/ios-traditional-probe/PIXEL_COMPARISON.json`，包含原图/场景 SHA、每区实际字体和所有成对像素比较。该诊断集标记 `training_eligible:false`，不作为训练、校准或正式测试样本。

## 新数据场景

新采集固定为 1,000 页、31 种原生/资产字体样式，seed `2026091291`、namespace `ios-hant-v1`。场景 SHA-256：`102c1592fa98c9006b4bc42f145ca205f8c6dd59a2f907804d5878519aeff74c`。

| 分区 | 页 | 简体区域 | 繁体区域 | 英文区域 | 数字区域 |
|---|---:|---:|---:|---:|---:|
| train | 800 | 2,804 | 2,740 | 1,600 | 1,600 |
| calibration | 100 | 351 | 342 | 200 | 200 |
| test | 100 | 343 | 361 | 200 | 200 |

每页仍有两个英文区域和两个数字区域。SC、TC、HK 每个 family 在每个分区都有简体与繁体内容；文字、字体、字号、颜色独立分配。每种字体不能靠特定文字类型取得标签。八个最终字体家族仍包括 HarmonyOS Sans SC、MiSans、Noto Sans CJK SC、OPPO Sans、PingFang、SF Pro、Helvetica、Alipay Number。

**Alipay Number 是明确加载的外部字体资产，不是 iOS 自带字体。** 本次保留该类，因为仍需识别应用数字字体。它从本地 `AlipayNumber-Regular.ttf` 加载，源 SHA 为 `6074082d8cb92e175184b177e28335e0171f3ddfb2b5818d7853295f7fb0fada`；manifest 的 kind 为 asset，必须完成源文件 SHA、注册 PostScript、实际字体 URL 及 glyph 来源核验。PingFang、系统 SF Pro、Helvetica 的 kind 为 system，不能混为同一种来源。

新旧所有规范化文字组合交集为零，source/page/content namespace 也分离。第二批显式采用更大的短英文词库，否则原来有限的三词组合不足以容纳两批各 2,000 个互不重复英文区域。新 test 应作为本轮冻结后评估，旧 100 页只作为旧域回归；不得根据新 test 调整权重、门槛或采集分布。

## 可复现命令

```bash
.venv-dev/bin/python training/capture/generate_scenes.py \
  --output artifacts/mobile-font-capture/ios-hant-scenes-v1 \
  --platform ios --pages 1000 --seed 2026091291 --namespace ios-hant-v1 \
  --han-orthography mixed --extended-english --include-pingfang-traditional \
  --ios-font-inventory artifacts/mobile-font-capture/ios-traditional-probe/font-inventory.json \
  --exclude-scenes artifacts/mobile-font-capture/ios-capture-v1/Scenes.json

.venv-dev/bin/python training/capture/capture_ios.py \
  --simulator CC398960-5392-4E41-9672-DA2AB6641BFF \
  --scenes artifacts/mobile-font-capture/ios-hant-scenes-v1/ios-scenes.json \
  --assets artifacts/mobile-font-capture/ios-hant-scenes-v1/assets \
  --output artifacts/mobile-font-capture/ios-hant-capture-v1
```

采集 app 需先编译并安装。相同场景和模拟器的中断续跑可加 `--resume`；不要往旧采集目录写入不同场景。新捕获注册表函数是 `training.capture.generate_scenes.capture_families()`，保持旧 15 类次序，末尾追加原生 TC/HK。训练输出分组另外定义，不能将分组名误当成原生可请求字体。

采集 app、驱动、场景文件及实际 PNG 的 SHA 写入 capture protocol/labels。新 1,000 页仅构成受控原生 iOS 模拟器截图；第三方应用和物理手机准确率仍需要独立验证。

本次 1,000 页已实际采集完成。全量 source 审计逐张验证 PNG SHA、解码像素身份、原生 frame 与 labels 一致性，并核验 10,941 个区域的 CTFont/CTRun、字形覆盖、字体资产和字号真值，全部通过、零拒绝。各分区七类来源/像素/文本身份交集均为零；新旧文本使用 **NFKC + casefold + 去空白** 比较，交集为零。

可重新执行完整核验：

```bash
.venv-dev/bin/python scripts/audit_native_capture.py \
  --captures artifacts/mobile-font-capture/ios-hant-capture-v1 \
  --exclude-scenes artifacts/mobile-font-capture/ios-capture-v1/Scenes.json
```

审计结果：[ios-traditional-source-audit.json](ios-traditional-source-audit.json)。实际 labels SHA 为 `454aae37f4c6e427de6634d1a49bf7365f84a8ba6790aeb2263af903abde5395`，capture protocol SHA 为 `ae2f4989f8c96178f68ae2c2158e66ae92aabb2010152e03fbe84f4ad8679db6`。这些是来源与标注核验结果，没有执行字体模型预测，也没有提前计算新测试集指标。
