# 安卓模式的开放字体来源

本目录记录独立安卓字体模型的数据来源，不把字体名称当作设备系统的证明。
原始字体和许可证保存在本地 `artifacts/android-font-v1/sources`；
[FONTS.json](FONTS.json) 是字体登记表的公开副本，包含 35 个静态 face 的路径、
SHA-256、实际 PostScript 名称、TTC index、字重、许可证、版本及 cmap 核验。
公开副本只将本机绝对路径改为仓库相对路径，字体文件 SHA 与来源事实不变；原始冻结登记表 SHA 为
`544a8eddcd5278173938aedce5174f5e5399f6e70174747a2730bbf67d4aef99`，公开副本 SHA 为
`706ab40d9a1997e62db98044cc701ba566cf26a1bd71d62e4df45a5268f454be`。
原生 Android 模拟器采集已完成：800 页、9,600 个文字区域，35 个 face 的
54,236 个原生 glyph 均通过实际字体文件 SHA、TTC index 和 PostScript 核验。
数据在 `artifacts/android-font-v1/capture-v2`，采集清单 SHA-256 为
`85cd830b1f45475d19f61c2c8bc6078de2d9bff6965b3e7ab3012eb12c9557c5`，
完整核验记录见 `artifacts/android-font-v1/CAPTURE_AUDIT.json`。
这确认了截图的采集来源和标签。训练和失败验收见[逐轮验证记录](../../android-font-results.md)；
[实验版发布记录](PREVIEW_RELEASE.json) 明确标为未通过稳定验收的候选模型。

## 已知类别

| 模型类别 | 选用 face | 开放来源 |
|---|---|---|
| Noto Sans CJK SC | Light / Regular / Medium / Bold | [Noto CJK 固定版本](https://github.com/notofonts/noto-cjk/tree/f8d157532fbfaeda587e826d4cd5b21a49186f7c)，OFL 1.1 |
| Noto Serif CJK SC | Light / Regular / Medium / Bold | 同一 Noto CJK 版本，OFL 1.1 |
| LXGW WenKai | Light / Regular / Medium | [霞鹜文楷上游](https://github.com/lxgw/LxgwWenKai/tree/50f4b182415a8c33d9a456df220b66a284e2509b)，OFL 1.1 |
| WenQuanYi Micro Hei | 原始 TTC 中的非 Mono face | [Debian 官方镜像中的原始 0.2.0-beta 包](https://deb.debian.org/debian/pool/main/f/fonts-wqy-microhei/)，Apache 2.0 或 GPLv3 加原文嵌入例外 |
| ZCOOL KuaiLe | Regular | [Google Fonts 上游分发](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/zcoolkuaile)，OFL 1.1 |
| ZCOOL XiaoWei | Regular | [Google Fonts 上游分发](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/zcoolxiaowei)，OFL 1.1 |
| ZCOOL QingKe HuangYou | Regular | [Google Fonts 上游分发](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/zcoolqingkehuangyou)，OFL 1.1 |
| Ma Shan Zheng | Regular | [Google Fonts 上游分发](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/mashanzheng)，OFL 1.1 |
| Roboto | 宽度 75/100 × 字重 300/400/500/700/900 | [已验证的 Google Fonts 版本](https://github.com/google/fonts/tree/1c627bfa375fc51cf86fabeca4f6e08a95f0aa5c/ofl/roboto)，OFL 1.1；仅英数 |

Noto Sans 与思源黑体、Noto Serif 与思源宋体分别只设一个视觉类别，避免同源的
重叠中文字形被当作竞争标签。本轮实际取样的是登记表中固定的 Noto 文件；
这些分组不宣称两个项目所有版本、字符、字距或二进制文件完全相同，也不从截图
辨认出具体分发项目。

Roboto 原始变量文件完整保留。用于采集的十个静态 TTF 复用上轮已验证实例，
记录原文件 SHA、`fontTools.varLib.instantiateVariableFont` 的轴坐标、工具版本
及派生文件 SHA；仅名称 ID 3/4/6 使用独立名称以避免注册冲突，未进行栅格拉伸
或把其他字体改名为 Roboto。

文泉驿字体直接取自 Debian 发布的原始上游归档，其 SHA-256 和长度与 `.dsc`
记录一致；没有使用 Debian 修改后的字体文件。许可证采用原包 README 的表达，
不是将它误写为 OFL。原始两个许可证及 README 均保留。
MiSans、HarmonyOS Sans、OPPO Sans 不在这份开放字体登记表内，也未被笼统称为开源。

## 未覆盖字体来源隔离

| 分区 | 实际字体家族 | 来源 |
|---|---|---|
| 训练未知类 | Long Cang、Zhi Mang Xing | Google Fonts 上述固定 commit，分别为 [Long Cang](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/longcang)、[Zhi Mang Xing](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/zhimangxing)，OFL 1.1 |
| 训练未知类 | FandolHei Regular/Bold、FandolSong Regular/Bold、FandolKai Regular | [CTAN Fandol 0.3 原包](https://ctan.org/pkg/fandol)，GPLv3 加原文 font embedding exception |
| 校准未知类 | Liu Jian Mao Cao | [Google Fonts 固定版本](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/liujianmaocao)，OFL 1.1 |
| 测试未知类 | Smiley Sans / 得意黑 | [上游 v2.0.1](https://github.com/atelier-anchor/smiley-sans/releases/tag/v2.0.1)，commit `67e3821f4b06cbd7155fa6fa69daff4b6f311b76`，OFL 1.1 |

这七个真实家族不在九个已知类别中；登记表保留实际 `family`，训练映射为
`__unknown__`。分区在任何截图生成或模型推理前固定，整种未知字体源不会跨越
训练、校准、测试。得意黑是独立的窄斜美术黑体，不是文泉驿、Noto 或霞鹜文楷
的同字形别名。未知类测试通过也只能反映这些留出家族，不能代表所有安卓主题字体。

Fandol 的五个原始 OTF 在首次正式采集和训练前加入，只用于训练未知类，补充黑体、
宋体和楷体。原始归档、README 和 COPYING 完整保存，未把它们称为 OFL 字体。
文件名为 Bold 的两个字体，其原始 OS/2 `usWeightClass` 实际为 400，登记表如实记录，
采集使用文件自身字形，不根据文件名施加伪加粗。旧登记表保存在
`FONTS.before-fandol.json`，扩展的来源和构建脚本 SHA 由新登记表绑定。

## 文件和字形覆盖核验

所有 25 个中文 face（16 个已知、9 个未知）都在其实际 cmap 中覆盖 GB2312 的
**6,763 / 6,763** 个汉字。Roboto 的 10 个 face 只用于英数，汉字覆盖指标不适用。
该检查说明字库有对应字形，不是字体识别准确率，也不代表全部 Unicode 中文覆盖。

每个 face 的完整 cmap 独立保存并绑定 SHA。原生场景生成仍须逐字查覆盖，采集时
还要检查实际字形使用及字体文件，防止系统回退造成错误标签。用户截图未用作字体
名称真值，没有上传用户图像。

本地 `DOWNLOADS.json` 保留 46 个下载文件的固定 URL、重定向地址、长度、SHA 和
检索时间；`EXTRACTED_AND_REUSED_SOURCES.json` 记录归档成员和已有资产的复用来源。
新增 Fandol 原包的下载记录另存 `fandol/DOWNLOAD.json`，归档 SHA-256 为
`9278f01b417ded5766d98c3937192a1a6a2c73a5e94a3493fdfc932b2a55005a`。
本目录的 `licenses/` 保存许可证原件。数据登记表 SHA-256：
`544a8eddcd5278173938aedce5174f5e5399f6e70174747a2730bbf67d4aef99`。
