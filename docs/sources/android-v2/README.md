# 安卓字体第二轮来源

第二轮保留第一轮失败测试记录，另建字体登记表与数据分区。
[FONTS.json](FONTS.json) 对应本地
`artifacts/android-font-v1/sources-v2/FONTS.json`，该原始冻结文件 SHA-256：
`38e8afe2a15a8e9d9c8a625e827cf9d45329833c22b0651b17606a0b7b5109e5`。
公开副本将本机绝对路径改为仓库相对路径，其 SHA 为
`d4ad6ea266ece578c83b499a393bd1d46a75d5b18ee354004693f34a54c8d1b3`；字体 SHA 和来源事实不变。
第一轮 `sources/FONTS.json` 及其字体、许可证、采集和测试文件未修改。

九个已知类别及其 26 个 face 复用第一轮已验证的原文件。
新登记表共 63 个静态 face，其余 37 个用于未知类训练或留出验证。
未知类仍保留真实字体家族名称，仅训练目标映射为 `__unknown__`。

| 分区 | 真实未知字体家族 | Face 与覆盖范围 |
|---|---|---|
| 训练 | Long Cang、Zhi Mang Xing、FandolHei、FandolSong、FandolKai、Liu Jian Mao Cao、Smiley Sans | 复用原 9 个 face，均覆盖 GB2312 汉字 6,763 个 |
| 训练 | Lato | Light / Regular / Medium / Bold，仅英数 |
| 训练 | Open Sans | 宽度 75/100 × 字重 300/400/500/700，8 个静态实例，仅英数 |
| 校准 | M PLUS 1p | Light / Regular / Medium / Bold，各覆盖 GB2312 汉字 2,849 个 |
| 校准 | Yusei Magic | Regular，覆盖 GB2312 汉字 3,378 个 |
| 校准 | Ubuntu | Light / Regular / Medium / Bold，仅英数 |
| 测试 | IBM Plex Sans SC | Light / Regular / Medium / Bold，各覆盖 GB2312 汉字全部 6,763 个 |
| 测试 | Yuji Syuku | Regular，覆盖 GB2312 汉字 3,396 个 |
| 测试 | Liberation Sans | Regular / Bold，仅英数 |

Smiley Sans 已用于第一轮测试，Liu Jian Mao Cao 已用于第一轮校准；两者在本轮
改为训练未知类，不能再称为训练未见的验证字体。登记表绑定第一轮失败 TEST
报告 SHA，并记录各旧字体的分区历史。新校准和新测试的完整家族及字体文件互不
重叠，也不在本轮训练分区中。这一核验针对本地安卓数据来源，不代替初始化模型
全部预训练历史的审计。

## 固定来源及许可证

- **IBM Plex Sans SC**：直接下载 [IBM 官方固定版本](https://github.com/IBM/plex/tree/222657a016f31e6d913ede88520de7dec84dc81b/packages/plex-sans-sc)
  的 hinted OTF，release tag `@ibm/plex-sans-sc@1.1.0`，字体内版本 1.000，OFL 1.1。
  字体内真实 PostScript 名称包括 `IBMPlexSansSC` 和 `IBMPlexSansSC-Medm`，
  不按文件名猜测名称。它是独立的规整无衬线简中测试来源。
- **M PLUS 1p**：[Google Fonts 固定分发版本](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/mplus1p)，
  源于 Coji Morishita 的 M+ Outline Fonts Project。分发说明明确是基于 1.061g
  经 Google Fonts 小幅修改的版本，OFL 1.1；不称为未经修改的上游二进制文件。
  此家族作为校准中的规整 CJK 无衬线来源，与 IBM 测试家族分开。
- **Yusei Magic**：[tanukifont 上游固定 commit](https://github.com/tanukifont/YuseiMagic/tree/d622b0ea84059a411a1344bd2f948d9e0df92a1b)，
  字体内版本 1.200，OFL 1.1，基于记号笔手写设计。
- **Yuji Syuku**：[Google Fonts 固定分发版本](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ofl/yujisyuku)，
  字体内版本 3.002，OFL 1.1；正确上游为 [Kinutafontfactory/Yuji](https://github.com/Kinutafontfactory/Yuji)，
  将书法家 Yuji Kataoka 的手写字形制作成字体。只选择 Syuku，未将共享汉字设计的
  Yuji Mai / Boku 拆成额外独立家族。
- **Lato / Open Sans**：同一 [Google Fonts 固定 commit](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8)，
  OFL 1.1。Lato 字体版本 2.015，原静态 TTF；Open Sans 原变量字体版本 3.003，
  原文件完整保留。八个采集实例使用 fontTools 轴插值，不拉伸截图或施加伪加粗；
  源文件 SHA、轴值、工具版本、独立名称 ID 3/4/6 及产物 SHA 均保存。
- **Ubuntu**：[Google Fonts 固定分发版本](https://github.com/google/fonts/tree/809e4d8b8d7e9364a914909bb777679606c178b8/ufl/ubuntu)，
  字体版本 0.83，许可证为 **Ubuntu Font Licence 1.0**，不是 OFL。
- **Liberation Sans**：[Debian 官方发行包](https://deb.debian.org/debian/pool/main/f/fonts-liberation/)
  `fonts-liberation_2.1.5-3_all.deb` 内原样提取的两个 TTF，OFL 1.1。
  同时保留上游 source archive、`.dsc`、原 LICENSE 及 Debian 包 control/copyright；
  上游归档 SHA 和长度与 `.dsc` 一致。明确这是 Debian 编译分发字体，不称为上游
  预编译二进制文件。登记表保存归档和成员 SHA。

## 字形和来源边界

M PLUS 1p、Yusei Magic、Yuji Syuku 是含汉字的日文字体，并非完整简体中文字库。
场景生成只可使用实际 cmap 的交集；Latin-only 字体不接收汉字。全部 63 个 face
覆盖 ASCII 大写字母、小写字母与数字。cmap 覆盖不等于字体识别准确率，也不保证
日文字形符合中文地区规范。

Klee One 被排除：已知字体 LXGW WenKai 基于 Klee，不能制造已知／未知矛盾标签。
同样未选择可能与文泉驿共享字形的 Droid Sans Fallback，或与文楷同家族的
WenKai Mono。Noto 与思源继续保持第一轮的视觉组规则，不重复计类。

本轮来源核验没有渲染新测试图片、运行测试推理或上传用户截图。
原生采集仍须逐字核实实际 Font SHA、TTC index、PostScript 和字形覆盖。
本地 `DOWNLOADS.json`、`DOWNLOADS_ADDITIONAL.json`、`STATIC_DERIVATIONS.json`
及 `liberationsans/EXTRACTED.json` 保留下载、派生与提取证据；原始许可证另存本目录
`licenses/`。下载记录中保留一次 macOS 文件名大小写碰撞：IBM 冗余 package LICENSE
随后以独立名称重新下载；实际字体目录的 `license.txt` 始终完整。
