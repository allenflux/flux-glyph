# 无 OCR 字体族评测

`scripts/evaluate_region_fonts.py` 对完整原始 PNG 运行文字区域检测、整行字体与字号模型，并分别评估字体、字号和颜色。推理只收到 PNG 路径及输出位置，不接收原生文字、简繁类别、字体标签或字形框。运行期间禁止 PPReader、CTC、字符分割及旧单字分类器调用。

评测要求完整预定的 1,000 页采集集，使用其固定 100 页 test。检测框与原生墨迹框只按 IoU ≥0.5 做一对一匹配；先最大化配对数量，再最大化总 IoU。漏检、多余检测、错误字体、检出后待确认分别计数。

## PingFang 评分约定

模型 metadata 只有明确声明下列分组，且输出类别包含 `PingFang`、不再同时包含 `PingFang SC`／`TC`／`HK` 时，才允许按字体族计分：

```json
{"font_label_groups":{"PingFang":["PingFang SC","PingFang TC","PingFang HK"]}}
```

分组必须精确符合上述定义；少成员、多成员、合并其它字体、缺失声明或输出类别重叠都会拒绝。原生字体先通过 CTFont/CTRun 和来源文件审计，才映射到声明的字体族。没有该分组的旧 SC-only 模型识别原生 TC／HK 仍计错误，不能仅按名称前缀合并。

报告保留 `native_family` 和 `native_font_postscript`；`expected_family` 仅在合法分组时变成 `PingFang`。按族正确不代表识别出了 SC、TC、HK 的具体地区子类。报告同时提供按模型标签和原生字体分别统计的正确数、错误数、待确认数与覆盖率。

## 简体、繁体、英文、数字

汉字的 `simplified`／`traditional` 分类来自已绑定 SHA 的 Scenes 原生请求，校验其文字、script、原生字体与语言声明一致，不能根据字体名猜简繁。英文和数字仅根据原生真值文字区分：包含 ASCII 字母的 Latin 区域归英文，仅数字及标点的 Latin 区域归数字。这些字段仅用于报告，不作为模型输入。

`metrics.by_orthography` 总是包含 simplified、traditional、english、numeric 四组；旧采集缺乏简繁声明时另列 han_unspecified，不伪造分类。每组报告接受覆盖率、错误率、待确认数、漏检以及字号／颜色质量。

## 新测试与复用测试

调用必须显式选择测试历史：

```sh
PYTHONPATH=src .venv-dev/bin/python scripts/evaluate_region_fonts.py \
  --models PATH_TO_VERIFIED_BUNDLE \
  --captures PATH_TO_NATIVE_CAPTURE/labels.jsonl \
  --output PATH_TO_NEW_EVALUATION \
  --test-history fresh
```

新采集、此前未评估的固定分区使用 `fresh`；已评估过的分区使用 `reused`。该选项是调用者对测试历史的明确声明，不自动证明其与所有历史数据文本互斥。报告保留父模型已接触同类渲染域的限制：新文本或新截图不能等同于未知设备、未知应用或未知字体的盲测。跨本次 train/calibration/test 的来源、像素和内容隔离仍由原生采集审计验证。

缓存绑定模型清单、显式字体分组、推理源码、评分源码、原图 SHA 与完整评估协议；模型、分组或历史声明变化时不能悄悄复用旧评分。字体族定义和判定门槛都应在测试推理前确定。
