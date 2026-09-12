# Sans font capture data

This dataset strengthens comparison among the eight existing font families and
Roboto. It is a set of controlled **iOS Simulator screenshots**, including
bundled manufacturer fonts. It does not establish Android native rendering,
the phone model, or the true font of any user screenshot.

The plan has 300 training, 60 calibration and 60 new test pages, with 12 regions
per page. Five Chinese families share the same Chinese content and styling.
Seven Latin/numeric families share numeric content, including negative amounts,
decimals, dates and times. English pages keep Alipay Number on a separate numeric
line because that font has no alphabet. Rows are shuffled to avoid fixed family
positions. All text is generated and isolated from every historical scene split;
historical screenshot pixels are not opened for new training.

The 46 requested faces include PingFang SC/TC/HK, SF Pro, Helvetica, HarmonyOS
Sans SC, MiSans, Noto Sans CJK SC, OPPO Sans, Alipay Number and Roboto. New source
coverage includes HarmonyOS Light/Thin, MiSans Light/Medium and Noto Light.
The existing official OPPO Sans 3 package does not contain a Light face, so none
is synthesized or mislabeled as an original light face.

Roboto uses ten explicit variable-font instances: widths 75/100 and weights
300/400/500/700/900. `fontTools.varLib.instantiateVariableFont` instantiates the
original axes, then name IDs 3/4/6 receive unique names to avoid registration
collisions. The actual glyph shapes are variable-font instances, not artificially
thinned or stretched raster images. Source, derived file and tool version hashes
are recorded in [the font source report](sources/sans/SOURCE_FACES.json).

All requested faces first passed a 14-page, 168-region training-only smoke
capture. The existing native collector verifies actual CTFont names, every
glyph/run, coverage, clipping, registration, the actual font URL and file SHA.
The full preparation rejects failed native proof, validates asset hashes and
content separation, and records native pixel font size/color. All **5,040**
regions passed native proof. The existing pixel preprocessor excluded two
14-point HarmonyOS Thin amounts as low quality (one training, one calibration).
The pixel threshold and captured test pages were unchanged. Usable region
counts are **3,599 / 719 / 720**; the full test split is retained. The wrapper
records each excluded identity and rejects unexpectedly large preprocessing
losses above 0.5% in a split. Native/font/source failures remain fatal.
See [the preparation audit](sources/sans/DATA_AUDIT.json).
Only region image tiles enter the neural model; text and script annotations are
used for source auditing.

The new scripts are `training/capture/generate_sans_scenes.py` and
`training/capture/prepare_sans_regions.py`. They use the existing
`capture_ios.py` and `train_regions.prepare(..., group_pingfang=True,
test_history='fresh')`. The existing frozen data preparation and renderer code
are unchanged. The local artifact root is `artifacts/font-sans-v1`.

Source downloads come from the existing official HarmonyOS archive, the
[MiSans official download](https://hyperos.mi.com/font/zh/download/),
[Noto CJK upstream](https://github.com/notofonts/noto-cjk/tree/f8d157532fbfaeda587e826d4cd5b21a49186f7c),
and the previously verified Google Fonts Roboto snapshot. Font files stay local;
the repository records provenance, capture code and resulting model artifacts.
