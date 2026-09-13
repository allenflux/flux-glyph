package org.fluxglyph.androidcapture;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.fonts.Font;
import android.graphics.fonts.FontFamily;
import android.graphics.fonts.FontVariationAxis;
import android.graphics.text.PositionedGlyphs;
import android.graphics.text.TextRunShaper;
import android.os.Build;
import android.os.Bundle;
import android.view.Choreographer;
import android.view.View;
import android.view.WindowManager;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;

/** Native Android shaper and Canvas; the host captures the real framebuffer. */
public final class CaptureActivity extends Activity {
    private JSONObject scenes;
    private final Map<String, JSONObject> registry = new HashMap<>();
    private final Map<String, Typeface> typefaces = new HashMap<>();
    private final Map<Font, JSONObject> proofs = new HashMap<>();
    private int pageIndex;
    private String token;
    private SceneView view;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setDecorFitsSystemWindows(false);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_FULLSCREEN |
            View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_LAYOUT_STABLE |
            View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION);
        try {
            try (InputStream input = getAssets().open("Scenes.json")) {
                scenes = new JSONObject(new String(input.readAllBytes(), StandardCharsets.UTF_8));
            }
            if (!"flux-glyph-android-scenes-v1".equals(scenes.getString("schema"))) throw new IllegalArgumentException("scene schema");
            JSONArray fonts = scenes.getJSONArray("fonts");
            for (int i = 0; i < fonts.length(); i++) {
                JSONObject font = fonts.getJSONObject(i);
                registry.put(font.getString("id"), font);
            }
            view = new SceneView();
            setContentView(view);
            select(getIntent());
        } catch (Exception error) { writeError(error); }
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent); setIntent(intent); select(intent);
    }
    private void select(Intent intent) {
        pageIndex = intent.getIntExtra("page_index", 0);
        token = intent.getStringExtra("capture_token");
        if (token == null) token = "interactive";
        if (view != null) { view.readyScheduled = false; view.invalidate(); }
    }
    private void write(JSONObject value) throws Exception {
        File temporary = new File(getFilesDir(), "ready.tmp");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(value.toString().getBytes(StandardCharsets.UTF_8));
        }
        if (!temporary.renameTo(new File(getFilesDir(), "ready.json"))) throw new IllegalStateException("atomic readiness write failed");
    }
    private void writeError(Exception error) {
        try { write(new JSONObject().put("capture_token", token).put("page_index", pageIndex)
            .put("error", error.getClass().getSimpleName() + ": " + error.getMessage())); }
        catch (Exception ignored) { }
    }
    private Typeface face(JSONObject source) throws Exception {
        String id = source.getString("id");
        if (!typefaces.containsKey(id)) {
            Font.Builder builder = new Font.Builder(getAssets(), source.getString("asset_path"));
            builder.setTtcIndex(source.optInt("ttc_index", 0));
            JSONObject axes = source.optJSONObject("axes");
            if (axes != null && axes.length() != 0) {
                StringBuilder setting = new StringBuilder();
                for (java.util.Iterator<String> keys = axes.keys(); keys.hasNext();) {
                    String key = keys.next();
                    if (setting.length() != 0) setting.append(", ");
                    setting.append("'").append(key).append("' ").append(axes.getDouble(key));
                }
                builder.setFontVariationSettings(setting.toString());
            }
            Font font = builder.build();
            if (!verified(source, proof(font))) throw new IllegalArgumentException("loaded font bytes differ: " + id);
            FontFamily family = new FontFamily.Builder(font).build();
            // Android may still attempt system fallback. Every actual shaped
            // Font is audited below, so fallback can never enter training labels.
            typefaces.put(id, new Typeface.CustomFallbackBuilder(family).setStyle(font.getStyle()).build());
        }
        return typefaces.get(id);
    }
    private static String digest(ByteBuffer buffer) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        ByteBuffer copy = buffer.duplicate(); copy.position(0); digest.update(copy);
        StringBuilder result = new StringBuilder();
        for (byte value : digest.digest()) result.append(String.format("%02x", value & 255));
        return result.toString();
    }
    private static String postscript(ByteBuffer source, int ttcIndex) {
        try {
            ByteBuffer bytes = source.duplicate().order(ByteOrder.BIG_ENDIAN);
            int offset = 0;
            if (bytes.getInt(0) == 0x74746366) {
                int count = bytes.getInt(8);
                if (ttcIndex < 0 || ttcIndex >= count || count > 128) return null;
                offset = bytes.getInt(12 + 4 * ttcIndex);
            } else if (ttcIndex != 0) return null;
            int tables = Short.toUnsignedInt(bytes.getShort(offset + 4));
            if (tables > 512) return null;
            for (int table = 0; table < tables; table++) {
                int entry = offset + 12 + 16 * table;
                if (bytes.getInt(entry) != 0x6e616d65) continue;
                int base = bytes.getInt(entry + 8);
                int count = Short.toUnsignedInt(bytes.getShort(base + 2));
                int strings = base + Short.toUnsignedInt(bytes.getShort(base + 4));
                String fallback = null;
                for (int i = 0; i < count && i < 4096; i++) {
                    int record = base + 6 + 12 * i;
                    int platform = Short.toUnsignedInt(bytes.getShort(record));
                    int language = Short.toUnsignedInt(bytes.getShort(record + 4));
                    int nameId = Short.toUnsignedInt(bytes.getShort(record + 6));
                    int length = Short.toUnsignedInt(bytes.getShort(record + 8));
                    int start = strings + Short.toUnsignedInt(bytes.getShort(record + 10));
                    if (nameId != 6 || length == 0 || length > 1024 || start < 0 || start + length > bytes.limit()) continue;
                    byte[] value = new byte[length]; ByteBuffer copy = bytes.duplicate(); copy.position(start); copy.get(value);
                    String name = new String(value, platform == 0 || platform == 3 ? StandardCharsets.UTF_16BE : StandardCharsets.ISO_8859_1);
                    if (platform == 3 && language == 0x409) return name;
                    if (platform == 0 || fallback == null) fallback = name;
                }
                return fallback;
            }
        } catch (RuntimeException ignored) { }
        return null;
    }
    private JSONObject proof(Font font) throws Exception {
        if (!proofs.containsKey(font)) {
            JSONArray axes = new JSONArray();
            if (font.getAxes() != null) for (FontVariationAxis axis : font.getAxes()) {
                axes.put(new JSONObject().put("tag", axis.getTag()).put("value", axis.getStyleValue()));
            }
            proofs.put(font, new JSONObject().put("sha256", digest(font.getBuffer()))
                .put("postscript", postscript(font.getBuffer(), font.getTtcIndex())).put("ttc_index", font.getTtcIndex())
                .put("weight", font.getStyle().getWeight()).put("slant", font.getStyle().getSlant()).put("axes", axes));
        }
        return proofs.get(font);
    }
    private boolean verified(JSONObject source, JSONObject actual) throws Exception {
        return source.getString("sha256").equals(actual.getString("sha256"))
            && source.getString("postscript").equals(actual.optString("postscript", ""))
            && source.optInt("ttc_index", 0) == actual.getInt("ttc_index");
    }
    private static JSONArray box(float left, float top, float right, float bottom) {
        return new JSONArray().put((int)Math.floor(left)).put((int)Math.floor(top)).put((int)Math.ceil(right)).put((int)Math.ceil(bottom));
    }
    private final class SceneView extends View {
        private boolean readyScheduled;
        SceneView() { super(CaptureActivity.this); }
        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            try {
                JSONObject page = scenes.getJSONArray("pages").getJSONObject(pageIndex);
                JSONArray size = scenes.getJSONArray("canvas_px");
                int[] origin = new int[2]; getLocationOnScreen(origin);
                if (getWidth() != size.getInt(0) || getHeight() != size.getInt(1) || origin[0] != 0 || origin[1] != 0) {
                    throw new IllegalStateException("framebuffer must equal exact scene pixels: " + getWidth() + "x" + getHeight());
                }
                canvas.drawColor(Color.parseColor(page.getString("background")));
                JSONObject captured = new JSONObject().put("schema", "flux-glyph-android-render-proof-v1")
                    .put("capture_token", token).put("page_index", pageIndex).put("page_id", page.getString("id"))
                    .put("split", page.getString("split")).put("canvas_px", size).put("build_fingerprint", Build.FINGERPRINT)
                    .put("sdk_int", Build.VERSION.SDK_INT).put("binding_method", "single asset FontFamily; every TextRunShaper glyph Font SHA256/TTC/PostScript verified; Canvas.drawGlyphs");
                JSONArray rendered = new JSONArray();
                JSONArray regions = page.getJSONArray("regions");
                for (int index = 0; index < regions.length(); index++) {
                    JSONObject source = regions.getJSONObject(index), requested = registry.get(source.getString("font_id"));
                    if (requested == null) throw new IllegalArgumentException("unknown font_id");
                    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.SUBPIXEL_TEXT_FLAG);
                    paint.setTypeface(face(requested)); paint.setTextSize((float)source.getDouble("font_size_px"));
                    paint.setColor(Color.parseColor(source.getString("color")));
                    JSONArray rectangle = source.getJSONArray("bbox");
                    float left = (float)rectangle.getDouble(0), top = (float)rectangle.getDouble(1);
                    float right = (float)rectangle.getDouble(2), bottom = (float)rectangle.getDouble(3);
                    String text = source.getString("text");
                    PositionedGlyphs shaped = TextRunShaper.shapeTextRun(text, 0, text.length(), 0, text.length(), 0, 0, false, paint);
                    // Actual native glyph bounds account for negative side
                    // bearings and ink above the font ascent. Translate the
                    // complete shaped line into its declared box; never resize
                    // glyphs or clip overhanging Latin/handwritten strokes.
                    RectF shapedInk = new RectF(); boolean shapedHasInk = false;
                    for (int i = 0; i < shaped.glyphCount(); i++) {
                        RectF bounds = new RectF();
                        shaped.getFont(i).getGlyphBounds(shaped.getGlyphId(i), paint, bounds);
                        bounds.offset(shaped.getGlyphX(i), shaped.getGlyphY(i));
                        if (bounds.width() > 0 && bounds.height() > 0) {
                            if (shapedHasInk) shapedInk.union(bounds); else { shapedInk.set(bounds); shapedHasInk = true; }
                        }
                    }
                    float textOriginX = left + 12 - shapedInk.left, baseline = top + 12 - shapedInk.top;
                    boolean valid = shaped.glyphCount() > 0, hasInk = false;
                    RectF union = new RectF(); JSONArray glyphs = new JSONArray();
                    for (int i = 0; i < shaped.glyphCount(); i++) {
                        Font font = shaped.getFont(i); JSONObject actual = proof(font); int glyphId = shaped.getGlyphId(i);
                        float x = textOriginX + shaped.getGlyphX(i), y = baseline + shaped.getGlyphY(i);
                        RectF bounds = new RectF(); font.getGlyphBounds(glyphId, paint, bounds); bounds.offset(x, y);
                        boolean matches = verified(requested, actual) && glyphId != 0;
                        valid &= matches;
                        if (bounds.width() > 0 && bounds.height() > 0) {
                            valid &= bounds.left >= left - 1 && bounds.top >= top - 1 && bounds.right <= right + 1 && bounds.bottom <= bottom + 1;
                            if (hasInk) union.union(bounds); else { union.set(bounds); hasInk = true; }
                        }
                        canvas.drawGlyphs(new int[]{glyphId}, 0, new float[]{x, y}, 0, 1, font, paint);
                        glyphs.put(new JSONObject().put("glyph_id", glyphId).put("font", actual).put("font_verified", matches)
                            .put("position", new JSONArray().put(x).put(y)).put("bbox", box(bounds.left,bounds.top,bounds.right,bounds.bottom)));
                    }
                    rendered.put(new JSONObject().put("id", source.getString("id")).put("font_id", source.getString("font_id"))
                        .put("font_verified", valid && hasInk).put("glyphs", glyphs).put("bbox", rectangle)
                        .put("ink_bbox", box(union.left,union.top,union.right,union.bottom))
                        .put("font_size_px", paint.getTextSize()).put("color", String.format("#%06X", paint.getColor() & 0xffffff))
                        .put("text", text).put("typeface_weight", paint.getTypeface().getWeight()).put("advance_px", shaped.getAdvance()));
                }
                captured.put("regions", rendered);
                if (!readyScheduled) {
                    readyScheduled = true; final String capturedToken = token;
                    Choreographer.getInstance().postFrameCallback(first -> Choreographer.getInstance().postFrameCallback(second -> {
                        if (!capturedToken.equals(token)) return;
                        try { write(captured); } catch (Exception error) { writeError(error); }
                    }));
                }
            } catch (Exception error) { writeError(error); }
        }
    }
}
