# main.py  —  Flask + PyMuPDF (coverage-first, vector-safe)
# - Robust helpers for raster / vector / text coverage inside a clip
# - Counts FILLED vector paths (most logos) and ignores giant mock panels
# - Mixed-page friendly decision with gentle raster bias + vector tie-breaks

from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ───────── helpers ─────────

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def bottom_half_rect(page):
    r = page.rect
    return fitz.Rect(r.x0, r.y0 + r.height / 2.0, r.x1, r.y1)

def rect_from_params(page, data):
    """
    If form-data contains coords in PDF space (points) with coordsOrigin=pdf,
    crop to that rect. Otherwise default to the bottom half of the page.
    """
    if not data or (data.get("coordsOrigin") or "").lower() != "pdf":
        return bottom_half_rect(page)
    try:
        x = float(data["x"]); y = float(data["y"])
        w = float(data["width"]); h = float(data["height"])
        pr = page.rect
        return fitz.Rect(max(pr.x0, x), max(pr.y0, y),
                         min(pr.x1, x + w), min(pr.y1, y + h))
    except Exception:
        return bottom_half_rect(page)

def count_vector_segments_in(page, clip):
    """
    Count meaningful vector segments in the clip.
    - Count FILLED paths even if stroke width is tiny/0 (very common in logos).
    - Allow 're' rectangles when filled (logos may use them).
    - Ignore giant filled panels that cover the mock area.
    """
    drawings = page.get_drawings()
    segs = 0
    for d in drawings:
        rect = fitz.Rect(d["rect"]) if "rect" in d else None
        if rect and not rect.intersects(clip):
            continue

        items = d.get("items", []) or []
        width = float(d.get("width") or 0)
        has_fill = d.get("fill") is not None

        # Ignore huge panels that cover ≥15% of the clip and have few items (likely the dark mock box)
        if rect:
            frac = rect.intersect(clip).get_area() / max(1.0, clip.get_area())
            if frac >= 0.15 and len(items) < 8:
                continue

        for it in items:
            op = (str(it[0]).lower() if it else "")
            # path ops (include 're' if filled)
            if op in {"m", "l", "c", "v", "y", "h", "re"}:
                # If not filled, keep a tiny width floor to skip hairlines/guides
                if not has_fill and width <= 0.25:
                    continue
                segs += 1
    return segs

def sum_raster_coverage_in_clip(page, clip):
    """
    Sum covered area of raster images that intersect the clip.
    Returns: (total_intersect_area, coverage_fraction_0to1)
    """
    clip_area = max(1.0, clip.get_area())
    total = 0.0
    for img in page.get_images(full=True):
        # page.get_images returns tuples; index 0 is the xref
        xref = img[0] if isinstance(img, (tuple, list)) else img
        if not xref:
            continue
        try:
            bbox = page.get_image_bbox(xref)
        except Exception:
            # some PyMuPDF builds also accept the full tuple; try that
            try:
                bbox = page.get_image_bbox(img)
            except Exception:
                continue
        inter = fitz.Rect(bbox).intersect(clip)
        total += inter.get_area()
    return total, min(1.0, total / clip_area)

def text_coverage_in_clip(page, clip):
    """
    Fraction of clip covered by TEXT blocks.
    """
    clip_area = max(1.0, clip.get_area())
    total = 0.0
    try:
        for b in page.get_text("blocks") or []:
            block_type = b[6] if len(b) >= 7 else 0
            if block_type != 0:      # keep only TEXT blocks
                continue
            r = fitz.Rect(b[:4])
            total += r.intersect(clip).get_area()
    except Exception:
        pass
    return min(1.0, total / clip_area)

def drawings_coverage_in_clip(page, clip):
    """
    Down-weight big filled panels/rectangles so they don't dominate.
    Returns a weighted fraction that is intentionally conservative.
    """
    drawings = page.get_drawings()
    clip_area = max(1.0, clip.get_area())
    weighted_area = 0.0
    for d in drawings:
        width = float(d.get("width") or 0)
        if width <= 0.25 and d.get("fill") is None:
            # hairline without fill — usually guides
            continue
        if "rect" not in d:
            continue
        rect = fitz.Rect(d["rect"])
        inter = rect.intersect(clip)
        inter_area = inter.get_area()
        if inter_area <= 0:
            continue
        frac = inter_area / clip_area
        items = d.get("items", [])
        is_rect_op = any((it and str(it[0]).lower() == "re") for it in items)
        is_filled = d.get("fill") is not None
        has_stroke = width > 0.25
        small_path = len(items) <= 5

        # heavily down-weight obvious panels
        if (is_rect_op and frac >= 0.15) or (is_filled and not has_stroke and small_path and frac >= 0.15):
            weight = 0.05
        elif is_rect_op:
            weight = 0.15
        else:
            weight = 0.30
        weighted_area += inter_area * weight
    return min(1.0, weighted_area / clip_area)

def find_largest_image_in_clip(page, clip):
    """
    Return (xref, intersection_rect, native_rect) for the largest raster intersecting the clip.
    """
    best = None
    best_area = 0.0
    for img in page.get_images(full=True):
        xref = img[0] if isinstance(img, (tuple, list)) else img
        if not xref:
            continue
        try:
            bbox = page.get_image_bbox(xref)
        except Exception:
            try:
                bbox = page.get_image_bbox(img)
            except Exception:
                continue
        rb = fitz.Rect(bbox)
        inter = rb.intersect(clip)
        a = inter.get_area()
        if a > best_area:
            best = (xref, inter, rb)
            best_area = a
    return best

def decide_format_coverage(raster_cov, vector_cov, vector_segments, raster_count, text_cov, native_dpi_min=0):
    """
    Coverage-majority with:
      • raster bias on near ties (for mixed pages),
      • vector-only guard: if there are no rasters and any meaningful text/drawings, choose vector,
      • pure-vector safeguard with lower segment threshold,
      • DPI tie-breaker when the raster looks like a flat panel.
    """
    # If there are effectively no rasters, treat any vector coverage as vector.
    if raster_cov <= 0.02 and (vector_cov >= 0.03 or text_cov >= 0.025 or raster_count == 0 or vector_segments >= 20):
        return "has vector"

    # Clear wins in mixed cases
    if raster_cov >= vector_cov + 0.01:
        return "has raster"
    if vector_cov >= raster_cov + 0.12:
        return "has vector"

    # Image present + relatively small vector coverage → raster
    if raster_cov >= 0.15 and vector_cov <= 0.30:
        return "has raster"

    # Pure-vector safeguard (logos with no rasters but many segments)
    if raster_cov <= 0.03 and vector_segments >= 40:
        return "has vector"

    # If there's clear vector structure and the biggest raster in the clip is low-detail, prefer vector.
    if vector_segments >= 18 and native_dpi_min and native_dpi_min < 40 and raster_cov >= 0.10:
        return "has vector"

    # Otherwise: near tie → raster
    return "has raster"

# ───────── routes ─────────

@app.route("/")
def home():
    return "Art Format Detector is live (coverage-first, vector-safe)."

@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400

        file = request.files["pdf"]
        filename = file.filename or "upload.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            doc = fitz.open(filepath)
            page_index = safe_int(request.form.get("pageIndex"), 0)
            page_index = max(0, min(page_index, len(doc) - 1))
            page = doc.load_page(page_index)
            clip = rect_from_params(page, request.form)

            # coverage metrics
            _, raster_cov = sum_raster_coverage_in_clip(page, clip)
            text_cov      = text_coverage_in_clip(page, clip)
            draw_cov      = drawings_coverage_in_clip(page, clip)
            vector_cov    = max(0.0, min(1.0, text_cov + 0.30 * draw_cov))

            # extra signals
            vector_segments = count_vector_segments_in(page, clip)
            raster_count = len(page.get_images(full=True))

            # native raster metrics (largest in clip)
            native = {}
            largest = find_largest_image_in_clip(page, clip)
            if largest:
                xref, inter_rect, _ = largest
                native_px_w = native_px_h = 0
                if xref:
                    try:
                        info = doc.extract_image(xref)
                        native_px_w = int(info.get("width") or 0)
                        native_px_h = int(info.get("height") or 0)
                    except Exception:
                        pass
                placed_w_in = inter_rect.width / 72.0 if inter_rect.width else 0
                placed_h_in = inter_rect.height / 72.0 if inter_rect.height else 0
                dpi_x = (native_px_w / placed_w_in) if placed_w_in > 0 else 0
                dpi_y = (native_px_h / placed_h_in) if placed_h_in > 0 else 0
                dpi_min = min(dpi_x, dpi_y) if dpi_x and dpi_y else 0
                native = {
                    "nativePxW": native_px_w,
                    "nativePxH": native_px_h,
                    "placedWIn": round(placed_w_in, 3),
                    "placedHIn": round(placed_h_in, 3),
                    "nativeDpiX": round(dpi_x, 1),
                    "nativeDpiY": round(dpi_y, 1),
                    "nativeDpiMin": round(dpi_min, 1),
                }

            fmt = decide_format_coverage(
                raster_cov=raster_cov,
                vector_cov=vector_cov,
                vector_segments=vector_segments,
                raster_count=raster_count,
                text_cov=text_cov,
                native_dpi_min=(native.get("nativeDpiMin", 0) or 0),
            )

            return jsonify({
                "format": fmt,
                "metrics": {
                    "region": "clip" if (request.form.get("coordsOrigin") or "").lower() == "pdf" else "bottom-half",
                    "rasterCount": int(raster_count),
                    "rasterCoverage": round(raster_cov, 3),
                    "textCoverage": round(text_cov, 3),
                    "drawingsCoverage": round(draw_cov, 3),
                    "effectiveVectorCoverage": round(vector_cov, 3),
                    "vectorSegments": int(vector_segments),
                    **({"nativeRaster": native} if native else {})
                }
            })
        finally:
            try:
                doc.close()
            except Exception:
                pass
            try:
                os.remove(filepath)
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extract-design-image", methods=["POST"])
def extract_design_image():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400
        file = request.files["pdf"]
        filename = file.filename or "upload.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            doc = fitz.open(filepath)
            page_index = safe_int(request.form.get("pageIndex"), 0)
            page_index = max(0, min(page_index, len(doc) - 1))
            page = doc.load_page(page_index)
            clip = rect_from_params(page, request.form)

            pix = page.get_pixmap(clip=clip, dpi=300)
            image_path = os.path.join(UPLOAD_FOLDER, f"{filename}_design.png")
            pix.save(image_path)
            return send_file(image_path, mimetype="image/png")
        finally:
            try:
                doc.close()
            except Exception:
                pass
            # we intentionally keep the PNG so the client can download it
            try:
                os.remove(filepath)
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
