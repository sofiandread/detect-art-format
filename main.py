# main.py
from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os
import io

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- helpers --------------------

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
    Build a fitz.Rect from coordsOrigin=pdf + x,y,width,height;
    if missing, default to bottom-half.
    """
    if not data or data.get("coordsOrigin") != "pdf":
        return bottom_half_rect(page)
    try:
        x = float(data["x"]); y = float(data["y"])
        w = float(data["width"]); h = float(data["height"])
        pr = page.rect
        rx0 = max(pr.x0, x); ry0 = max(pr.y0, y)
        rx1 = min(pr.x1, x + w); ry1 = min(pr.y1, y + h)
        return fitz.Rect(rx0, ry0, rx1, ry1)
    except Exception:
        return bottom_half_rect(page)

def count_vector_segments_in(page, clip):
    """
    Fallback signal for pure-vector pages:
    count vector path ops intersecting the clip, ignoring hairline/template crumbs
    and ignoring rectangle ops (background panels).
    """
    drawings = page.get_drawings()
    segs = 0
    VECTOR_OPS = {"m", "l", "c", "v", "y", "h"}  # intentionally exclude 're' (rectangles)
    for d in drawings:
        # skip hairlines
        if float(d.get("width") or 0) <= 0.25:
            continue
        # quick reject if bbox outside clip
        if "rect" in d and not fitz.Rect(d["rect"]).intersects(clip):
            continue
        for it in d.get("items", []):
            op = str(it[0]).lower() if it else ""
            if op in VECTOR_OPS:
                segs += 1
    return segs

def sum_raster_coverage_in_clip(page, clip):
    """
    Sum (approximately) the raster image coverage within the clip.
    Returns (sum_area, coverage_ratio 0..1).
    Note: We sum image intersections; overlap is rare in proofs and the total
    is clamped to the clip area.
    """
    clip_area = max(1.0, clip.get_area())
    total = 0.0
    for img in page.get_images(full=True):
        try:
            # PyMuPDF accepts xref or tuple depending on version
            bbox = page.get_image_bbox(img)
        except Exception:
            xref = img[0] if isinstance(img, (tuple, list)) else img
            if not xref:
                continue
            bbox = page.get_image_bbox(xref)
        inter = fitz.Rect(bbox).intersect(clip)
        total += inter.get_area()
    cov = min(1.0, total / clip_area)
    return total, cov

def text_coverage_in_clip(page, clip):
    """
    Sum area of TEXT blocks only (block_type == 0) inside the clip.
    Returns ratio 0..1.
    """
    clip_area = max(1.0, clip.get_area())
    total = 0.0
    try:
        # blocks: (x0, y0, x1, y1, "text", block_no, block_type, ...)
        for b in page.get_text("blocks") or []:
            # Only text blocks (block_type==0). Some PyMuPDF versions place type in b[6].
            block_type = None
            if len(b) >= 7:
                block_type = b[6]
            if block_type not in (0, None) and block_type != 0:
                continue  # skip non-text (images, etc.)
            r = fitz.Rect(b[:4])
            total += r.intersect(clip).get_area()
    except Exception:
        pass
    return min(1.0, total / clip_area)

def drawings_coverage_in_clip(page, clip):
    """
    Approximate vector-shape coverage from page.get_drawings().
    Big filled panels (background rectangles or path-filled boxes) are heavily
    down-weighted so they don't dominate. Returns ratio 0..1.
    """
    drawings = page.get_drawings()
    clip_area = max(1.0, clip.get_area())
    weighted_area = 0.0

    for d in drawings:
        width = float(d.get("width") or 0)
        if width <= 0.25:  # hairlines / template crumbs
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

        # Heuristic: "big filled shape with little/no stroke" → treat as background panel
        is_filled = d.get("fill") is not None
        has_stroke = width > 0.25
        small_path = len(items) <= 5

        if (is_rect_op and frac >= 0.15) or (is_filled and not has_stroke and small_path and frac >= 0.15):
            weight = 0.05      # panels: barely count
        elif is_rect_op:
            weight = 0.15      # smaller rectangles
        else:
            weight = 0.30      # generic vector shapes

        weighted_area += inter_area * weight

    return min(1.0, weighted_area / clip_area)

def find_largest_image_in_clip(page, clip):
    """
    Return (xref, interRect, bboxRect) for the image covering largest area in clip.
    """
    best = None
    best_area = 0.0
    for img in page.get_images(full=True):
        try:
            bbox = page.get_image_bbox(img)
            xref = img[0] if isinstance(img, (tuple, list)) else None
        except Exception:
            xref = img[0] if isinstance(img, (tuple, list)) else None
            if not xref:
                continue
            bbox = page.get_image_bbox(xref)
        rb = fitz.Rect(bbox)
        inter = rb.intersect(clip)
        a = inter.get_area()
        if a > best_area:
            best = (xref, inter, rb)
            best_area = a
    return best  # (xref, interRect, fullBBox) or None

def decide_format_coverage(raster_cov, vector_cov, vector_segments):
    """
    Coverage-majority with raster bias on near ties and safeguard for pure-vector pages.
    """
    # Clear wins
    if raster_cov >= vector_cov + 0.01:   # small margin → raster
        return "has raster"
    if vector_cov >= raster_cov + 0.12:   # vector needs a clearer margin
        return "has vector"

    # If there is a noticeable image and vectors are relatively small → raster
    if raster_cov >= 0.15 and vector_cov <= 0.30:
        return "has raster"

    # Pure-vector safeguard (logos/illustrations with no rasters)
    if raster_cov <= 0.03 and vector_segments >= 80:
        return "has vector"

    # Otherwise default to raster (text shouldn't beat a big photo)
    return "has raster"


# -------------------- routes --------------------

@app.route("/")
def home():
    return "Art Format Detector is live (coverage-first)."

@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400

        file = request.files["pdf"]
        filename = file.filename or "upload.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page_index = safe_int(request.form.get("pageIndex"), 0)
        page_index = max(0, min(page_index, len(doc) - 1))
        page = doc.load_page(page_index)
        clip = rect_from_params(page, request.form)

        # coverage metrics
      text_cov = text_coverage_in_clip(page, clip)
draw_cov = drawings_coverage_in_clip(page, clip)

# ↓ shrink drawings influence so text can't outrank a big photo
vector_cov = min(1.0, text_cov + 0.30 * draw_cov)

# (optional: expose for debugging)
effective_vector_cov = vector_cov


        # backup signal for true-vector pages
        vector_segments = count_vector_segments_in(page, clip)

        # native raster (largest in clip)
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

        fmt = decide_format_coverage(raster_cov, vector_cov, vector_segments)

        return jsonify({
            "format": fmt,  # 'has raster' or 'has vector'
            "metrics": {
                "region": "clip" if request.form.get("coordsOrigin") == "pdf" else "bottom-half",
                "rasterCount": len(page.get_images(full=True)),
                "rasterCoverage": round(raster_cov, 3),
                "vectorCoverage": round(vector_cov, 3),
                "textCoverage": round(text_cov, 3),
                "drawingsCoverage": round(draw_cov, 3),
                "vectorSegments": int(vector_segments),
                **({"nativeRaster": native} if native else {})
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extract-design-image", methods=["POST"])
def extract_design_image():
    """
    Optional helper: returns a PNG of the supplied clip at 300 dpi.
    """
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400

        file = request.files["pdf"]
        filename = file.filename or "upload.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page_index = safe_int(request.form.get("pageIndex"), 0)
        page_index = max(0, min(page_index, len(doc) - 1))
        page = doc.load_page(page_index)
        clip = rect_from_params(page, request.form)

        pix = page.get_pixmap(clip=clip, dpi=300)
        image_path = os.path.join(UPLOAD_FOLDER, f"{filename}_design.png")
        pix.save(image_path)

        return send_file(image_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
