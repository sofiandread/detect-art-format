# app.py  (Flask + PyMuPDF)
from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- helpers --------------------
def text_coverage_in_clip(page, clip):
    """
    Approximate vector/text coverage by summing text block areas that intersect the clip.
    Works well for typical PDF text (fonts). Returned as ratio 0..1.
    """
    clip_area = max(1.0, clip.get_area())
    total = 0.0
    try:
        for b in page.get_text("blocks") or []:
            # blocks = (x0, y0, x1, y1, text, block_no, block_type, ...)
            r = fitz.Rect(b[:4])
            total += r.intersect(clip).get_area()
    except Exception:
        pass
    return min(1.0, total / clip_area)


def drawings_coverage_in_clip(page, clip):
    """
    Approximate vector shape coverage from page.get_drawings().
    We apply low weights to rectangles (likely backgrounds) and modest weights to other shapes.
    Returned as ratio 0..1.
    """
    drawings = page.get_drawings()
    clip_area = max(1.0, clip.get_area())
    weighted_area = 0.0

    for d in drawings:
        # skip hairlines and anything entirely outside the clip
        if float(d.get("width") or 0) <= 0.25:
            continue
        if "rect" not in d:
            continue

        rect = fitz.Rect(d["rect"])
        inter = rect.intersect(clip)
        inter_area = inter.get_area()
        if inter_area <= 0:
            continue

        frac = inter_area / clip_area
        # is this drawing just a rectangle op?
        is_rect_op = any((it and str(it[0]).lower() == "re") for it in d.get("items", []))

        # weight logic:
        # - large rectangles (likely panels/backgrounds) get very small weight
        # - smaller rectangles get small weight
        # - other shapes get medium weight
        if is_rect_op:
            weight = 0.15 if frac >= 0.25 else 0.30
        else:
            weight = 0.60

        weighted_area += inter_area * weight

    return min(1.0, weighted_area / clip_area)


def safe_int(v, default=0):
    try:
        if v is None: return default
        s = str(v).strip()
        if s == "" or s.lower() == "undefined": return default
        return int(float(s))
    except Exception:
        return default

def bottom_half_rect(page):
    r = page.rect
    return fitz.Rect(r.x0, r.y0 + r.height / 2.0, r.x1, r.y1)

def rect_from_params(page, data):
    """
    Build a fitz.Rect from coordsOrigin=pdf + x,y,width,height; 
    if missing/invalid, default to bottom-half.
    """
    if not data or data.get("coordsOrigin") != "pdf":
        return bottom_half_rect(page)
    try:
        x = float(data.get("x")); y = float(data.get("y"))
        w = float(data.get("width")); h = float(data.get("height"))
        pr = page.rect
        rx0 = max(pr.x0, x); ry0 = max(pr.y0, y)
        rx1 = min(pr.x1, x + w); ry1 = min(pr.y1, y + h)
        return fitz.Rect(rx0, ry0, rx1, ry1)
    except Exception:
        return bottom_half_rect(page)

def count_vector_segments_in(page, clip):
    """
    Count meaningful vector path segments intersecting the clip,
    ignoring hairline/template crumbs.
    """
    drawings = page.get_drawings()
    segs = 0
    VECTOR_OPS = {"m","l","c","v","y","re","h"}  # move/line/cubic/rect/close
    clip_area = max(1.0, clip.get_area())

    for d in drawings:
        # quick reject: bbox outside clip
        if "rect" in d and not fitz.Rect(d["rect"]).intersects(clip):
            continue
        # ignore ultra-thin template lines
        if float(d.get("width") or 0) <= 0.25:
            continue
        # ignore geometry whose bbox is tiny vs clip
        if "rect" in d:
            inter_area = fitz.Rect(d["rect"]).intersect(clip).get_area()
            if inter_area / clip_area < 0.0005:  # < 0.05% of clip area
                continue

        for it in d.get("items", []):
            op = str(it[0]).lower() if it else ""
            if op in VECTOR_OPS:
                segs += 1
    return segs

def sum_raster_coverage_in_clip(page, clip):
    """
    Return total raster area intersecting the clip, and coverage ratio (0..1).
    (Simple sum; overlaps may overcount a bit but it's fine for a heuristic.)
    """
    clip_area = max(1.0, clip.get_area())
    total_inter = 0.0
    for img in page.get_images(full=True):
        try:
            bbox = page.get_image_bbox(img)
            # if page.get_image_bbox raises on tuple, try xref
        except Exception:
            xref = img[0] if isinstance(img, (tuple, list)) else None
            if not xref:
                continue
            bbox = page.get_image_bbox(xref)

        inter = fitz.Rect(bbox).intersect(clip).get_area()
        total_inter += inter

    coverage = min(1.0, total_inter / clip_area) if clip_area > 0 else 0.0
    return total_inter, coverage

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
    Majority by coverage with sensible tie-breaks.
    - raster_cov, vector_cov are 0..1
    - vector_segments is a fallback signal for pure-vector pages
    """
    # clear wins by margin
    if raster_cov >= vector_cov + 0.08:
        return "has raster"
    if vector_cov >= raster_cov + 0.08:
        return "has vector"

    # bias to raster if a noticeable image exists and vector coverage is small
    if raster_cov >= 0.15 and vector_cov <= 0.25:
        return "has raster"

    # pure-vector safeguard (e.g., logos/illustrations with no images)
    if raster_cov <= 0.03 and vector_segments >= 120:
        return "has vector"

    # otherwise, pick the larger coverage; tie → raster
    return "has raster" if raster_cov >= vector_cov else "has vector"


# -------------------- routes --------------------

@app.route("/")
def home():
    return "Art Format Detector is live!"

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
        if page_index < 0 or page_index >= len(doc):
            page_index = 0
        page = doc.load_page(page_index)
        clip = rect_from_params(page, request.form)   # your existing bottom-half or coords

        # measures
        vector_segments = count_vector_segments_in(page, clip)      # keep as a backup signal
        _, raster_coverage = sum_raster_coverage_in_clip(page, clip)
        vector_cov_text = text_coverage_in_clip(page, clip)
        vector_cov_draw = drawings_coverage_in_clip(page, clip)
        vector_coverage = max(0.0, min(1.0, vector_cov_text + vector_cov_draw))

        # native raster metrics (as you had)
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
                "nativeDpiMin": round(dpi_min, 1)
            }

        fmt = decide_format_coverage(raster_coverage, vector_coverage, vector_segments)

        return jsonify({
            "format": fmt,  # 'has raster' or 'has vector'
            "metrics": {
                "region": "clip" if request.form.get("coordsOrigin") == "pdf" else "bottom-half",
                "rasterCount": len(page.get_images(full=True)),
                "rasterCoverage": round(raster_coverage, 3),
                "vectorCoverage": round(vector_coverage, 3),
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
        if page_index < 0 or page_index >= len(doc):
            page_index = 0
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
