# app.py  (Flask + PyMuPDF)
from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- helpers --------------------

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

def decide_format_majority(raster_coverage, vector_segments):
    """
    Always return only 'has raster' or 'has vector'.
    Majority rule:
      - Compute vector_score in [0..1] vs a strong threshold.
      - Use raster_score = coverage in [0..1].
      - Choose the higher score. Ties break to raster (safer for production).
    """
    VEC_MIN_WEAK   = 20
    VEC_MIN_STRONG = 80

    # quick wins
    if raster_coverage <= 0.01 and vector_segments >= VEC_MIN_WEAK:
        return "has vector"
    if raster_coverage >= 0.25 and vector_segments < VEC_MIN_WEAK:
        return "has raster"

    # normalized scores
    vector_score = max(0.0, min(1.0, vector_segments / float(VEC_MIN_STRONG)))
    raster_score = max(0.0, min(1.0, raster_coverage))

    if vector_score > raster_score:
        return "has vector"
    if raster_score > vector_score:
        return "has raster"
    # tie
    return "has raster"

# -------------------- routes --------------------

@app.route("/")
def home():
    return "Art Format Detector is live!"

@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    """
    Form-Data:
      pdf: file
      pageIndex: int (default 0)
      coordsOrigin: 'pdf' to enable x/y/width/height
      x, y, width, height: PDF points (72 pt = 1 in)
    Returns:
      {
        "format": "has raster" | "has vector",
        "metrics": {
          "region": "clip" | "bottom-half",
          "rasterCount": int,
          "rasterCoverage": float(0..1),
          "vectorSegments": int,
          "nativeRaster": {...optional...},
          "scores": {"raster": float, "vector": float}
        }
      }
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

        # vector + raster measures
        vector_segments = count_vector_segments_in(page, clip)
        _, raster_coverage = sum_raster_coverage_in_clip(page, clip)

        # largest raster (for DPI metrics)
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

        # decision (no 'mixed')
        fmt = decide_format_majority(raster_coverage, vector_segments)

        # diagnostics (optional but handy)
        # mirror the internal scores used for the decision
        scores = {
            "raster": round(max(0.0, min(1.0, raster_coverage)), 3),
            "vector": round(max(0.0, min(1.0, vector_segments / 80.0)), 3),
        }

        return jsonify({
            "format": fmt,  # only 'has raster' or 'has vector'
            "metrics": {
                "region": "clip" if request.form.get("coordsOrigin") == "pdf" else "bottom-half",
                "rasterCount": len(page.get_images(full=True)),
                "rasterCoverage": round(raster_coverage, 3),
                "vectorSegments": int(vector_segments),
                **({"nativeRaster": native} if native else {}),
                "scores": scores
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
