# app.py  (Flask + PyMuPDF)
from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- helpers --------------------

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

def decide_format(raster_count, vector_segments):
    """
    Heuristic: prefer 'has raster' if any image and few vectors.
    """
    VEC_MIN_WEAK   = 20
    VEC_MIN_STRONG = 80
    if raster_count > 0 and vector_segments < VEC_MIN_WEAK:
        return "has raster"
    if raster_count > 0 and vector_segments >= VEC_MIN_WEAK:
        return "mixed"
    if vector_segments >= VEC_MIN_STRONG and raster_count == 0:
        return "has vector"
    if vector_segments >= VEC_MIN_WEAK and raster_count == 0:
        return "has vector"
    if raster_count > 0:
        return "has raster"
    return "unknown"

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
        "format": "has raster" | "has vector" | "mixed" | "unknown",
        "metrics": {
          "region": "clip" | "bottom-half",
          "rasterCount": int,
          "vectorSegments": int,
          "nativeRaster": {
            "nativePxW": int,
            "nativePxH": int,
            "placedWIn": float,
            "placedHIn": float,
            "nativeDpiX": float,
            "nativeDpiY": float,
            "nativeDpiMin": float
          }
        }
      }
    """
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400

        file = request.files["pdf"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page_index = int(request.form.get("pageIndex", "0"))
        page = doc.load_page(page_index)
        clip = rect_from_params(page, request.form)

        # count vector segments in clip
        vector_segments = count_vector_segments_in(page, clip)

        # count total rasters on page (cheap metric)
        raster_count = len(page.get_images(full=True))

        # largest raster intersecting the clip (for native DPI)
        native = {}
        largest = find_largest_image_in_clip(page, clip)
        if largest:
            xref, inter_rect, full_bbox = largest
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

        fmt = decide_format(raster_count, vector_segments)

        return jsonify({
            "format": fmt,
            "metrics": {
                "region": "clip" if request.form.get("coordsOrigin") == "pdf" else "bottom-half",
                "rasterCount": raster_count,
                "vectorSegments": vector_segments,
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
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page_index = int(request.form.get("pageIndex", "0"))
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
