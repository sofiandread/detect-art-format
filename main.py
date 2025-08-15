from flask import Flask, request, jsonify, send_file
import fitz, os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def rect_from_params(page, data):
    """Build a fitz.Rect from coordsOrigin=pdf + x,y,width,height; else bottom-half."""
    if not data or data.get("coordsOrigin") != "pdf":
        r = page.rect
        return fitz.Rect(r.x0, r.y0 + r.height/2.0, r.x1, r.y1)
    try:
        x = float(data["x"]); y = float(data["y"])
        w = float(data["width"]); h = float(data["height"])
        # clamp to page
        pr = page.rect
        rx0 = max(pr.x0, x); ry0 = max(pr.y0, y)
        rx1 = min(pr.x1, x + w); ry1 = min(pr.y1, y + h)
        return fitz.Rect(rx0, ry0, rx1, ry1)
    except Exception:
        # fallback: bottom-half
        r = page.rect
        return fitz.Rect(r.x0, r.y0 + r.height/2.0, r.x1, r.y1)

def count_rasters_in(page, clip):
    c = 0; area_biggest = 0.0
    for img in page.get_images(full=True):
        try:
            bbox = page.get_image_bbox(img)
        except Exception:
            xref = img[0] if isinstance(img, (tuple, list)) else None
            if not xref: continue
            bbox = page.get_image_bbox(xref)
        rb = fitz.Rect(bbox)
        if rb.intersects(clip):
            c += 1
            area_biggest = max(area_biggest, rb.intersect(clip).get_area())
    return c, area_biggest

def count_vector_segments_in(page, clip):
    """Count meaningful vector segments in the clip; ignore tiny/template bits."""
    drawings = page.get_drawings()
    segs = 0
    clip_area = max(1.0, clip.get_area())

    VECTOR_OPS = {"m","l","c","v","y","re","h"}  # path ops
    for d in drawings:
        # fast reject: drawing bbox outside clip
        if "rect" in d and not fitz.Rect(d["rect"]).intersects(clip):
            continue
        stroke_w = float(d.get("width") or 0)
        # very thin template lines (hairlines)
        if stroke_w and stroke_w <= 0.25:
            continue

        # ignore very small geometry by bbox area %
        bbox = fitz.Rect(d["rect"]) if "rect" in d else None
        if bbox:
            if bbox.intersect(clip).get_area() / clip_area < 0.0005:  # <0.05% of clip
                continue

        # count real path ops that touch the clip
        for it in d.get("items", []):
            op = str(it[0]).lower() if it else ""
            if op not in VECTOR_OPS: continue
            pts = it[1] if len(it) > 1 else None
            if pts and any((hasattr(p, "x") and clip.contains(p)) for p in pts if p is not None):
                segs += 1
            else:
                segs += 1
    return segs

def decide_format(raster_count, vector_segments, raster_area, clip_area):
    """Prefer raster if image occupies a meaningful portion of the print box."""
    VEC_MIN_WEAK = 20     # ignore stray vector crumbs below this
    VEC_MIN_STRONG = 80
    RASTER_AREA_RATIO = 0.15  # if image covers ≥15% of clip, call raster/mixed

    raster_ratio = raster_area / max(1.0, clip_area)

    if raster_count > 0 and raster_ratio >= RASTER_AREA_RATIO and vector_segments < VEC_MIN_WEAK:
        return "has raster"
    if raster_count > 0 and raster_ratio >= RASTER_AREA_RATIO and vector_segments >= VEC_MIN_WEAK:
        return "mixed"
    if raster_count > 0 and vector_segments < VEC_MIN_WEAK:
        return "has raster"
    if vector_segments >= VEC_MIN_STRONG and raster_count == 0:
        return "has vector"
    if vector_segments >= VEC_MIN_WEAK and raster_count == 0:
        return "has vector"
    if raster_count > 0:
        return "has raster"
    return "unknown"

@app.route("/")
def home():
    return "Art Format Detector is live!"

@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400
        file = request.files["pdf"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page_index = int(request.form.get("pageIndex", "0"))
        page = doc.load_page(page_index)

        # build clip rect from params (if provided)
        clip = rect_from_params(page, request.form)

        raster_count, raster_area_biggest = count_rasters_in(page, clip)
        vector_segments = count_vector_segments_in(page, clip)
        fmt = decide_format(raster_count, vector_segments, raster_area_biggest, clip.get_area())

        return jsonify({
            "format": fmt,
            "metrics": {
                "region": "clip" if request.form.get("coordsOrigin") == "pdf" else "bottom-half",
                "rasterCount": raster_count,
                "vectorSegments": vector_segments,
                "rasterAreaLargest": raster_area_biggest,
                "clipArea": clip.get_area()
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extract-design-image", methods=["POST"])
def extract_design_image():
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
