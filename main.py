from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Art Format Detector is live!"

# ---------- helpers ----------

def bottom_half_rect(page):
    r = page.rect
    return fitz.Rect(r.x0, r.y0 + r.height / 2.0, r.x1, r.y1)

def count_rasters_in(page, clip_rect):
    """
    Count raster <image> placements intersecting the clip.
    """
    count = 0
    for img in page.get_images(full=True):
        try:
            bbox = page.get_image_bbox(img)
        except Exception:
            # Older PyMuPDF needs xref at [0]
            xref = img[0] if isinstance(img, (tuple, list)) else None
            if xref:
                bbox = page.get_image_bbox(xref)
            else:
                continue
        if fitz.Rect(bbox).intersects(clip_rect):
            count += 1
    return count

def count_vector_segments_in(page, clip_rect):
    """
    Count vector path segments intersecting the clip.
    We sum up the number of drawing 'items' that are actual path segments.
    """
    drawings = page.get_drawings()
    seg_count = 0

    # Operators that imply geometry (move/line/curve/rect/close)
    # NOTE: PyMuPDF item format: (op, points, color, ...). We check by op tag.
    VECTOR_OPS = {"m", "l", "c", "v", "y", "re", "h"}  # move, line, cubic, rect, close

    for d in drawings:
        # Fast reject if the drawing's bbox is outside the clip
        if "rect" in d and not fitz.Rect(d["rect"]).intersects(clip_rect):
            continue
        for it in d.get("items", []):
            op = str(it[0]).lower() if it and len(it) > 0 else ""
            if op in VECTOR_OPS:
                # Optional: quick point-in-clip test if points exist
                pts = it[1] if len(it) > 1 else None
                if pts:
                    # pts can be a list of fitz.Point; if any point lies in clip, count it
                    if any((hasattr(p, "x") and clip_rect.contains(p)) for p in pts if p is not None):
                        seg_count += 1
                    else:
                        # If we can’t test points (e.g., None), still count based on bbox hint
                        seg_count += 1
                else:
                    seg_count += 1
    return seg_count

def decide_format(raster_count, vector_segments):
    """
    Prefer 'raster' when any image is present and vector geometry is small.
    Use thresholds to avoid false 'vector' for page frames / guides.
    """
    # Tunable thresholds
    VEC_MIN_STRONG = 50   # strong vector presence if >= 50 segments
    VEC_MIN_WEAK   = 10   # weak vector presence if >= 10 segments

    if raster_count > 0 and vector_segments < VEC_MIN_WEAK:
        return "has raster"
    if vector_segments >= VEC_MIN_STRONG and raster_count == 0:
        return "has vector"
    if raster_count > 0 and vector_segments >= VEC_MIN_WEAK:
        return "mixed"
    if vector_segments >= VEC_MIN_WEAK:
        return "has vector"
    if raster_count > 0:
        return "has raster"
    return "unknown"

# ---------- routes ----------

@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No file field 'pdf'"}), 400

        file = request.files["pdf"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page = doc.load_page(0)
        clip_rect = bottom_half_rect(page)

        raster_count = count_rasters_in(page, clip_rect)
        vector_segments = count_vector_segments_in(page, clip_rect)

        result = decide_format(raster_count, vector_segments)

        return jsonify({
            "format": result,
            "metrics": {
                "rasterCount": raster_count,
                "vectorSegments": vector_segments,
                "region": "bottom-half"
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
        page = doc.load_page(0)
        clip_rect = bottom_half_rect(page)

        pix = page.get_pixmap(clip=clip_rect, dpi=300)
        image_path = os.path.join(UPLOAD_FOLDER, f"{filename}_design.png")
        pix.save(image_path)

        return send_file(image_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
