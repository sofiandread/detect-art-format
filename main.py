from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Art Format Detector is live!"

@app.route('/detect-art-format', methods=['POST'])
def detect_art_format():
    try:
        file = request.files['data']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page = doc.load_page(0)  # First page

        # Get bottom half of the page
        rect = page.rect
        mid_y = rect.y0 + (rect.height / 2)
        clip_rect = fitz.Rect(rect.x0, mid_y, rect.x1, rect.y1)

        # Raster: Detect images in bottom half
        raster_count = 0
        for img in page.get_images(full=True):
            bbox = page.get_image_bbox(img)
            if bbox.intersects(clip_rect):
                raster_count += 1

        # Vector: Detect vector paths in bottom half
        vector_count = 0
        for item in page.get_drawings():
            if 'rect' in item:
                path_rect = fitz.Rect(item['rect'])
                if path_rect.intersects(clip_rect):
                    vector_count += 1

        # Decide dominant type
        if vector_count > raster_count:
            result = "has vector"
        elif raster_count > vector_count:
            result = "has raster"
        elif vector_count == raster_count and vector_count > 0:
            result = "has vector"  # default to vector when tied
        else:
            result = "unknown"

        return jsonify({"format": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/extract-design-image', methods=['POST'])
def extract_design_image():
    try:
        file = request.files['data']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        page = doc.load_page(0)

        # Crop to bottom half of the page
        rect = page.rect
        clip_rect = fitz.Rect(rect.x0, rect.y0 + rect.height / 2, rect.x1, rect.y1)

        pix = page.get_pixmap(clip=clip_rect, dpi=300)
        image_path = os.path.join(UPLOAD_FOLDER, f"{filename}_design.png")
        pix.save(image_path)

        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
