from flask import Flask, request, jsonify
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
        pixmaps = []
        vector_paths = []

        # Page dimensions
        rect = page.rect
        mid_y = rect.y0 + (rect.height / 2)
        clip_rect = fitz.Rect(rect.x0, mid_y, rect.x1, rect.y1)

        # Raster: Check for images in bottom half
        for img in page.get_images(full=True):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            if bbox.intersects(clip_rect):
                pix = fitz.Pixmap(doc, xref)
                pixmaps.append(pix)

        # Vector: Check for vector paths in bottom half
        drawings = page.get_drawings()
        for item in drawings:
            if 'rect' in item:
                path_rect = fitz.Rect(item['rect'])  # Convert tuple to Rect
                if path_rect.intersects(clip_rect):
                    vector_paths.append(item)

        # Determine result
        has_vector = len(vector_paths) > 0
        has_raster = len(pixmaps) > 0

        if has_vector and has_raster:
            result = "Mixed (Vector + Raster)"
        elif has_vector:
            result = "Vector"
        elif has_raster:
            result = "Raster"
        else:
            result = "Unknown"

        return jsonify({"format": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
