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

        # Decide dominant
