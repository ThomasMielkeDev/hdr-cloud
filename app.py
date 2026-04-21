import os
import zipfile
import cv2
from flask import Flask, request, send_file
from hdr import process
from flask_cors import CORS

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/hdr", methods=["POST"])
def hdr():

    file = request.files["file"]

    zip_path = os.path.join(UPLOAD_DIR, "input.zip")
    file.save(zip_path)

    extract_path = os.path.join(UPLOAD_DIR, "extracted")
    os.makedirs(extract_path, exist_ok=True)

    # clear old files
    for f in os.listdir(extract_path):
        os.remove(os.path.join(extract_path, f))

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    images = []

    for name in sorted(os.listdir(extract_path)):
        if name.lower().endswith((".jpg",".jpeg",".png")):
            img = cv2.imread(os.path.join(extract_path, name))
            if img is not None:
                images.append(img)

    if len(images) < 2:
        return "Need at least 2 images", 400

    result = process(images)

    out_path = os.path.join(UPLOAD_DIR, "result.jpg")
    cv2.imwrite(out_path, result)

    return send_file(out_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
