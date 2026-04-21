import os
import zipfile
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
from hdr import process_images

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
EXTRACT_DIR = os.path.join(UPLOAD_DIR, "extracted")

os.makedirs(EXTRACT_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return "HDR service running ✔"


@app.route("/hdr", methods=["POST"])
def hdr():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        zip_path = os.path.join(UPLOAD_DIR, "input.zip")
        file.save(zip_path)

        # clear folder
        for f in os.listdir(EXTRACT_DIR):
            try:
                os.remove(os.path.join(EXTRACT_DIR, f))
            except:
                pass

        # extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        images = []

        for root, _, files in os.walk(EXTRACT_DIR):
            for name in files:

                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                path = os.path.join(root, name)

                img = cv2.imread(path)

                if img is None:
                    continue

                images.append(img)

        if len(images) < 2:
            return jsonify({"error": "Need at least 2 images"}), 400

        result = process_images(images)

        out_path = os.path.join(UPLOAD_DIR, "result.jpg")
        cv2.imwrite(out_path, result)

        return send_file(out_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
