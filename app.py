import os
import zipfile
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from hdr import process

app = Flask(__name__)
CORS(app, origins=["https://thomasmielke.dev"])

UPLOAD_DIR = "uploads"
EXTRACT_DIR = os.path.join(UPLOAD_DIR, "extracted")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return "HDR API running ✔"


@app.route("/hdr", methods=["POST"])
def hdr():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        vividness = float(request.form.get("vividness", 1.2))
        sky_boost = float(request.form.get("sky", 1.3))

        zip_path = os.path.join(UPLOAD_DIR, "input.zip")
        file.save(zip_path)

        # clear extraction folder
        for f in os.listdir(EXTRACT_DIR):
            try:
                os.remove(os.path.join(EXTRACT_DIR, f))
            except:
                pass

        # extract zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        images = []

        for root, _, files in os.walk(EXTRACT_DIR):
            for name in files:

                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                path = os.path.join(root, name)

                img = cv2.imread(path, cv2.IMREAD_COLOR)

                if img is None:
                    print("Skipping unreadable:", path)
                    continue

                images.append(img)

        print("Loaded images:", len(images))

        if len(images) < 2:
            return jsonify({"error": "Need at least 2 valid images"}), 400

        result = process(images, vividness, sky_boost)

        out_path = os.path.join(UPLOAD_DIR, "result.jpg")
        cv2.imwrite(out_path, result)

        return send_file(out_path, mimetype="image/jpeg")


    except Exception as e:
        print("HDR ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
