import os
import zipfile
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from hdr import process

app = Flask(__name__)

# 🔓 allow your website to call this API
CORS(app, origins=["https://thomasmielke.dev"])

UPLOAD_DIR = "uploads"
EXTRACT_DIR = os.path.join(UPLOAD_DIR, "extracted")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


# ---------------- HDR ENDPOINT ----------------
@app.route("/hdr", methods=["POST"])
def hdr():

    try:
        # ---------------- CHECK FILE ----------------
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        zip_path = os.path.join(UPLOAD_DIR, "input.zip")
        file.save(zip_path)

        # ---------------- CLEAN OLD FILES ----------------
        for f in os.listdir(EXTRACT_DIR):
            try:
                os.remove(os.path.join(EXTRACT_DIR, f))
            except:
                pass

        # ---------------- EXTRACT ZIP ----------------
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        # ---------------- LOAD IMAGES ----------------
        images = []

        for root, dirs, files in os.walk(EXTRACT_DIR):
            for name in files:

                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                path = os.path.join(root, name)

                img = cv2.imread(path, cv2.IMREAD_COLOR)

                if img is None:
                    print("❌ Skipping unreadable image:", path)
                    continue

                images.append(img)

        print("✅ Images loaded:", len(images))

        # ---------------- VALIDATION ----------------
        if len(images) < 2:
            return jsonify({
                "error": "Not enough valid images in ZIP (need at least 2 JPG/PNG files)"
            }), 400

        # ---------------- HDR PROCESS ----------------
        result = process(images)

        if result is None:
            return jsonify({"error": "HDR processing failed"}), 500

        # ---------------- SAVE OUTPUT ----------------
        out_path = os.path.join(UPLOAD_DIR, "result.jpg")
        cv2.imwrite(out_path, result)

        return send_file(out_path, mimetype="image/jpeg")


    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():
    return "HDR API running ✔"


# ---------------- START SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
