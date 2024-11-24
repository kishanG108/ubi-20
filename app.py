from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance
import cv2
import os
import numpy as np
import easyocr
import math
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def enhance_image(image_path):
    image = Image.open(image_path)

    # Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(image)
    image_contrast = contrast_enhancer.enhance(2.0)

    # Enhance sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(image_contrast)
    image_sharp = sharpness_enhancer.enhance(2.0)

    enhanced_path = os.path.join("outputs", "enhanced.jpg")
    image_sharp.save(enhanced_path)
    return enhanced_path

def resize_image(image_path, target_size):
    with Image.open(image_path) as img:
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_path = os.path.join("outputs", "resized.jpg")
        resized_img.save(resized_path)
    return resized_path

def find_roi(image_path, min_area=1000, max_area=10000):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box_area = w * h
        if min_area <= bounding_box_area <= max_area:
            cropped = image[y:y+h, x:x+w]
            cropped_images.append(cropped)

    roi_paths = []
    for i, cropped in enumerate(cropped_images):
        roi_path = os.path.join("outputs", f"roi_{i}.jpg")
        cv2.imwrite(roi_path, cropped)
        roi_paths.append(roi_path)

    return roi_paths

def calculate_speed(image):
    # Placeholder for speed calculation logic, can use the provided logic here
    return "Speed calculation not implemented in detail here."

def extract_odometer_reading(image_path):
    reader = easyocr.Reader(['en'])
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.multiply(gray_image, 0.9)
    _, white_thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    result = reader.readtext(gray_image)
    six_digit_numbers = [text for _, text, _ in result if text.isdigit() and len(text) == 6]

    return six_digit_numbers[0] if six_digit_numbers else "No valid reading found."

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    input_path = os.path.join("uploads", filename)
    file.save(input_path)

    # Enhance image
    enhanced_path = enhance_image(input_path)

    # Resize image
    resized_path = resize_image(enhanced_path, (1600, 720))

    # Find ROI
    roi_paths = find_roi(resized_path)

    # Resize ROI and extract odometer reading for each ROI
    results = {"speed": None, "odometer_readings": []}
    for roi_path in roi_paths:
        resized_roi_path = resize_image(roi_path, (941, 167))
        reading = extract_odometer_reading(resized_roi_path)
        results["odometer_readings"].append(reading)

    # Calculate speed (if needed)
    results["speed"] = calculate_speed(cv2.imread(resized_path))

    return jsonify({
        "enhanced_image": enhanced_path,
        "resized_image": resized_path,
        "roi_images": roi_paths,
        "results": results
    })

if __name__ == '_main_':
    app.run(debug=True)
