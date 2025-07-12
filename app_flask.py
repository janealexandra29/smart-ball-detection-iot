#!/usr/bin/env python3
"""
Smart Ball Detection System - IoT Project
Made by: Alvin, Daffa, Abidzar, Ridho
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
from PIL import Image
from inference_sdk import InferenceHTTPClient
import base64
from io import BytesIO
import supervision as sv
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Roboflow Inference SDK configuration
API_KEY = "Am7lrBZW1eYzrwt0rWvc"
MODEL_ID = "ball-t8zxj/15"

# Initialize Inference Client
try:
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )
    print("✅ Inference client initialized successfully!")
    print(f"🔍 Model ID: {MODEL_ID}")
    print(f"🔍 API URL: https://serverless.roboflow.com")

    # Test client with a simple prediction (if test image exists)
    test_image_path = "test_ball.jpg"
    if os.path.exists(test_image_path):
        try:
            print("🧪 Testing client with sample image...")
            test_result = CLIENT.infer(test_image_path, model_id=MODEL_ID)
            print(f"🧪 Test result: {len(test_result.get('predictions', []))} predictions")
        except Exception as test_e:
            print(f"🧪 Test failed: {test_e}")
    else:
        print("🧪 No test image found, skipping client test")
except Exception as e:
    print(f"❌ Error initializing client: {e}")
    CLIENT = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def detect_balls(image_path):
    """Detect balls in image using Inference SDK"""
    if CLIENT is None:
        return {"error": "Inference client not available"}

    try:
        # Check original image size and file info
        from PIL import Image as PILImage
        original_img = PILImage.open(image_path)
        file_size = os.path.getsize(image_path)
        print(f"📏 Original image size: {original_img.size} (width x height)")
        print(f"📁 File size: {file_size} bytes, format: {original_img.format}")

        print(f"🚀 Sending to Roboflow Inference API...")
        result_json = CLIENT.infer(image_path, model_id=MODEL_ID)

        # Debug: Print raw result
        print(f"🔍 Raw Roboflow result: {len(result_json.get('predictions', []))} predictions")
        if result_json.get('predictions'):
            for i, pred in enumerate(result_json['predictions']):
                print(f"  🎯 {i+1}: {pred.get('class', 'unknown')} - {pred.get('confidence', 0):.1%}")

        # Check if Roboflow returns image dimensions
        roboflow_width = None
        roboflow_height = None

        if 'image' in result_json:
            roboflow_dims = result_json['image']
            print(f"📏 Roboflow image dimensions: {roboflow_dims}")
            if 'width' in roboflow_dims and 'height' in roboflow_dims:
                roboflow_width = int(roboflow_dims['width'])
                roboflow_height = int(roboflow_dims['height'])
        else:
            print("📏 No image dimensions in Roboflow response")

        # Scale coordinates back to original size if needed
        if roboflow_width and roboflow_height:
            original_width, original_height = original_img.size
            scale_x = original_width / roboflow_width
            scale_y = original_height / roboflow_height

            print(f"🔄 Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")

            # Scale all predictions
            if 'predictions' in result_json:
                for pred in result_json['predictions']:
                    pred['x'] = pred['x'] * scale_x
                    pred['y'] = pred['y'] * scale_y
                    pred['width'] = pred['width'] * scale_x
                    pred['height'] = pred['height'] * scale_y
                    print(f"🔄 Scaled {pred['class']}: center=({pred['x']:.1f}, {pred['y']:.1f}) size=({pred['width']:.1f}x{pred['height']:.1f})")

        return result_json
    except Exception as e:
        return {"error": str(e)}

def draw_bounding_boxes_simple(image_path, predictions):
    """Simple manual drawing for debugging"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        print(f"✅ PIL Image loaded: {image.size}")

        # Try to get font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

        for i, pred in enumerate(predictions):
            x_center = pred['x']
            y_center = pred['y']
            width = pred['width']
            height = pred['height']

            # Manual conversion
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            color = colors[i % len(colors)]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw center point
            draw.ellipse([x_center-3, y_center-3, x_center+3, y_center+3], fill=color)

            # Draw label
            label = f"{pred['class']} {pred['confidence']:.1%}"
            draw.text((x1, y1-25), label, fill=color, font=font)

            print(f"  📦 {i+1}: {pred['class']} - Manual box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")

        return image

    except Exception as e:
        print(f"❌ Simple drawing error: {e}")
        return Image.open(image_path)

def draw_bounding_boxes_supervision(image_path, predictions):
    """Draw bounding boxes using Supervision library - like Roboflow Universe"""
    try:
        # Load image with OpenCV
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Could not load image: {image_path}")
            # Try with PIL as fallback
            pil_image = Image.open(image_path)
            return pil_image

        print(f"✅ Image loaded: {image.shape}")

        if len(predictions) == 0:
            print("ℹ️ No predictions, returning original image")
            # Convert back to PIL and return
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)

        # Prepare data for Supervision
        xyxy_boxes = []
        class_ids = []
        confidences = []
        class_names = []

        print(f"✅ Processing {len(predictions)} predictions")
        print(f"📏 Image dimensions: {image.shape}")

        # Check if coordinates are normalized
        first_pred = predictions[0] if predictions else None
        if first_pred:
            is_normalized = (first_pred['x'] <= 1.0 and first_pred['y'] <= 1.0 and
                           first_pred['width'] <= 1.0 and first_pred['height'] <= 1.0)
            print(f"🔍 Coordinates appear {'NORMALIZED (0-1)' if is_normalized else 'ABSOLUTE (pixels)'}")
            print(f"🔍 Raw values: x={first_pred['x']}, y={first_pred['y']}, w={first_pred['width']}, h={first_pred['height']}")

        # Prepare center coordinates for conversion
        xcycwh_boxes = []

        for i, pred in enumerate(predictions):
            x_center = pred['x']
            y_center = pred['y']
            width = pred['width']
            height = pred['height']

            # Check if coordinates need denormalization
            if x_center <= 1.0 and y_center <= 1.0 and width <= 1.0 and height <= 1.0:
                # Coordinates are normalized, convert to pixels
                img_height, img_width = image.shape[:2]
                x_center = x_center * img_width
                y_center = y_center * img_height
                width = width * img_width
                height = height * img_height
                print(f"  🔄 Denormalized coordinates for prediction {i+1}")

            xcycwh_boxes.append([x_center, y_center, width, height])
            class_ids.append(i)  # Use index for different colors
            confidences.append(pred['confidence'])
            class_names.append(pred['class'].upper())

            print(f"  📦 {i+1}: {pred['class']} center=({x_center:.1f}, {y_center:.1f}) size=({width:.1f}x{height:.1f}) - {pred['confidence']:.2%}")

        # Convert center coordinates to corner coordinates using Supervision
        xcycwh_array = np.array(xcycwh_boxes)
        xyxy_array = sv.xcycwh_to_xyxy(xcycwh_array)
        xyxy_boxes = xyxy_array.tolist()

        print(f"✅ Converted coordinates:")
        for i, (xyxy, xcycwh) in enumerate(zip(xyxy_boxes, xcycwh_boxes)):
            print(f"  📦 {i+1}: center({xcycwh[0]:.1f}, {xcycwh[1]:.1f}) → corner({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f})")

        # Create Supervision Detections object
        detections = sv.Detections(
            xyxy=np.array(xyxy_boxes),
            class_id=np.array(class_ids),
            confidence=np.array(confidences)
        )

        # Create annotators - like Roboflow Universe style
        box_annotator = sv.BoxAnnotator(
            thickness=3,
            color_lookup=sv.ColorLookup.INDEX
        )

        label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.8,
            color_lookup=sv.ColorLookup.INDEX
        )

        # Create labels with class name and confidence
        labels = [
            f"{class_name} {confidence:.1%}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        # Annotate image
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )

        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        # Convert back to PIL
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(image_rgb)

        print(f"✅ Supervision annotation complete: {result_image.size}")
        return result_image

    except Exception as e:
        print(f"❌ Error drawing bounding boxes with supervision: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to original image
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image_rgb)
            else:
                return Image.open(image_path)
        except:
            return Image.open(image_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect balls
        result = detect_balls(filepath)

        # Always try to draw bounding boxes, even if no detections
        try:
            # Use simple drawing for debugging
            image_with_boxes = draw_bounding_boxes_simple(filepath, result.get('predictions', []))

            # Convert RGBA to RGB if needed
            if image_with_boxes.mode == 'RGBA':
                image_with_boxes = image_with_boxes.convert('RGB')

            # Convert to base64
            buffer = BytesIO()
            image_with_boxes.save(buffer, format='JPEG', quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            print(f"✅ Image processed successfully, base64 length: {len(img_base64)}")

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            # Fallback to original image
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()

        # Clean up
        os.remove(filepath)
        
        # Prepare response
        response_data = {
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'result': result,
            'predictions_count': len(result.get('predictions', [])),
            'timestamp': time.time()
        }

        print(f"✅ Sending response: {response_data['predictions_count']} predictions, image size: {len(img_base64)} chars")

        return jsonify(response_data)
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    import os
    from socket import gethostname

    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'

    # Only run if not on PythonAnywhere
    if 'liveconsole' not in gethostname():
        app.run(debug=debug, host='0.0.0.0', port=port)
