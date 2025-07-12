#!/usr/bin/env python3
"""
Smart Ball Detection System - IoT Project
Made by: Alvin, Daffa, Abidzar, Ridho
Simple version without OpenCV dependencies
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Roboflow API configuration
API_KEY = "Am7lrBZW1eYzrwt0rWvc"
MODEL_ID = "ball-t8zxj/15"
API_URL = "https://serverless.roboflow.com"

print("✅ Smart Ball Detection System initialized!")
print(f"🔍 Model ID: {MODEL_ID}")
print(f"🔍 API URL: {API_URL}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def detect_balls_api(image_path):
    """Detect balls using direct Roboflow API"""
    try:
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Get image info
        original_img = Image.open(image_path)
        file_size = os.path.getsize(image_path)
        print(f"📏 Original image size: {original_img.size} (width x height)")
        print(f"📁 File size: {file_size} bytes, format: {original_img.format}")
        
        # Prepare API request
        url = f"{API_URL}/{MODEL_ID}"
        
        # Send request
        print(f"🚀 Sending to Roboflow API...")
        response = requests.post(
            url,
            files={"file": image_data},
            data={"api_key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"🔍 API result: {len(result.get('predictions', []))} predictions")
            
            for i, pred in enumerate(result.get('predictions', [])):
                print(f"  🎯 {i+1}: {pred['class']} - {pred['confidence']:.1%}")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return {"error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

def draw_bounding_boxes_simple(image_path, predictions):
    """Draw bounding boxes on image using PIL"""
    try:
        # Open image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Colors for different classes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
        
        for i, pred in enumerate(predictions):
            # Get coordinates
            x = pred['x']
            y = pred['y'] 
            width = pred['width']
            height = pred['height']
            
            # Calculate box coordinates
            x1 = x - width/2
            y1 = y - height/2
            x2 = x + width/2
            y2 = y + height/2
            
            # Draw rectangle
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{pred['class']} {pred['confidence']:.1%}"
            draw.text((x1, y1-20), label, fill=color)
            
            print(f"  📦 {i+1}: {pred['class']} - Box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
        
        print(f"✅ Drew {len(predictions)} bounding boxes")
        return image
        
    except Exception as e:
        print(f"❌ Error drawing boxes: {e}")
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
        result = detect_balls_api(filepath)

        # Draw bounding boxes
        try:
            image_with_boxes = draw_bounding_boxes_simple(filepath, result.get('predictions', []))

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
