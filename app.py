import os
from flask import Flask, request, render_template, redirect
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_darkness(image):
    alpha = 1.5  
    beta = -50   
    dark_sketch = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return dark_sketch

def create_sketch(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    invert = cv2.bitwise_not(gray_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray_img, inverted_blur, scale=200.0)
    sketch = enhance_darkness(sketch)
    
    # Save the processed image
    sketch_path = os.path.join(app.config['PROCESSED_FOLDER'], 'sketch.png')
    cv2.imwrite(sketch_path, sketch)
    return sketch_path

@app.route('/', methods=['GET', 'POST'])
def index():
    sketch_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            sketch_path = create_sketch(filepath)
            
            return render_template('index.html', sketch_url=sketch_path)
    
    return render_template('index.html', sketch_url=None)

if __name__ == '__main__':
    app.run(debug=True)
