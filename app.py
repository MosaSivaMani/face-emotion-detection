from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import base64
import os
from werkzeug.utils import secure_filename
import tempfile
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_emotion(image):
    """Analyze emotion in the given image"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(rgb_image, actions=['emotion'], enforce_detection=False)
        
        # Get dominant emotion and confidence scores
        dominant_emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']
        
        # Convert numpy values to regular Python floats for JSON serialization
        emotion_scores_serializable = {}
        for emotion, score in emotion_scores.items():
            emotion_scores_serializable[emotion] = float(score)
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_scores': emotion_scores_serializable,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def detect_faces_and_emotions(image):
    """Detect faces and analyze emotions for each face"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Analyze emotion for this face
            emotion_result = analyze_emotion(face_roi)
            
            if emotion_result['success']:
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'emotion': emotion_result['dominant_emotion'],
                    'confidence_scores': emotion_result['emotion_scores']
                })
        
        return results
    except Exception as e:
        print(f"Error in detect_faces_and_emotions: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read the image
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Detect faces and analyze emotions
            results = detect_faces_and_emotions(image)
            
            # Encode image for display
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': img_base64,
                'faces': results
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/webcam', methods=['POST'])
def webcam_analysis():
    try:
        # Get base64 image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        img_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect faces and analyze emotions
        results = detect_faces_and_emotions(image)
        
        return jsonify({
            'success': True,
            'faces': results
        })
        
    except Exception as e:
        print(f"Error in webcam_analysis: {str(e)}")
        return jsonify({'error': f'Error processing webcam image: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Face Emotion Detection API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 