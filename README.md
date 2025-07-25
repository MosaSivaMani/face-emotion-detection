# Face Emotion Detection Web Application

A real-time face emotion detection web application built with Flask, OpenCV, and DeepFace. This application can detect and classify emotions from uploaded images or live webcam feed.

## Features

- **Image Upload**: Drag and drop or browse to upload images for emotion analysis
- **Webcam Support**: Real-time emotion detection using your device's camera
- **Multiple Face Detection**: Analyze emotions for multiple faces in a single image
- **Confidence Scores**: View detailed confidence scores for all emotion categories
- **Modern UI**: Beautiful, responsive web interface with Bootstrap 5
- **Real-time Processing**: Fast emotion detection with visual feedback

## Supported Emotions

The application can detect the following emotions:
- 😊 Happiness
- 😢 Sadness  
- 😠 Anger
- 😨 Fear
- 😲 Surprise
- 🤢 Disgust
- 😐 Neutral
- 😏 Contempt

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd face-emotion-detection
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Image Upload
1. Click on the "Upload Image" tab
2. Drag and drop an image or click "Choose File" to browse
3. Wait for the analysis to complete
4. View the detected emotions and confidence scores

### Webcam Analysis
1. Click on the "Webcam" tab
2. Click "Start Webcam" and grant camera permissions
3. Click "Capture & Analyze" to analyze the current frame
4. View the results and confidence scores

## Deployment Options

### Local Development
```bash
python app.py
```

### Production Deployment with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 5000
   
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

2. Build and run:
   ```bash
   docker build -t face-emotion-app .
   docker run -p 5000:5000 face-emotion-app
   ```

### Cloud Deployment

#### Heroku
1. Create a `Procfile`:
   ```
   web: gunicorn -w 4 -b 0.0.0.0:$PORT app:app
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### Railway
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Python app and deploy it

#### Render
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and analyze image
- `POST /webcam` - Analyze webcam frame

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Computer Vision**: OpenCV for face detection
- **AI/ML**: DeepFace for emotion recognition
- **Image Processing**: Real-time image analysis and processing

## Requirements

- Python 3.7+
- Webcam (for webcam functionality)
- Modern web browser with JavaScript enabled

## Troubleshooting

### Common Issues

1. **Camera not working**: Ensure you've granted camera permissions to your browser
2. **No faces detected**: Try adjusting lighting or using a clearer image
3. **Slow performance**: The first analysis may take longer as models are loaded

### Performance Tips

- Use images with clear, well-lit faces
- Ensure good lighting for webcam analysis
- Close other applications to free up system resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepFace library for emotion recognition
- OpenCV for computer vision capabilities
- Bootstrap for the responsive UI framework
# faceEmotionDetection
