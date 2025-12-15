# ForensicaAI - Streamlit Deepfake Detection Frontend

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** - The app will automatically open at `http://localhost:8501`

## Features

- 📤 **Video Upload**: Upload MP4, AVI, MOV, or MKV files
- 🔍 **Real-time Analysis**: Process videos with progress indicators
- 📊 **Confidence Scores**: View detailed confidence metrics
- 📋 **Detailed Reports**: Get comprehensive analysis reports
- 📥 **Export Reports**: Download analysis reports as text files
- 🖼️ **Frame Preview**: See sample frames analyzed by the model

## Usage

1. Upload a video file using the file uploader
2. Click "🚀 Analyze Video" button
3. Wait for processing (extracts 20 frames and analyzes)
4. View results:
   - Prediction (REAL/FAKE)
   - Confidence percentage
   - Detailed report
   - Sample frames

## Requirements

- Trained model at `models/best_model.pth`
- Python 3.8+
- All dependencies from `requirements.txt`

## Troubleshooting

- **Model not found**: Make sure you've trained the model first using `python src/train.py`
- **Video too short**: Videos must have at least 20 frames
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

