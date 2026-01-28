# 🧬 **ForensicaAI – Deepfake Video Detection Using CNN + LSTM (Spatial–Temporal Model)**

ForensicaAI is a lightweight, CPU-friendly deepfake detection prototype built using a **hybrid spatial–temporal architecture**:
a **Convolutional Neural Network (CNN)** for frame-level spatial feature extraction and an **LSTM (RNN)** for temporal sequence modeling across video frames.

Designed for academic research and prototype development, this system processes raw videos, extracts frames, learns visual artifacts commonly found in deepfake content, and produces a **Real/Fake** prediction with confidence.

---

## 📌 **Key Features**

* ✔ **End-to-end deepfake detection pipeline**
* ✔ Lightweight CNN for spatial feature extraction
* ✔ LSTM sequence model for capturing temporal inconsistencies
* ✔ Supports any dataset with `video,label` format
* ✔ CPU-friendly (works on low-end laptops)
* ✔ Jupyter Notebook included for training, testing, and inference
* ✔ Modular & easy to extend to ViT, MobileNet, EfficientNet, etc.
* ✔ Clean folder structure for academic submission

---

# 🔍 **1. Project Description**

Deepfakes use generative models to manipulate human faces in videos. While visually convincing, these fakes often contain subtle artifacts in:

* **Spatial domain (frame-level)**

  * Texture irregularities
  * Lighting inconsistencies
  * Blending boundaries
  * GAN-specific fingerprints

* **Temporal domain (across frames)**

  * Unnatural blinking
  * Irregular lip motion
  * Inconsistent identity features
  * Sudden frame transitions

To detect such anomalies, ForensicaAI uses a **hybrid CNN–LSTM model**:

* CNN processes each frame and extracts a 256-dimensional spatial embedding.
* LSTM reads the sequence of embeddings and detects temporal abnormalities.
* Final classifier outputs **REAL** or **FAKE** with confidence.

This makes the system ideal for forensic analysis, research, and early-stage product prototyping.

---

# 🎯 **2. Problem Statement**

With the rapid evolution of deepfake generation techniques, it has become increasingly difficult to identify manipulated videos using traditional visual inspection. The lack of automated, accessible deepfake detection systems poses significant risks to:

* Public trust
* Media authenticity
* Cybersecurity
* Legal proceedings
* Personal identity/privacy

**Goal:**
Build a practical and efficient deepfake video detection model that works on small-scale hardware, while providing strong spatial–temporal analysis.

---

# 📁 **3. Dataset Description**

The system expects a dataset structured as:

```
data/
 ├── raw_videos/
 │     ├── video1.mp4
 │     ├── video2.mp4
 │     └── ...
 ├── frames/
 └── labels.csv
```

### Celeb-DF-v2 quick start

1. Place the original dataset at `Celeb-DF-v2/` (sibling to this repo) with `Celeb-real` and `Celeb-synthesis` inside.
2. Build labels and copy videos into the project layout:
   ```
   python src/data_prep/prepare_celeb_df_v2.py --celeb_root Celeb-DF-v2
   ```
   This writes `data/labels.csv` (`video,label`) and fills `data/raw_videos/` with mp4s.
3. Extract frames into `data/frames/<video_name>/frame_i.jpg`:
   ```
   python src/data_prep/extract_frames.py
   ```
4. Train/evaluate using the generated frames and labels.

Configurable paths live in `config.yaml` (`raw_video_dir`, `frames_dir`, `labels_csv`, `num_frames`, `frame_size`).

### **labels.csv format**

```
video,label
video1.mp4,0
video2.mp4,1
...
```

Where:

* **0 = REAL**
* **1 = FAKE**

### Supported datasets:

* FaceForensics++ (manually preprocessed)
* DFDC Preview dataset
* Celeb-DF
* Custom recorded dataset

### Final dataset example used:

* **500 videos**
* 20 frames extracted per video
* 10,000 processed image frames in total

---

# 🧠 **4. System Architecture**

```
Raw Video
   │
   ├── Frame Extraction (20 frames per video)
   │
   ├── CNN (Spatial Feature Extractor)
   │       ↓ 256-dim embedding
   │
   ├── LSTM (Temporal Sequence Model)
   │       ↓
   ├── Fully Connected Layer
   │
   └── Output: Real / Fake + Confidence
```

### **Spatial Model (CNN)**

* Lightweight 3-layer convolutional encoder
* Downsamples frames by factor of 8
* Outputs fixed 256-dimensional vectors

### **Temporal Model (LSTM)**

* Hidden size: 128
* Reads sequences of 20 frames
* Learns motion/consistency patterns

---

# ⚙️ **5. Installation**

Clone the repository:

```bash
git clone https://github.com/yourusername/ForensicaAI.git
cd ForensicaAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Pillow

---

# 🛠️ **6. Usage Guide**

## **A. Extract Frames**

```python
python extract_frames.py
```

Or run the extraction cell in the Jupyter Notebook.

## **B. Train Model**

Run the notebook:

```
ForensicaAI_OptionA_Prototype.ipynb
```

Or train using CLI:

```python
python train.py
```

## **C. Evaluate Model**

```python
python evaluate.py
```

## **D. Run Streamlit App (Video Upload & Detection)**

The easiest way to use the trained model is through the Streamlit web interface:

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Or manually:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. You can:
- Upload video files (MP4, AVI, MOV, MKV)
- Get real-time predictions (REAL/FAKE)
- View confidence scores and detailed reports
- Download analysis reports

**Note:** Make sure `models/best_model.pth` exists before running the app.

## **E. Command Line Inference (Alternative)**

```python
python src/inference.py
```

Then enter the video path when prompted.

Output example:

```
Prediction: FAKE
Confidence: 0.94
```

---

# 📦 **7. Folder Structure**

```
ForensicaAI/
│
├── data/
│   ├── raw_videos/
│   ├── frames/
│   └── labels.csv
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── walkthrough.ipynb
│
├── src/
│   ├── dataset/
│   │   └── video_dataset.py
│   ├── models/
│   │   ├── cnn_feature_extractor.py
│   │   └── rnn_classifier.py
│   ├── data_prep/
│   │   ├── extract_frames.py
│   │   └── prepare_celeb_df_v2.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
│
├── models/
│   └── best_model.pth
│
├── app.py (Streamlit frontend)
├── run_app.bat (Windows launcher)
├── run_app.sh (Linux/Mac launcher)
├── config.yaml
├── README.md
└── requirements.txt
```

---

# 📊 **8. Model Training Summary**

| Property                   | Value                                |
| -------------------------- | ------------------------------------ |
| Number of Videos           | 500                                  |
| Frames per Video           | 20                                   |
| Total Frames               | 10,000                               |
| Batch Size                 | 4                                    |
| Resolution                 | 128×128                              |
| Training Time (Laptop CPU) | ~1.5 – 4 hours                       |
| Best Accuracy              | Depends on dataset; typically 70–85% |

---

# 🔬 **9. Results & Observations**

* CNN alone struggles with temporal deepfake artifacts
* LSTM significantly boosts detection accuracy
* Model is small enough to run entirely on CPU
* Adding precomputed CNN features drastically speeds up training
* Increasing frame count → better temporal learning
* Next step: apply hybrid CNN + ViT + LSTM or X3D architecture

---

# 🚀 **10. Future Work**

Several enhancements are planned:

* Add Vision Transformer (ViT) for global attention
* Add MobileNet/EfficientNet backbone for improved spatial encoding
* Improve temporal modeling using GRU or Temporal Convolution Networks
* Deploy model as lightweight REST API (FastAPI)
* Expand dataset to 2000+ videos for higher generalization

---

# 🤝 **11. Credits**

This project was developed as part of a Forensic AI research prototype / Final Year Project.

Special thanks to:

* PyTorch Open Source Community
* FaceForensics++ & DFDC datasets
* Vision researchers contributing to spatial–temporal deepfake detection

---

# ❤️ **12. License**

You may use this code for academic, research, and non-commercial purposes.
Proper citation is appreciated when using this work.
