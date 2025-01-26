# Face Mask Detection using Deep Learning 😷

A real-time face mask detection system built using Deep Learning and Computer Vision. This project can detect whether a person is wearing a face mask or not from images or video streams.

## 🎯 Features
- Real-time face mask detection
- Custom CNN architecture optimized for mask detection
- Training progress visualization
- High accuracy on mask/no-mask classification
- Easy-to-use training pipeline

## 🛠️ Tech Stack
- Python 3.x
- TensorFlow 2.x
- OpenCV
- Keras
- NumPy
- Matplotlib

## 📁 Project Structure

Structure of the project:
face-mask-detection/
├── datasets/
│ ├── train/
│ │ ├── with_mask/
│ │ └── without_mask/
│ └── validation/
│ ├── with_mask/
│ └── without_mask/
├── Mask.py # Model architecture and utilities
├── train.py # Training script
├── requirements.txt # Project dependencies
└── README.md

## 🚀 Installation

1. Clone the repository

bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training
1. Organize your dataset in the following structure:
```
datasets/
    train/
        with_mask/
        without_mask/
    validation/
        with_mask/
        without_mask/
```

2. Start training:
```bash
python train.py
```

3. The trained model will be saved as 'mask_detector_model.h5'

## 🧠 Model Architecture
- Input: 224x224x3 RGB image
- Convolutional Neural Network (CNN):
  - 3 Convolutional blocks with MaxPooling
  - Dense layers with Dropout
  - Binary classification output

## 📊 Results
The model provides real-time visualization of:
- Training vs Validation Accuracy
- Training vs Validation Loss

## 👥 Author
Harahad Kaushal
- GitHub: [@yourusername](https://github.com/HarshadKaushal)

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments
- Dataset source: [Add source if applicable]
- Inspiration: [Add any inspirations]
- References: [Add any references]

## 📧 Contact
Your Name - your.email@example.com

Project Link: [https://github.com/HarshadKaushaal/FaceMask-Detection-using-Deep-Learning](https://github.com/HarshadKaushal/FaceMask-Detection-using-Deep-Learning)

---

⭐️ If you found this project helpful, please give it a star!