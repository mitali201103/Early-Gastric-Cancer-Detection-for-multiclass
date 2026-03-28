# 🔬 Early Gastric Cancer Detection — Multi-Class Classification

Deep learning project to detect and classify gastric cancer 
into multiple stages using histopathology images.

---

## 📁 Project Structure
```
Early Gastric Cancer Detection Multi Class/
    ├── efficient_net.py           # EfficientNetB3 transfer learning model
    ├── cnn_inception_attention.py # CNN with Inception + Attention model
    ├── binary_to_multi.py         # Converts binary labels to multi-class
    ├── pre_processing.py          # Data preprocessing & augmentation
    └── README.md
```

> 🚫 Raw images and model weights are excluded via `.gitignore`

---

## 🧠 Models Implemented

### 1. EfficientNetB3 (`efficient_net.py`)
- Transfer learning from ImageNet
- Two-phase training (frozen → fine-tuned)
- Input: 224×224

### 2. CNN + Inception + Attention (`cnn_inception_attention.py`)
- Custom CNN architecture
- Inception modules for multi-scale features
- Attention mechanism for focus on cancer regions

---

## 🎯 Classification Classes

| Class | Stage |
|-------|-------|
| 0 | Normal |
| 1 | Early |
| 2 | Intermediate |
| 3 | Advanced |

---

## 📊 EfficientNetB3 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 90.50% |
| Macro F1 | 0.9940 |

| Class | Accuracy |
|-------|----------|
| Normal | 94.75% |
| Early | 96.75% |
| Intermediate | 81.00% |
| Advanced | 89.50% |

---

## ⚙️ Requirements
```bash
pip install -r requirements.txt
```
```
tensorflow>=2.12
keras>=2.12
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
```

---

## 🚀 How to Run

### Preprocessing
```bash
python pre_processing.py
```

### Train EfficientNetB3
```bash
python efficient_net.py
```

### Train CNN Inception Attention
```bash
python cnn_inception_attention.py
```

---

## 📁 Dataset
- Histopathology images of gastric tissue
- 4 classes, 4000 images per class (balanced)
- Source: Kaggle — `gastric-cancer-data`
> Dataset not included. Download from Kaggle and 
> place in `raw_images/` folder.

---

## 👩‍💻 Author
**Mitali** — [GitHub](https://github.com/mitali201103)

---

## 📜 License
For educational and research purposes only.