# Flower Image Classification using CNN

A complete TensorFlow/Keras project that classifies flower images into five categories using a Convolutional Neural Network (CNN), evaluates model performance, and provides an interactive Streamlit application.

## Classes

The TensorFlow `flower_photos` dataset contains:
- daisy
- dandelion
- roses
- sunflowers
- tulips

The dataset contains approximately 3,670 images. It is downloaded automatically when training starts.

## Project structure

```text
Flower-CNN-Classification/
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── artifacts/                 # Generated after training
├── data/                      # Downloaded dataset (not committed)
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 1. Requirements

- Python 3.10–3.12 recommended
- pip
- Internet connection for the first dataset download
- Optional: NVIDIA GPU with a compatible TensorFlow setup

Create and activate a virtual environment.

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Train the CNN

From the repository root:

```bash
python src/train.py
```

Optional settings:

```bash
python src/train.py --epochs 30 --batch-size 32
```

Training automatically downloads and extracts the dataset into `data/`, uses an 80/20 training-validation split, and saves the best model and evaluation outputs in `artifacts/`.

## 3. Evaluate

Evaluation runs automatically after training. The following files are generated:

| File | Description |
|---|---|
| `best_flower_model.keras` | Best model checkpoint by validation accuracy |
| `class_names.json` | Class-name mapping |
| `metrics.json` | Validation loss and accuracy |
| `classification_report.json` | Per-class precision, recall and F1-score |
| `accuracy.png` | Training and validation accuracy |
| `loss.png` | Training and validation loss |
| `confusion_matrix.png` | Validation confusion matrix |

Metrics are produced by your own training run; do not claim a fixed accuracy before measuring it.

## 4. Run the Streamlit application

Train the model first, then run:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. Upload a JPG, JPEG, PNG, or WEBP flower image to see the predicted class, confidence, and class probabilities.

## 5. Predict from Python

```python
from src.predict import predict_image

label, confidence, probabilities = predict_image("my_flower.jpg")
print("Prediction:", label)
print("Confidence:", confidence)
print("All probabilities:", probabilities)
```

The `src` directory includes `__init__.py`, so the import works from the repository root.

## Model architecture

- Rescaling
- Random horizontal flip and rotation
- Convolution + ReLU (32 filters)
- Max pooling
- Convolution + ReLU (64 filters)
- Max pooling
- Convolution + ReLU (128 filters)
- Max pooling
- Dropout
- Fully connected layer (256 units)
- Softmax output (5 classes)

The model uses Adam, sparse categorical cross-entropy, early stopping, model checkpointing, and learning-rate reduction.

## Dataset attribution

Dataset: TensorFlow flower photos example dataset.

Original archive:
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

The dataset is downloaded at runtime and excluded from Git to keep the repository lightweight. Review the dataset's source and applicable terms before redistribution.

## Troubleshooting

- **TensorFlow installation fails:** use Python 3.10–3.12 and install in a fresh virtual environment. Check TensorFlow's official installation compatibility guidance for your operating system.
- **Streamlit says model is missing:** run `python src/train.py` successfully before starting the app.
- **Out-of-memory:** reduce the batch size, for example `python src/train.py --batch-size 16`.
- **PowerShell activation blocked:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal, then activate the environment.

## Future improvements

- Transfer learning with MobileNetV2 or EfficientNet
- Add a held-out test set
- Add prediction history and downloadable results
- Deploy the Streamlit app to Streamlit Community Cloud

## Skills demonstrated

Python, TensorFlow, Keras, CNN, image classification, image preprocessing, data augmentation, model evaluation, scikit-learn, Matplotlib, Streamlit, Git and GitHub.
