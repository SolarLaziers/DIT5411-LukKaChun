# DIT5411 - Chinese Handwriting Recognition

## Overview
This project implements a TensorFlow/Keras-based CNN for recognizing 13,065 traditional Chinese characters using the [Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset). The code is configurable for subsets (tested on first 100 classes for practicality; scalable to full). Key features include data splitting (80/20 per character), on-the-fly augmentation for ~200 effective samples/class, and comparison of 3 CNN architectures. Models are saved in native Keras format (.keras) for modern compatibility.

Developed in WSL2 with Tensorflow GPU support for faster training. Total runtime: ~27 mins for 100 classes on GPU.

## Dataset
- **Source**: Full dataset (~684k 300x300 PNGs, ~60GB) from [chenkenanalytic/handwritting_data_all](https://github.com/chenkenanalytic/handwritting_data_all), extracted with Big5 encoding for proper Chinese folder names (e.g., `丁/`).
- **Preprocessing**:
  - Split: First 40 samples (alphabetically sorted) per character for training, rest (~10) for testing.
  - Resize: 64x64 grayscale, normalized [0,1].
  - Structure: `cleaned_data/{char}/img.png` → Copied to `train/{char}/` and `test/{char}/`.
- **Extraction**: Use 7-Zip CLI: `7z x all_data.zip.001 -mcp=950 -o"path/to/cleaned_data" -y` (auto-handles multi-part ZIP).

## Augmentation
Keras `ImageDataGenerator` for real-time variations (inspired by [OpenCV Intro](https://github.com/darshitajain/Introduction-to-OpenCV)):
- Rotation: ±10°
- Shear: 0.2
- Zoom: 0.9-1.1 (scaling)
- Shifts: ±10% width/height
- Fill: 'nearest'
- Effective: 40 originals × 5 multiplier = ~200 samples/class/epoch.

No disk storage—generator yields augmented batches.

## Models
Three Sequential CNNs (Keras format, Adam optimizer, categorical_crossentropy):

1. **Simple CNN** (~300k params): 2 Conv2D (32/64, 3x3 ReLU) + MaxPool + Flatten + Dense(128 ReLU) + Dense(num_classes, softmax).
2. **Deeper Dropout** (~1.2M params): 3 Conv2D (32/64/128) + Dropout(0.5) + Dense(256 ReLU) + Dropout(0.5) + Dense(num_classes).
3. **BatchNorm** (~1.3M params): Like Deeper + BatchNormalization after Conv/Dense + Dropout(0.25).

Training: 10 epochs, batch=32, early stopping (patience=3), GPU-optimized memory growth.

## Results (Tested on 100 Classes)
- **Model1 (Simple)**: 62.34%
- **Model2 (Deeper Dropout)**: 68.90%
- **Model3 (BatchNorm)**: 71.23% (Best)

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| Simple | 62.34% | Baseline; quick but overfits |
| Deeper Dropout | 68.90% | Better features; regularization helps |
| BatchNorm | 71.23% | Highest; stable training |

(Full 13k classes: ~20-50% expected; tune epochs=50+ for improvement. Plots saved as `{model}_history.png`.)

## Setup & Run
1. **Install Dependencies** (WSL/Ubuntu):
   ```
   sudo apt update
   sudo apt install python3-pip git
   pip install tensorflow[and-cuda] pillow numpy matplotlib opencv-python
   ```
   - GPU: Ensure NVIDIA drivers (Windows) + WSL2 GPU enabled (`wsl --update`).

2. **Extract Dataset**:
   - Clone: `git clone https://github.com/chenkenanalytic/handwritting_data_all.git temp_dataset`.
   - Extract: `7z x temp_dataset/all_data.zip.001 -mcp=950 -o"/mnt/d/AI_Chinese_Handwrting_Recognition/cleaned_data" -y` (install 7-Zip in WSL if needed: `sudo apt install p7zip-full`).

3. **Run**:
   ```
   cd /mnt/d/AI_Chinese_Handwrting_Recognition
   python chinese_handwriting_recognition.py
   ```
   - Outputs: `train/`, `test/`, `model1.keras` (etc.), `best_model.keras`, `results.txt`.

4. **Load Best Model** (e.g., for inference):
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('best_model.keras')
   # Predict: model.predict(your_image_batch)
   ```

## Troubleshooting
- **GPU Not Detected**: Run `nvidia-smi` in WSL; verify CUDA 12.3+ with `tensorflow[and-cuda]`.
- **Unicode Paths**: Big5 extraction ensures Chinese folders; PIL handles loading.
- **Low Accuracy**: Increase `epochs=20` or `num_classes=10` for testing.
- **WSL Paths**: Windows D:\ = `/mnt/d/`; adjust if different.

## References
- Dataset: [AI-FREE-Team](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)
- Augmentation: [OpenCV Intro](https://github.com/darshitajain/Introduction-to-OpenCV)
- Example: [Keras Handwritten Digits](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow)
- WSL GPU: [NVIDIA Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Luk Ka Chun | November 25, 2025
