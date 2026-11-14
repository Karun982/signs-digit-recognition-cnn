# Hand Sign Classification: CNN vs. ResNet50 Transfer Learning

Image classification of hand signs (classes 0–5) with TensorFlow/Keras. The project compares a small CNN trained from scratch against a pretrained ResNet50 used as a frozen feature extractor.

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Baseline CNN (from scratch) | 89.17% | 0.2925 |
| ResNet50 (transfer learning, frozen base) | **95.83%** | 0.1531 |

Transfer learning improved test accuracy by about 6.7 percentage points.

### Baseline CNN: per-class performance

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| 0 | 1.00 | 0.95 | 0.97 |
| 1 | 1.00 | 0.90 | 0.95 |
| 2 | 0.83 | 0.95 | 0.88 |
| 3 | 0.90 | 0.95 | 0.93 |
| 4 | 0.93 | 0.65 | 0.76 |
| 5 | 0.76 | 0.95 | 0.84 |

Class 4 is the weakest for the baseline (recall 0.65). The confusion matrix in the notebook shows where those samples are being misclassified.

> **Note:** The test set has only 120 images (20 per class), so one image changes accuracy by about 0.8 points. Treat the gap between models as indicative, not definitive.

## Dataset

- **Source:** [Signs Detection Dataset on Kaggle](https://www.kaggle.com/datasets/maneesh99/signs-detection-dataset)
- **Format:** HDF5 (`Signs_Data_Training.h5`, `Signs_Data_Testing.h5`)
- **Size:** 1,080 training images and 120 test images
- **Image shape:** 64 × 64 × 3 (RGB)
- **Classes:** 6 (labels 0–5)

The dataset is downloaded automatically in the notebook with `kagglehub`. Please check the dataset page for its license and citation requirements. This dataset is also widely used in deep learning coursework.

## Approach

1. **Load data** from the HDF5 files and inspect shapes and labels.
2. **Preprocess:** scale pixel values to the 0–1 range for the baseline CNN.
3. **Split:** 80% train / 20% validation (stratified, `random_state=42`). The test set is held out and used only for final evaluation.
4. **Baseline CNN:**
   - Two `Conv2D` + `MaxPooling2D` blocks (32 and 64 filters)
   - `Flatten` → `Dense(128)` → `Dense(6, softmax)`
   - Adam optimizer, sparse categorical cross-entropy
   - Early stopping on validation loss (patience 5, best weights restored)
5. **ResNet50 transfer learning:**
   - ImageNet-pretrained ResNet50 (`include_top=False`), base layers frozen
   - `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(6, softmax)`
   - Inputs prepared with ResNet's own `preprocess_input`
   - Early stopping on validation loss (patience 5)
6. **Evaluate** both models on the test set with accuracy, loss, a classification report, and a confusion matrix.

## Getting Started

### Requirements

- Python 3.9+
- TensorFlow / Keras
- h5py, NumPy, matplotlib, scikit-learn
- kagglehub

```bash
pip install tensorflow h5py numpy matplotlib scikit-learn kagglehub
```

### Run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook CNN1-2.ipynb
```

You can also upload the notebook to Google Colab and run all cells. A GPU runtime speeds up the ResNet50 training.

## Repository Structure

```
.
├── CNN1-2.ipynb    # Full notebook: data, models, evaluation
└── README.md
```

## Possible Improvements

- Unfreeze the top ResNet50 blocks and fine-tune with a low learning rate
- Add data augmentation (small rotations, shifts, zoom)
- Set global random seeds for fully reproducible runs
- Try other pretrained backbones (EfficientNet, MobileNet)
- Use cross-validation, since the test set is small

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · NumPy · Matplotlib · h5py

## License

Add your preferred license here (for example, MIT). Dataset usage is subject to the license on its Kaggle page.
