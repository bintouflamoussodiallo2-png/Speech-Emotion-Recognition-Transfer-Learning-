# Speech Emotion Recognition via Transfer Learning

A comparative study of CNN-based transfer learning architectures for Speech Emotion Recognition (SER),
benchmarked across two public datasets.

## Objective

This project investigates how well image-based pretrained CNN models (VGG16, ResNet-18/34/50)
can classify human emotions from speech, by treating Mel-Spectrograms as images and leveraging
ImageNet pretrained weights.

We compare:
- **Architectures**: VGG16 vs ResNet-18 vs ResNet-34 vs ResNet-50
- **Datasets**: TESS (Toronto Emotional Speech Set) → RAVDESS

## Why Transfer Learning for Audio?

Mel-Spectrograms are 2D time-frequency representations of audio signals that visually resemble
images. This allows us to repurpose powerful vision models pretrained on ImageNet for audio
classification tasks — a form of cross-domain transfer learning.

## Repository Structure

```
Speech-Emotion-Recognition-Transfer-Learning/
│
├── data/ # Datasets (not tracked by git)
│ ├── TESS/
│ └── RAVDESS/
│
├── src/
│ ├── dataset.py # Dataset classes for TESS and RAVDESS
│ ├── models.py # VGG16, ResNet-18/34/50 with custom heads
│ ├── train.py # Training loop with CLI arguments
│ ├── evaluate.py # Accuracy, confusion matrix
│ └── utils.py # Helper functions
│
├── notebooks/ # Exploratory analysis and visualizations
├── results/ # Training curves, confusion matrices, logs
├── scripts/
│ └── train.sh # SLURM job script (Compute Canada)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Datasets

| Dataset | Speakers | Emotions | Samples |
|---------|----------|----------|---------|
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2 (F) | 7 | 2,800 |
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 24 (M+F) | 8 | 1,440 |

**TESS emotions:** angry, disgust, fear, happy, neutral, pleasant surprise, sad

**RAVDESS emotions:** neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Models

All models are pretrained on ImageNet. Only the final classification head is trained (frozen backbone).

| Model | Parameters | Top-1 ImageNet Acc |
|-------|------------|-------------------|
| VGG16 | 138M | 71.6% |
| ResNet-18 | 11M | 69.8% |
| ResNet-34 | 21M | 73.3% |
| ResNet-50 | 25M | 76.1% |

## Experimental Pipeline
```
Audio (.wav)
│
▼
Mel-Spectrogram (librosa)
│ n_mels=128, sr=16000
▼
3-channel image (128×128)
│ repeated grayscale → RGB
▼
Pretrained CNN (frozen backbone)
│
▼
Custom classification head
│
▼
Emotion label
```


## Installation

```bash
git clone https://github.com/bintouflamoussodiallo2-png/Speech-Emotion-Recognition-Transfer-Learning.git
cd Speech-Emotion-Recognition-Transfer-Learning

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

**Download datasets (Kaggle API required):**
```bash
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess -p data/TESS --unzip
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p data/RAVDESS --unzip
```

**Train a model:**
```bash
python src/train.py --model vgg16 --dataset TESS --data_path data/TESS --epochs 20
python src/train.py --model resnet18 --dataset TESS --data_path data/TESS --epochs 20
python src/train.py --model resnet50 --dataset RAVDESS --data_path data/RAVDESS --epochs 20
```

**On Compute Canada (SLURM):**
```bash
sbatch scripts/train.sh
```

## Results

*Results will be updated as experiments complete.*

| Model | Dataset | Test Accuracy |
|-------|---------|---------------|
| VGG16 | TESS | - |
| ResNet-18 | TESS | - |
| ResNet-34 | TESS | - |
| ResNet-50 | TESS | - |
| VGG16 | RAVDESS | - |
| ResNet-50 | RAVDESS | - |

## Tech Stack

- **Python 3.11**
- **PyTorch** — model training
- **Torchvision** — pretrained models
- **Librosa** — audio feature extraction
- **Scikit-learn** — evaluation metrics
- **Matplotlib** — visualizations
- **Compute Canada (Narval/Nibi)** — GPU training (H100)

## Author

**Bintou Flamousso Diallo**
Applied Computer Science Student — Université de Moncton
[GitHub](https://github.com/bintouflamoussodiallo2-png) · [LinkedIn](https://www.linkedin.com/in/bintou-flamousso-diallo)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.