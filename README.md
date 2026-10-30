
# STC-Net

Official implementation of STC-Net for ECG-based classification of Normal, Congestive Heart Failure (CHF), and Coronary Artery Disease (CAD).

## 1. Requirements

- Python 3.9+
- PyTorch
- NumPy
- Pandas
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

```

## 2. Network Architecture

STC-Net consists of three main modules:

- Spatial Representation (SR): Three CRRS blocks with SE attention and residual connections.
- Temporal Perception (TP): A bidirectional LSTM for temporal feature modeling.
- Classification Decision (CD): A fully connected classifier for three-class prediction.

The model contains 569,603 trainable parameters.


