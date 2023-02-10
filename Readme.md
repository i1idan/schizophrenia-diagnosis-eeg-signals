# Schizophrenia Diagnosis via FFT and Wavelet Convolutional Neural Networks utilizing EEG signals
This is the official implementation of the paper.

# Abstract
## Background:
> Schizophrenia is a chronic mental illness in which a person’s perception of realit
y is distorted. Early diagnosis can help to manage symptoms and increase long-term treatment. The electroencephalogram (EEG) is now used to diagnose specific mental disorders.
## Method:
> In this paper, we developed an artificial intelligence methodology built on deep c
onvolutional neural networks with specialized layers to detect schizophrenia from EEG signals directly, recordings include 14 paranoid schizophrenia pat
ients (7 females) with ages ranging from 27 to 32 and 14 normal subjects (7 females) with ages ranging from 26 to 32. In the first phase, we used the Gr
amian Angular Field (GAF), including two methods: the Gramian Angular Summation Field (GASF) and the Gramian Angular Difference Field (GADF) to represen
t the EEG signals as various types of images. Then, well-known CNN architectures, namely Transformer and CNN-LSTM, are applied in addition to two new cu
stom architectures. These models utilize two-dimensional Fast Fourier transform layers (CNN-FFT) and wavelet layers (CNN-Wavelet) to extract useful info
rmation from the data. These layers allow automated feature extraction from EEG representation in the time and frequency domains. Ultimately these models were evaluated using common metrics such as accuracy, sensitivity and specificity.
## Results: 
> CNN-FFT and Transformer models derive the most effective features from signals bas
ed on the findings. CNN-FFT obtained the highest accuracy of 99.00 percent. The transformer, which has a 98.32 percent accuracy rate, also performs admirably.
## Conclusion:
> This experiment outperformed other previous studies. Consequently, the strategy can aid medical practitioners in the automated detection and early treatment of schizophrenia.

# Installation:
```
pip install -r requirements.txt
```

# Download data
```
gdown --id 1jnWHWrArzJQIvny0cQkfPP42hEJAp_56
mv DATA.mat data/DATA.mat
```

# Train a model for once:
```
python train.py --model-name FFTCustom --epochs 200 --seed 1234 --data-path ./data/DATA.mat
```
For further information about the input arguments run `python train.py -h` or see `train.py`'s contents.

# To train a model more than once:
 Open up the `train.ipynb` notebook

# Try on colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/i1idan/schizophrenia-diagnosis-eeg-signals/blob/main/train.ipynb)

