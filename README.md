# UNet Architecture For Underwater Image Restoration

## Introduction

This repository contains the code for the paper [UNet Architecture For Underwater Image Restoration](https://arxiv.org/abs/2007.10964).

Proposed architecture used in Thesis [A Statistical Framework For Degraded Underwater Video Generation](thesis.pdf) is also available in this repository.

## Dataset

Two datasets have been used throughout the project. 

First one is [MSRB Dataset](https://paperswithcode.com/dataset/msrb). MSRB Dataset used as a pre-training dataset for the proposed architecture.

Second one is custom data generated by [A Statistical Framework For Degraded Underwater Video Generation](https://github.com/serkansatak/Underwater-Fish-Environment). This dataset is used for fine-tuning the proposed architecture.

### Download

To download MSRB Dataset please run the following command:

```bash
python download_data.py
```

### Preprocessing

Before feeding images to network images resized to 386x386 and normalized to [0, 1] range.

### Requirements

- Python 3.10
- PyTorch 2.0.1
- Torchvision 0.15.2
- Pillow
- Torcheval 0.0.7
- Torchsummary 1.5.1
- Tensorboard 2.14.0
- prettyprinter 0.18.0
- gdown
- tqdm

## Usage

After downloading and setting up the dataset, you can start pre-training by running the following command:

```bash
python main.py
```

All configuration is handled by [config.py](src/config.py) file. You can change the configuration by changing the values in this file.
By changing mode and single_test_file parameters one can test the model on a single image.

## Authors

- [Serkan Şatak](https://github.com/serkansatak)
