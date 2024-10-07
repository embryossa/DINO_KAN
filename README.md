# Neural Network for IVF Image Analysis using DINO Feature Extraction

This repository contains the code to train a neural network model using features extracted from images with DINO (Distillation with No Labels), specifically Vision Transformer (ViT) models. The purpose of the project is to enhance image classification for biomedical images, particularly in IVF research.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Feature Extraction with DINO](#feature-extraction-with-dino)
  - [Training Neural Network on Extracted Features](#training-neural-network-on-extracted-features)
  - [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project utilizes Vision Transformer models pre-trained using DINO to extract high-quality features from images, particularly for complex biomedical images such as those used in IVF (in vitro fertilization). These features are then used to train a neural network (KAN) model for classification tasks.

- **DINO** extracts global contextual features by dividing images into patches and using a transformer-based approach.
- **KAN (Kolmogorov-Arnold Network)** is a neural network architecture that efficiently handles non-linear relationships between features, offering robust performance on the extracted DINO features.

### Why use DINO?
DINO models outperform traditional CNNs in scenarios requiring an understanding of global image context, as they rely on a transformer-based attention mechanism rather than local feature extraction typical for CNNs. In biomedical tasks like embryo image classification, DINO captures global structures and fine-grained details, making it superior for feature extraction in such complex tasks.

## Requirements

To run the code, you need the following libraries:

- Python 3.8+
- PyTorch
- timm (for DINO model support)
- torchvision
- scikit-learn
- argparse
- numpy
- pandas
- matplotlib

You can install the necessary dependencies with:

```bash
pip install torch torchvision timm scikit-learn numpy pandas matplotlib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/embryossa/DINO_KAN.git
   cd ivf-dino-feature-extraction
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is prepared (see Usage for details on dataset requirements).

## Usage

### Feature Extraction with DINO

The feature extraction uses a pretrained DINO model to process the images into feature vectors.

```bash
python extract_features.py --model_name vit_large_patch14_dinov2.lvd142m --dataset_path path/to/images --save_path path/to/save/features.csv
```

**Arguments:**
- `--model_name`: Name of the DINO model from the `timm` library. Default is `vit_large_patch14_dinov2.lvd142m`.
- `--dataset_path`: Path to the image dataset.
- `--save_path`: Path to save the extracted features in CSV format.

**Example:**
```bash
python extract_features.py --model_name vit_large_patch14_dinov2.lvd142m --dataset_path ./data/images --save_path ./data/features.csv
```

### Training Neural Network on Extracted Features

Once the features are extracted, they can be used to train a neural network model. The following script is used for training:

```bash
python train_model.py --features_path path/to/features.csv --labels_path path/to/labels.csv --save_model_path path/to/save/model.pth
```

**Arguments:**
- `--features_path`: Path to the CSV file containing extracted features.
- `--labels_path`: Path to the corresponding labels for classification.
- `--save_model_path`: Path to save the trained model.

**Example:**
```bash
python train_model.py --features_path ./data/features.csv --labels_path ./data/labels.csv --save_model_path ./models/kan_model.pth
```

### Making Predictions

To make predictions on new images, you will need to first extract the features from these new images using the same DINO model, and then pass these features into the trained KAN model.

1. **Extract Features** from new images:
   ```bash
   python extract_features.py --model_name vit_large_patch14_dinov2.lvd142m --dataset_path ./new_images --save_path ./new_features.csv
   ```

2. **Load Model and Predict**:
   ```bash
   python predict.py --model_path ./models/kan_model.pth --features_path ./new_features.csv --threshold 0.5
   ```

**Example Output:**
- The predictions will include the classification score and whether the image is "in-focus" based on the threshold.

## Project Structure

```bash
.
├── data/               # Folder for datasets and features
├── models/             # Folder to save the trained models
├── scripts/            # Feature extraction, training, and prediction scripts
├── extract_features.py # Script for DINO feature extraction
├── train_model.py      # Script for training the neural network
├── predict.py          # Script for running predictions on new data
└── README.md           # Project documentation
```

## Results

During our testing, the neural network trained on DINO-extracted features significantly outperformed traditional CNN-based approaches. The key metrics include:

**Accuracy:** 0.9624
**Precision:** 0.9434
**Recall:** 0.7937
**F1 Score:** 0.8621
**ROC AUC:** 0.9912
**MCC:** 0.8447


The model successfully identified whether the embryo images were "in-focus," essential for IVF image analysis tasks.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with detailed changes and testing steps.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

https://www.linkedin.com/feed/update/urn:li:activity:7249107731238780928/

### Models Used
1. **DINO (Self-Supervised Learning via Knowledge Distillation)**  
   We used the DINO model for feature extraction.
   > @inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
You can find more about DINO and access the pre-trained models via the [official DINO repository](https://github.com/facebookresearch/dino).

3. **KAN: Kolmogorov-Arnold Networks**  
   We implemented the KAN model as described in the following paper:  
   > Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks." *arXiv preprint* arXiv:2404.19756 (2024).  
   You can find the implementation of KAN used in this project on [GitHub](https://github.com/rotem154154/kan_classification).

### Datasets Used
3. **Blastocyst Dataset**  
   For testing the model on independent data, we used the Blastocyst Dataset. You can access the dataset at the [following link](https://github.com/software-competence-center-hagenberg/Blastocyst-Dataset).

### Acknowledgments
We acknowledge the authors of the DINO, KAN models, and the Blastocyst Dataset for their contributions and making their work available to the community. All rights to the mentioned models and datasets remain with their respective authors.


