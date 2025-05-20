# Comparison of CNN and Vision Transformer (ViT) on CIFAR-10

## Overview
This project compares the performance of a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) on the CIFAR-10 dataset using a GPU T4 on Google Colab. The goal was to evaluate which model performs better in terms of accuracy, training time, and resource usage for a small dataset like CIFAR-10.

## Dataset
- **CIFAR-10**: 50,000 training images and 10,000 test images, each 32x32 pixels, across 10 classes.

## Models
1. **CNN**:
   - Architecture: A simple CNN with convolutional and pooling layers.
   - Parameters: 154,462
2. **ViT**:
   - Architecture: Vision Transformer with patch size 4, embed_dim 192, num_heads 8, and num_layers 2.
   - Parameters: 519,882

## Results (Using GPU T4)
| Model | Accuracy | Epochs | Training Time (GPU T4) | Parameters |
|-------|----------|--------|-----------------------|------------|
| CNN   | 71.63%    | 5      | ~3  minutes          | 154,462    |
| ViT   | 58.12%   | 70     | ~45 minutes          | 519,882    |
| ViT   | 48.33%   | 100    | ~1 hour              | 519,882    |

### Key Findings
- **CNN outperformed ViT** on CIFAR-10 with GPU T4, achieving 71.63% accuracy with significantly less training time (5 epochs) and fewer parameters.
- **ViT showed potential** with 58.12% accuracy at 70 epochs, but performance dropped to 48.33% at 100 epochs, possibly due to overfitting or suboptimal tuning (e.g., Mixup settings).
** Conclusion **: For small datasets like CIFAR-10 with GPU T4, CNNs are more efficient and effective. ViT demonstrated potential at 70 epochs (58.12%) but requires further optimization or larger datasets with pretraining to outperform CNN.

## Setup
### Requirements
Install the required packages:

pip install -r requirements.txt

Running the Code

   CNN Model:
    
      python cnn_model.py

   ViT Model:

      python vit_model.py

Files

    cnn_model.py: Code for training the CNN model.
    vit_model.py: Code for training the ViT model.
    results/: Contains the output logs for both models.

Future Work

    Test ViT on larger datasets (e.g., ImageNet) with pretraining.
    Explore hybrid models combining CNN and Transformer architectures.

Contributors

    ShaaheeN
