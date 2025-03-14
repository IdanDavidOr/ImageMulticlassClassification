# Car Model Classification Using Deep Learning

## Project Overview
A comprehensive deep learning project exploring different approaches to classify car models from images. This project implements and compares three distinct methodologies for fine-grained visual classification of 196 different car models. The project demonstrates:

- Advanced transfer learning techniques with EfficientNetV2L
- Novel application of KNN with deep embeddings
- Custom implementation of ResNet architecture
- Systematic experimentation with model architectures and hyperparameters
- Comprehensive evaluation using multiple performance metrics

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodologies](#methodologies)
- [Results](#results)
- [Technical Details](#technical-details)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Environment Setup

### Google Colab Environment
This project is designed to run in Google Colab, leveraging its GPU capabilities for efficient model training. The notebook includes all necessary setup steps:

1. Google Drive mounting for data persistence
2. Directory structure creation
3. Required package installation
4. GPU runtime configuration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/IdanDavidOr/ImageMulticlassClassification/blob/main/final_project_basics_2025.ipynb)

## Dataset
The Stanford Cars Dataset provides a challenging benchmark for fine-grained visual classification:

- 196 unique car classes (make, model, year)
- 8,152 training images (split into training and validation)
- 8,048 test images
- Diverse image conditions:
  - Various lighting conditions
  - Different angles and perspectives
  - Complex backgrounds
  - Multiple scales and resolutions

## Project Structure
The project is organized to maintain clear separation between data, models, and documentation:

```
Google Drive/
└── [Project Directory]/
    ├── dataset/                  # Image data
    ├── models/                   # Saved models
    ├── final_project.ipynb      # Main notebook
    └── annotations.xlsx         # Class metadata
```
`dataset/` and `models/` directories can be created using designated notebook cells


## Methodologies

### 1. Transfer Learning with EfficientNetV2L
Exploration of transfer learning approaches using EfficientNetV2L as the backbone:

#### Three Strategic Configurations:
1. **Base Architecture**
   - Minimalist approach with single dense layer
   - Moderate regularization
   - Baseline for performance comparison

2. **Enhanced Capacity**
   - Deeper architecture with multiple dense layers
   - Hierarchical feature learning
   - Increased model expressiveness

3. **High Regularization**
   - Aggressive dropout strategy
   - Focus on generalization
   - Overfitting prevention

### 2. KNN with Deep Embeddings
An innovative hybrid approach combining deep learning feature extraction with traditional machine learning:

- Feature extraction using EfficientNetV2L
- Exploration of neighborhood dynamics:
  - Tight neighborhoods (k=3) for precision
  - Balanced approach (k=5) for general cases
  - Broad context (k=10) for robustness

### 3. Custom ResNet Architecture
Implementation of residual networks with systematic depth variation:

- **Architecture Philosophy**:
  - Skip connections for gradient flow
  - Batch normalization for training stability
  - Strategic depth increases

- **Depth Configurations**:
  1. Shallow (3 blocks): Fast and efficient
  2. Medium (5 blocks): Balanced complexity
  3. Deep (7 blocks): Maximum feature hierarchy

## Technical Details

### Data Preprocessing Strategy
Comprehensive image preprocessing pipeline:
- Standardized sizing (224x224)
- Advanced augmentation techniques
- Normalization and standardization
- Efficient data loading and caching

### Training Approach
Carefully crafted training configuration:
- Optimized batch sizes for GPU memory
- Strategic learning rate management
- Multi-metric evaluation
- Performance monitoring and validation

### Performance Optimization
Focus on efficient resource utilization:
- GPU memory management
- Data pipeline optimization
- Strategic model checkpointing
- Training time optimization

## Results
[Coming soon: Comprehensive comparison of model performances, including:
- Accuracy metrics across architectures
- Training dynamics analysis
- Resource utilization comparison
- Error analysis and insights]

## Future Work
Potential areas for exploration and improvement:
- Advanced data augmentation strategies
- Model ensemble techniques
- Architecture optimization
- Performance enhancement methods

## Author
Idan David Or Lavi
Senior Data Scientist

## Acknowledgments
- Stanford Cars Dataset creators
- TensorFlow and Google Colab teams
- Kaggle community

## Hardware Requirements
- GPU: Google Colab (Tesla T4/P100)
- RAM: 12GB+ (Colab provided)
- Storage: Google Drive space for dataset

## Runtime Notes
- The notebook is optimized for Google Colab's GPU runtime
- Training times may vary based on Colab's GPU availability
- Consider using Colab Pro for more consistent GPU access
