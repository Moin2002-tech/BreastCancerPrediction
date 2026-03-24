# Breast Cancer Prediction Model Visualization Guide

This guide shows how to visualize the performance of your three breast cancer prediction models using the visualization pipeline.

## Overview

The visualization system provides comprehensive performance analysis for:
- **PyTorch Neural Network** (NNModels.cpp)
- **Scratch Neural Network** (NNfromScratch.cpp) 
- **Custom Dataset Model** (NNCustumDatasets.cpp)

## Generated Visualizations

### 1. ROC Curves
- Shows the trade-off between true positive rate and false positive rate
- Includes AUC (Area Under Curve) scores for each model
- Higher AUC indicates better model performance

### 2. Confusion Matrices
- Visual representation of model predictions vs actual values
- Shows true positives, true negatives, false positives, false negatives
- Includes accuracy scores for each model

### 3. Performance Comparison Bar Charts
- Side-by-side comparison of model accuracy
- Loss comparison across all models
- Easy identification of best performing model

### 4. Training Curves
- Simulated training accuracy over epochs
- Training loss progression
- Shows convergence behavior for each model

## Setup Instructions

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build and Run Model Export
```bash
# Build the project
mkdir build && cd build
cmake ..
make

# Run the export utility to generate model results
./BreastCancerPrediction -r=xml -ts=exportData
```

### Step 3: Generate Visualizations
```bash
# Run the visualization script
python visualize_models.py
```

## Output Files

After running the visualization script, you'll get these PNG files:

- `roc_curves.png` - ROC curves with AUC scores
- `confusion_matrices.png` - Confusion matrices for all models
- `performance_comparison.png` - Bar charts comparing metrics
- `training_curves.png` - Training progress visualization

## Data Files

- `model_results.json` - Exported model predictions and metrics
- Generated automatically by running the export test case

## Example Usage

### Quick Start
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Generate model data
./BreastCancerPrediction -r=xml -ts=exportData

# Create visualizations
python visualize_models.py
```

### Custom Visualization
You can modify `visualize_models.py` to:
- Change color schemes
- Add custom metrics
- Export to different formats
- Create interactive plots

## Understanding the Visualizations

### ROC Curve Interpretation
- **Top-left corner**: Perfect classifier
- **Diagonal line**: Random classifier
- ** closer to top-left**: Better performance

### Confusion Matrix Interpretation
- **True Positive (TP)**: Correctly identified malignant
- **True Negative (TN)**: Correctly identified benign  
- **False Positive (FP)**: Benign incorrectly classified as malignant
- **False Negative (FN)**: Malignant incorrectly classified as benign

### Performance Metrics
- **Accuracy**: Overall correct prediction percentage
- **Loss**: Training objective value (lower is better)
- **AUC**: Probability that model ranks random positive higher than random negative

## Troubleshooting

### Common Issues

1. **Missing model_results.json**
   - Run `./BreastCancerPrediction -r=xml -ts=exportData` first to generate the data
   - Ensure the C++ models compile and run successfully

2. **Python import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility (3.7+ recommended)

3. **Display issues**
   - Ensure matplotlib backend is configured
   - Try running in a Jupyter notebook for better interactivity

4. **Build errors**
   - Check all dependencies are properly installed
   - Verify CUDA paths in CMakeLists.txt are correct

### Performance Tips

- For large datasets, consider reducing the number of epochs
- Use GPU acceleration if available for faster model training
- Save intermediate results to avoid retraining

## Advanced Features

### Adding Custom Metrics
Modify the `exportModelResults()` function to include:
- Precision and recall
- F1-score
- Specificity and sensitivity
- Calibration curves

### Real-time Visualization
For real-time training visualization:
1. Modify training loops to export intermediate results
2. Update visualization script to read incremental data
3. Use matplotlib animation features

### Web Dashboard
Convert to a web-based dashboard using:
- Flask/FastAPI backend
- Plotly for interactive charts
- Bootstrap for responsive layout

## Model Performance Summary

Based on your README.md:
- **PyTorch NN**: 95.614% accuracy, 0.2664 loss
- **Scratch NN**: 92.105% accuracy, 0.218185 loss  
- **Custom Dataset**: 94.7368% accuracy, 0.187598 loss

The visualizations will help you understand these metrics in detail and compare model behaviors across different evaluation criteria.
