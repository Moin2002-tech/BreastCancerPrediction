#!/usr/bin/env python3
"""
Breast Cancer Prediction Model Visualization
Visualizes performance metrics for PyTorch NN, Scratch NN, and Custom Dataset models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import json
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelVisualizer:
    def __init__(self, data_path="database/data.csv"):
        """Initialize the visualizer with the breast cancer dataset"""
        self.data_path = data_path
        self.data = None
        self.models_data = {}
        self.load_data()
    
    def load_data(self):
        """Load the breast cancer dataset"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Loaded dataset with {len(self.data)} samples")
            print(f"Diagnosis distribution: {self.data['diagnosis'].value_counts()}")
        except FileNotFoundError:
            print(f"Error: Could not find data file at {self.data_path}")
            print("Please ensure the data file exists in the database directory")
    
    def load_model_results(self, results_file="model_results.json"):
        """Load model prediction results from JSON file"""
        try:
            with open(results_file, 'r') as f:
                self.models_data = json.load(f)
            print(f"Loaded results for {len(self.models_data)} models")
        except FileNotFoundError:
            print(f"Error: Could not find results file at {results_file}")
            print("Please run the C++ models first to generate prediction data")
            # Create dummy data for demonstration
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy model results for demonstration"""
        print("Creating dummy data for demonstration...")
        np.random.seed(42)
        
        # Generate dummy predictions based on actual diagnosis
        if self.data is not None:
            n_samples = len(self.data)
            true_labels = (self.data['diagnosis'] == 'M').astype(int)
            
            # Create realistic predictions with some noise
            models = ['PyTorch_NN', 'Scratch_NN', 'Custom_Dataset']
            for model in models:
                # Generate predictions with varying accuracy based on README performance
                if model == 'PyTorch_NN':
                    accuracy = 0.95614
                elif model == 'Scratch_NN':
                    accuracy = 0.92105
                else:  # Custom_Dataset
                    accuracy = 0.947368
                
                # Generate predictions with specified accuracy
                predictions = np.random.binomial(1, accuracy, n_samples)
                # Align some predictions with true labels for realistic performance
                mask = np.random.random(n_samples) < accuracy
                predictions[mask] = true_labels[mask]
                
                # Generate probabilities (confidence scores)
                probabilities = np.random.beta(2, 2, n_samples)
                predictions_binary = (probabilities > 0.5).astype(int)
                predictions_binary[mask] = true_labels[mask]
                
                self.models_data[model] = {
                    'predictions': predictions_binary.tolist(),
                    'probabilities': probabilities.tolist(),
                    'true_labels': true_labels.tolist(),
                    'accuracy': accuracy,
                    'loss': np.random.uniform(0.18, 0.27)
                }
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model_data in self.models_data.items():
            y_true = np.array(model_data['true_labels'])
            y_scores = np.array(model_data['probabilities'])
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Breast Cancer Prediction Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (model_name, model_data) in enumerate(self.models_data.items()):
            y_true = np.array(model_data['true_labels'])
            y_pred = np.array(model_data['predictions'])
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       square=True, cbar=False, ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAccuracy: {model_data["accuracy"]:.3f}', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xticklabels(['Benign', 'Malignant'])
            axes[idx].set_yticklabels(['Benign', 'Malignant'])
        
        plt.suptitle('Confusion Matrices - Model Performance Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_comparison(self):
        """Plot bar charts comparing model performance metrics"""
        model_names = list(self.models_data.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy comparison — normalize any % values to 0-1 range
        accuracies = []
        for model in model_names:
            acc = self.models_data[model]['accuracy']
            if acc > 1.0:          # Scratch_NN bug: stored as 93.86 not 0.9386
                acc = acc / 100.0
            accuracies.append(acc)

        bars1 = ax1.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_ylim(0, 1.15)      # extra headroom for value labels
        ax1.grid(True, alpha=0.3)

        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

        # Loss comparison
        losses = [self.models_data[model]['loss'] for model in model_names]
        bars2 = ax2.bar(model_names, losses, color=['#d62728', '#9467bd', '#8c564b'])
        ax2.set_title('Model Loss Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=10)
        ax2.set_ylim(0, max(losses) * 1.2)   # explicit headroom — no floating labels
        ax2.grid(True, alpha=0.3)

        for bar, loss in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                     f'{loss:.4f}', ha='center', va='bottom', fontsize=10)

        plt.suptitle('Model Performance Metrics Comparison',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_training_curves(self):
        """Plot simulated training accuracy and loss curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = np.arange(1, 101)
        
        for model_name, color in zip(self.models_data.keys(), 
                                   ['blue', 'orange', 'green']):
            # Simulate training curves
            final_acc = self.models_data[model_name]['accuracy']
            final_loss = self.models_data[model_name]['loss']
            
            # Simulate accuracy curve (converging to final accuracy)
            acc_curve = final_acc * (1 - np.exp(-epochs/20)) + np.random.normal(0, 0.02, len(epochs))
            acc_curve = np.clip(acc_curve, 0, 1)
            
            # Simulate loss curve (decreasing to final loss)
            loss_curve = final_loss * np.exp(-epochs/15) + 0.1 + np.random.normal(0, 0.02, len(epochs))
            loss_curve = np.clip(loss_curve, 0, None)
            
            # Plot accuracy curve
            ax1.plot(epochs, acc_curve, label=model_name, color=color, linewidth=2)
            
            # Plot loss curve
            ax2.plot(epochs, loss_curve, label=model_name, color=color, linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Training Accuracy Curves', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Loss', fontsize=10)
        ax2.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Simulated Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("Generating model performance visualizations...")

        self.plot_roc_curves()
        self.plot_confusion_matrices()
        self.plot_performance_comparison()
        self.plot_training_curves()

        print("All visualizations saved as PNG files in assets/ directory")
        print("Files created: assets/roc_curves.png, assets/confusion_matrices.png, assets/performance_comparison.png, assets/training_curves.png")

def main():
    """Main function to run the visualization"""
    print("Breast Cancer Prediction Model Visualizer")
    print("=" * 50)
    
    visualizer = ModelVisualizer()
    visualizer.load_model_results()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
