"""
Medical Image Classification: Pneumonia Detection from Chest X-rays
Complete Training Pipeline with Transfer Learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    # Dataset paths (update these to your local paths)
    DATASET_PATH = 'chest_xray'
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    VAL_PATH = os.path.join(DATASET_PATH, 'val')
    TEST_PATH = os.path.join(DATASET_PATH, 'test')
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    
    # Training parameters
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    
    # Model saving
    MODEL_SAVE_PATH = 'models'
    
    # Class names
    CLASSES = ['NORMAL', 'PNEUMONIA']

config = Config()

# Create directories
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

print("=" * 70)
print("MEDICAL IMAGE CLASSIFICATION - PNEUMONIA DETECTION")
print("=" * 70)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print()


# ============================================================================
# PART 1: DATA PREPROCESSING AND AUGMENTATION
# ============================================================================

def create_data_generators():
    """Create data generators with augmentation for training"""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_PATH,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_PATH,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_PATH,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


# ============================================================================
# PART 2: MODEL ARCHITECTURES
# ============================================================================

def build_resnet_model():
    """Build ResNet50 model with transfer learning"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def build_vgg16_model():
    """Build VGG16 model with transfer learning"""
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def build_efficientnet_model():
    """Build EfficientNetB0 model with transfer learning"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


# ============================================================================
# PART 3: TRAINING FUNCTIONS
# ============================================================================

#def compile_model(model):
#   """Compile model with optimizer and loss"""
#    model.compile(
#        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
#        loss='binary_crossentropy',
#        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
#    )
#    return model

def compile_model(model):
    """Compile model with optimizer and loss"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


def get_callbacks(model_name):
    """Define callbacks for training"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_SAVE_PATH, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def train_model(model, model_name, train_gen, val_gen):
    """Train the model"""
    print(f"\n{'=' * 70}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'=' * 70}\n")
    
    model.summary()
    
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    return history


# ============================================================================
# PART 4: EVALUATION AND METRICS
# ============================================================================

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} Training History', fontsize=16)
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_gen, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{'=' * 70}")
    print(f"EVALUATING {model_name.upper()} ON TEST SET")
    print(f"{'=' * 70}\n")
    
    # Get predictions
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_gen.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=config.CLASSES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc
    }


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Load data
    print("Loading and preparing data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {config.CLASSES}")
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate all models
    models_to_train = {
        'ResNet50': build_resnet_model,
        'VGG16': build_vgg16_model,
        'EfficientNetB0': build_efficientnet_model
    }
    
    for model_name, model_builder in models_to_train.items():
        print(f"\n{'#' * 70}")
        print(f"# {model_name}")
        print(f"{'#' * 70}")
        
        # Build and compile model
        model = model_builder()
        model = compile_model(model)
        
        # Train model
        history = train_model(model, model_name, train_gen, val_gen)
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Evaluate model
        metrics = evaluate_model(model, test_gen, model_name)
        results[model_name] = metrics
        
        # Save final model
        model.save(os.path.join(config.MODEL_SAVE_PATH, f'{model_name}_final.h5'))
        print(f"\n{model_name} saved successfully!")
    
    # Compare all models
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    results_df.to_csv('model_comparison.csv')
    
    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.plot(kind='bar', ax=ax)
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Model: {results_df['accuracy'].idxmax()}")
    print(f"Best Accuracy: {results_df['accuracy'].max():.4f}")


if __name__ == "__main__":
    main()
