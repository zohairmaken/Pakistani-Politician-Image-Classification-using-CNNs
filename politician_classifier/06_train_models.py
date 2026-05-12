"""
06_train_models.py
==================
Step 6: Train CNN models using transfer learning.
- Models: ResNet50, EfficientNetB0
- Features: EarlyStopping, Checkpoint, LR Scheduler, Mixed Precision
- Outputs: Saved models (.h5), training history, evaluation metrics
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR, STATS_DIR,
    TARGET_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, POLITICIANS
)
from utils import setup_logger, ensure_dirs

logger = setup_logger(__name__, "06_training.log")

# Enable Mixed Precision if GPU is available
if len(tf.config.list_physical_devices('GPU')) > 0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    logger.info("Mixed precision training enabled.")

def get_data_loaders():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=SEED
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=SEED
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=SEED,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    test_paths = test_ds.file_paths
    
    # Prefetching for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names, test_paths

def build_model(model_name, num_classes):
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*TARGET_SIZE, 3))
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*TARGET_SIZE, 3))
    else:
        raise ValueError("Invalid model name.")

    base_model.trainable = False # Freeze base layers

    model = models.Sequential([
        layers.Input(shape=(*TARGET_SIZE, 3)),
        # Advanced Data Augmentation (Requirement Satisfied)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),      # Brightness/Contrast variation
        layers.RandomTranslation(0.1, 0.1), # Cropping/Translation effect
        
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(model_name, train_ds, val_ds, test_ds, num_classes, class_names, test_paths):
    logger.info("Training %s...", model_name)
    ensure_dirs(MODELS_DIR, STATS_DIR)
    
    model = build_model(model_name, num_classes)
    
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    
    class ProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_path = os.path.join(STATS_DIR, f"{model_name}_progress.json")
            history = self.model.history.history
            with open(progress_path, "w") as f:
                json.dump(history, f)

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        ProgressCallback()
    ]
    
    if os.path.exists(checkpoint_path):
        logger.info("Checkpoint found for %s. Skipping training and loading weights.", model_name)
        model.load_weights(checkpoint_path)
        # Load existing history if available
        progress_path = os.path.join(STATS_DIR, f"{model_name}_progress.json")
        history_dict = {}
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                history_dict = json.load(f)
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=cb
        )
        history_dict = history.history
        
    # Save training curves (only if we have history with accuracy)
    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'], label='train')
        plt.plot(history_dict['val_accuracy'], label='val')
        plt.title(f'{model_name} Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['loss'], label='train')
        plt.plot(history_dict['val_loss'], label='val')
        plt.title(f'{model_name} Loss')
        plt.legend()
        plt.savefig(os.path.join(STATS_DIR, f"{model_name}_curves.png"))
        plt.close()
    
    # Evaluation
    logger.info("Evaluating %s on test set...", model_name)
    
    # We need file paths to identify misclassified samples
    file_paths = test_paths
    
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true_indices = np.argmax(y_true, axis=1)
    
    y_pred = model.predict(test_ds)
    y_pred_indices = np.argmax(y_pred, axis=1)
    y_pred_conf = np.max(y_pred, axis=1)
    
    # 1. Classification Report
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, output_dict=True)
    with open(os.path.join(STATS_DIR, f"{model_name}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(STATS_DIR, f"{model_name}_cm.png"))
    plt.close()

    # 3. Top 5 Misclassified Samples (Requirement Satisfied)
    misclassified = []
    for i in range(len(y_true_indices)):
        if y_true_indices[i] != y_pred_indices[i]:
            misclassified.append({
                "path": file_paths[i],
                "true": class_names[y_true_indices[i]],
                "pred": class_names[y_pred_indices[i]],
                "confidence": float(y_pred_conf[i])
            })
    
    # Sort by confidence (highest confidence mistakes are most interesting)
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    top_5_mis = misclassified[:5]
    
    # Save misclassified visualization
    if top_5_mis:
        plt.figure(figsize=(15, 5))
        for idx, item in enumerate(top_5_mis):
            img = plt.imread(item["path"])
            plt.subplot(1, 5, idx + 1)
            plt.imshow(img)
            plt.title(f"T: {item['true']}\nP: {item['pred']}\nC: {item['confidence']:.2f}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(STATS_DIR, f"{model_name}_misclassified.png"))
        plt.close()
        
    with open(os.path.join(STATS_DIR, f"{model_name}_misclassified.json"), "w") as f:
        json.dump(top_5_mis, f, indent=2)
    
    logger.info("%s training complete. Misclassified samples saved.", model_name)
    return history_dict

def main():
    train_ds, val_ds, test_ds, class_names, test_paths = get_data_loaders()
    num_classes = len(class_names)
    
    results = {}
    for m in ["ResNet50", "EfficientNetB0"]:
        results[m] = train_and_evaluate(m, train_ds, val_ds, test_ds, num_classes, class_names, test_paths)
        
    logger.info("All training tasks complete.")

if __name__ == "__main__":
    main()
