import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.helpers import setup_logger, ensure_dirs
from src.models.architectures import get_model

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("trainer", config, "training.log")
        self.paths = config['paths']
        self.settings = config['training']
        self.proc_settings = config['processing']
        
        ensure_dirs(self.paths['models_dir'], self.paths['reports_dir'])

    def get_data_loaders(self):
        loader_kwargs = {
            "image_size": tuple(self.proc_settings['target_size']),
            "batch_size": self.settings['batch_size'],
            "label_mode": 'categorical',
            "seed": self.settings['seed']
        }
        
        train_ds = tf.keras.utils.image_dataset_from_directory(self.paths['train_dir'], **loader_kwargs)
        val_ds = tf.keras.utils.image_dataset_from_directory(self.paths['val_dir'], **loader_kwargs)
        test_ds = tf.keras.utils.image_dataset_from_directory(self.paths['test_dir'], shuffle=False, **loader_kwargs)
        
        class_names = train_ds.class_names
        return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), test_ds.prefetch(tf.data.AUTOTUNE), class_names

    def train(self, model_name: str):
        train_ds, val_ds, test_ds, class_names = self.get_data_loaders()
        num_classes = len(class_names)
        
        self.logger.info(f"Building {model_name}...")
        model = get_model(model_name, num_classes, tuple(self.proc_settings['target_size']), self.settings['learning_rate'])
        
        checkpoint_path = os.path.join(self.paths['models_dir'], f"{model_name}_best.h5")
        
        cb = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        self.logger.info(f"Starting training for {model_name}...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.settings['epochs'],
            callbacks=cb
        )
        
        self.evaluate(model, test_ds, class_names, model_name)
        return history.history

    def evaluate(self, model, test_ds, class_names, model_name):
        self.logger.info(f"Evaluating {model_name}...")
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_true_indices = np.argmax(y_true, axis=1)
        
        y_pred = model.predict(test_ds)
        y_pred_indices = np.argmax(y_pred, axis=1)
        
        # Save Report
        report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, output_dict=True)
        report_path = os.path.join(self.paths['reports_dir'], f"{model_name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        # Confusion Matrix
        cm = confusion_matrix(y_true_indices, y_pred_indices)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(os.path.join(self.paths['reports_dir'], f"{model_name}_cm.png"))
        plt.close()

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    trainer = Trainer(cfg)
    trainer.train("EfficientNetB0")
