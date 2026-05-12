import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0

def get_model(model_name: str, num_classes: int, target_size: tuple, learning_rate: float):
    """
    Build and compile a transfer learning model.
    """
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*target_size, 3))
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*target_size, 3))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(*target_size, 3)),
        # Data Augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
        
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
