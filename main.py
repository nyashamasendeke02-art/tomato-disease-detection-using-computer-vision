import yaml
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dataset paths and parameters
train_dir = config["dataset"]["train"]
val_dir = config["dataset"]["val"]
img_size = config["dataset"]["img_size"]
batch_size = config["dataset"]["batch_size"]

# Data augmentation (simplified for Pi)
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(config["dataset"]["augmentation"]["rotation_range"] / 360),
    layers.RandomZoom(config["dataset"]["augmentation"]["zoom_range"]),
    layers.RandomFlip("horizontal") if config["dataset"]["augmentation"]["horizontal_flip"] else layers.Lambda(lambda x: x)
])

# Data loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.prefetch(buffer_size=config["dataset"].get("prefetch_buffer", AUTOTUNE))
val_ds = val_ds.prefetch(buffer_size=config["dataset"].get("prefetch_buffer", AUTOTUNE))

# Model setup
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights=config["model"]["weights"],
    pooling='avg'
)
base_model.trainable = False  # Freeze base

inputs = layers.Input(shape=(img_size, img_size, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
x = base_model(x, training=False)
x = layers.Dropout(config["model"]["dropout_rate"])(x)
outputs = layers.Dense(config["model"]["num_classes"], activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=config["training"]["optimizer"].lower(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["initial_training"]["epochs"]
)

# Optionally unfreeze and fine-tune
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(config["training"]["fine_tuning"]["learning_rate"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["fine_tuning"]["epochs"]
)

# Save model
model.save("models/tomato_disease_model.h5")