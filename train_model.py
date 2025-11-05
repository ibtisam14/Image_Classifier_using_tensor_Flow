import tensorflow as tf  
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, Model

# ✅ Dataset path
DATASET_PATH = "dataset/garbage-dataset"

print("✅ Loading dataset...")

# ✅ 1. Load Training dataset (80%)
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=123
)

# ✅ 2. Load Validation dataset (20%)
val_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=123
)

print("✅ Classes found:", train_ds.class_names)

# ✅ 3. Data Augmentation (to improve accuracy)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ✅ 4. Load MobileNetV2 Pretrained
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # ✅ Freeze MobileNet weights initially

# ✅ 5. Build model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)  # ✅ prevents overfitting
outputs = layers.Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)

# ✅ 6. Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("✅ Starting training...")

# ✅ 7. Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[callback]
)

# ✅ 8. Fine-tune MobileNet (optional but improves accuracy)
print("✅ Fine-tuning MobileNet...")

base.trainable = True  # unfreeze all MobileNet layers

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[callback]
)

# ✅ 9. Save trained model
model.save("waste_model.h5")

print("✅✅ Training complete! Saved model as waste_model.h5")
