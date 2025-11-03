import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, Model

# ✅ Correct dataset path
DATASET_PATH = "dataset/garbage-dataset"

print("✅ Loading dataset...")

# ✅ Load dataset with correct path
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)

print("✅ Dataset loaded!")
print("✅ Classes found:", train_ds.class_names)

# ✅ Load pretrained MobileNetV2
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # Freeze backbone

# ✅ Add custom layers
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(128, activation="relu")(x)

# ✅ 10 classes (important)
out = layers.Dense(10, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Starting training...")
model.fit(train_ds, epochs=10)

model.save("waste_model.h5")
print("✅ ✅ Training complete! Saved model as waste_model.h5")
