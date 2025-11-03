import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model

DATASET_PATH = "dataset/"  # Your dataset folder

print("✅ Loading dataset...")

train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)

print("✅ Dataset loaded!")

# Load MobileNetV2 (pretrained on ImageNet)
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # Freeze base model

# Add new layers
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(6, activation="softmax")(x)

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
