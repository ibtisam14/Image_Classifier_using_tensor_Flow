import tensorflow as tf   # import TensorFlow
from tensorflow.keras.preprocessing import image_dataset_from_directory # for loading image dataset
from tensorflow.keras import layers, Model # for building model

# Dataset path
DATASET_PATH = "dataset/garbage-dataset" # path to dataset

print("✅ Loading dataset...") # indicate dataset loading

# 1. Load Training dataset (80%)
train_ds = image_dataset_from_directory( # load training dataset
    DATASET_PATH, # path to dataset
    image_size=(224, 224), # resize images to 224x224
    batch_size=32, # batch size
    label_mode="categorical", # categorical labels
    validation_split=0.2, # 20% for validation
    subset="training", # training subset 
    seed=123 # random seed
)

# 2. Load Validation dataset (20%)
val_ds = image_dataset_from_directory(  # load validation dataset
    DATASET_PATH, # path to dataset
    image_size=(224, 224), # resize images to 224x224
    batch_size=32, # batch size
    label_mode="categorical", # categorical labels
    validation_split=0.2, # 20% for validation
    subset="validation", # validation subset
    seed=123 # random seed
)

print("✅ Classes found:", train_ds.class_names) # print class names

# 3. Data Augmentation (to improve accuracy)
data_augmentation = tf.keras.Sequential([ # data augmentation layers
    layers.RandomFlip("horizontal"), # horizontal flip
    layers.RandomRotation(0.1), # small rotation
    layers.RandomZoom(0.1), # small zoom
    layers.RandomContrast(0.1), # small contrast change
])

# 4. Load MobileNetV2 Pretrained
base = tf.keras.applications.MobileNetV2( # load MobileNetV2 model
    input_shape=(224, 224, 3), # input shape
    include_top=False, # exclude top layers
    weights="imagenet" # use ImageNet weights
)
base.trainable = False   # freeze base model

# 5. Build model
inputs = tf.keras.Input(shape=(224, 224, 3)) # input layer
x = data_augmentation(inputs) # apply data augmentation
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # preprocess input for MobileNetV2
x = base(x) # pass through base model
x = layers.GlobalAveragePooling2D()(x) # global average pooling
x = layers.Dense(128, activation="relu")(x) # dense layer
x = layers.Dropout(0.3)(x)  # dropout for regularization 
outputs = layers.Dense(10, activation="softmax")(x) # output layer for 10 classes

model = Model(inputs, outputs) # create model

# 6. Compile model
model.compile( # compile the model
    optimizer="adam", # use Adam optimizer
    loss="categorical_crossentropy", # categorical crossentropy loss
    metrics=["accuracy"] # track accuracy metric
) 

#  Early stopping
callback = tf.keras.callbacks.EarlyStopping( # early stopping callback
    monitor="val_loss", # monitor validation loss
    patience=3, # stop after 3 epochs of no improvement
    restore_best_weights=True # restore best weights
)

print("✅ Starting training...") # indicate start of training

#  7. Train the model
model.fit( # train the model
    train_ds, # training dataset
    validation_data=val_ds, # validation dataset
    epochs=20, # maximum 20 epochs
    callbacks=[callback] # early stopping callback
)

# 8. Fine-tune MobileNet (optional but improves accuracy)
print("✅ Fine-tuning MobileNet...") # indicate fine-tuning

base.trainable = True  # unfreeze all MobileNet layers

model.compile( # re-compile the model
    optimizer=tf.keras.optimizers.Adam(1e-5), # lower learning rate
    loss="categorical_crossentropy", # categorical crossentropy loss
    metrics=["accuracy"] # track accuracy metric
)

model.fit( # continue training
    train_ds, # training dataset
    validation_data=val_ds, # validation dataset
    epochs=10, # additional 10 epochs
    callbacks=[callback] # early stopping callback
)

#  9. Save trained model
model.save("waste_model.h5") # save model to file

print("✅✅ Training complete! Saved model as waste_model.h5") # indicate training completion
