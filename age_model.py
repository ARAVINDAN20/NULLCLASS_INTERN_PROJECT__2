import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Define image directory path
image_dir = "UTKFace"

image_paths = []
ages = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        try:
            age = int(filename.split("_")[0])
            image_paths.append(os.path.join(image_dir, filename))
            ages.append(age)
        except ValueError:
            print(f"unexpected format: {filename}")

if not image_paths:
    raise ValueError("No valid image paths found. Please check the directory contents.")

print(f"Found {len(image_paths)} images and {len(ages)} ages.")

train_paths, val_paths, train_ages, val_ages = train_test_split(image_paths, ages, test_size=0.2, random_state=42)

class AgeDataGenerator(Sequence):
    def __init__(self, image_paths, ages, batch_size=32, target_size=(200, 200)):
        self.image_paths = image_paths
        self.ages = ages
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indices]
        batch_ages = [self.ages[i] for i in batch_indices]
        batch_images = [cv2.resize(cv2.imread(p), self.target_size) / 255.0 for p in batch_paths]
        return np.array(batch_images), np.array(batch_ages)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

train_gen = AgeDataGenerator(train_paths, train_ages)
val_gen = AgeDataGenerator(val_paths, val_ages)

# Define the model
model = Sequential([
    Input(shape=(200, 200, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')  # Age is a continuous variable
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(train_gen, validation_data=val_gen, epochs=10)

model.save("age_model.h5")
