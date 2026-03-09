import os
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical

# Keras 3 (Python 3.11) compatible legacy import
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# 1. Generator Function (Sahi path ke saath)
def generator(dir, gen=None, shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    if gen is None:
        gen = ImageDataGenerator(rescale=1./255)
    
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )

# 2. Parameters
BS = 32
TS = (24,24)

# 3. Data Loading (Dhyan rahe 'data' folder mein images honi chahiye)
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

# Steps calculation
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(f"Steps per epoch: {SPE}, Validation steps: {VS}")

# 4. CNN Model Architecture (Keras 3 Standard)
model = Sequential([
    Input(shape=(24, 24, 1)), # Naya input layer style
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax') # 2 classes: Open aur Closed
])

# 5. Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Training (fit_generator ki jagah fit use karein)
model.fit(
    train_batch, 
    validation_data=valid_batch, 
    epochs=15, 
    steps_per_epoch=SPE, 
    validation_steps=VS
)

# 7. Model Saving
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/cnnCat2.h5')
print("Model saved successfully in models/cnnCat2.h5")