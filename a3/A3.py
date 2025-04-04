# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_history(history, title_prefix="Model"):
    # Plot 1: Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} - Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title_prefix} - Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# Assuming you've already loaded x_train and y_train from CSVs
# Reshape x data into (num_samples, 28, 28, 1) and normalize
x_train = pd.read_csv('a3/datasets/x_train.csv').values
x_test = pd.read_csv('a3/datasets/x_test.csv').values
y_train = pd.read_csv('a3/datasets/y_train.csv').values
y_test = pd.read_csv('a3/datasets/y_test.csv').values

#normalize pixel data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding (needed for categorical crossentropy)
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)


# Define the model
model1 = Sequential([
    # First convolutional layer: 32 filters of size 3x3, stride 1, padding='same'
    Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),

    # Max pooling reduces size from 28x28 to 14x14
    MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional layer
    Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    
    # Flatten: turn 3D tensor (14, 14, 32) into 1D vector
    Flatten(),

    # Dense layer with 512 neurons and ReLU activation
    Dense(512, activation='relu'),

    # Final output layer: 5 neurons for 5 classes, softmax turns output into probabilities
    Dense(5, activation='softmax')
])

# Define the model
model2 = Sequential([
    # First convolutional layer: 32 filters of size 3x3, stride 1, padding='same'
    Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),

    # Max pooling reduces size from 28x28 to 14x14
    MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional layer
    Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    
    # Third convolutional layer for Q2
    Conv2D(32, (5, 5), strides=(1,1), activation='relu', padding='same'),
    
    # Flatten: turn 3D tensor (14, 14, 32) into 1D vector
    Flatten(),

    # Dense layer with 512 neurons and ReLU activation
    Dense(512, activation='relu'),

    # Final output layer: 5 neurons for 5 classes, softmax turns output into probabilities
    Dense(5, activation='softmax')
])


#Q1
model1.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model1.summary()

# Train the model
history1 = model1.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)


#Q2
model2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model2.summary()

# Train the model
history2 = model2.fit(x_train, y_train, epochs=6, batch_size=64, validation_split=0.1)


# For Q1
plot_training_history(history1, title_prefix="Q1 (SGD)")

# For Q2
plot_training_history(history2, title_prefix="Q2 (Adam)")