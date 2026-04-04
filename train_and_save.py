import json
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

print("Loading data...")
with open('xbox_draw/stroke_data.json', 'r') as f:
    dataset = json.load(f)

char_to_int = {c: i for i, c in enumerate(string.ascii_lowercase)}

sequences = []
labels = []

for item in dataset:
    label = item.get('label')
    seq = item.get('sequence', [])
    if not seq or label not in char_to_int: 
        continue
    
    features = [[pt['x'], pt['y']] for pt in seq]
    sequences.append(features)
    labels.append(char_to_int[label])

X = pad_sequences(sequences, dtype='float32', padding='post', value=0.0)
y = np.array(labels)

print("Data shapes:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building model...")
num_classes = 26
input_shape = (None, 2)

model = Sequential([
    Masking(mask_value=0.0, input_shape=input_shape),
    Conv1D(filters=16, kernel_size=1, activation='relu'),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

print("Saving model...")
model.save('stroke_model.h5')
print("Model saved to stroke_model.h5.")
