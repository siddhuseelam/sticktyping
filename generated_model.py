

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Conv1D

num_classes = 26
input_shape = (None, 2) # (Variable sequence length, 2 features)

def create_cnn_lstm_model():
    model = Sequential([
        # 1. Masking Layer (Ignores trailing zeros used for padding)
        Masking(mask_value=0.0, input_shape=input_shape),

        # 2. 1x1 CNN Layer for Feature Expansion
        # kernel_size=1 means it looks at one time step at a time.
        # It expands your 2 features (x,y) into 16 learned features.
        Conv1D(filters=16, kernel_size=1, activation='relu'),
        
        # 3. First LSTM Layer
        LSTM(64, return_sequences=True),
        Dropout(0.2),

        # 4. Second LSTM Layer
        LSTM(64),
        Dropout(0.2),

        # 5. Dense Classifier
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Instantiate and view the new architecture
cnn_lstm_model = create_cnn_lstm_model()
cnn_lstm_model.summary()



