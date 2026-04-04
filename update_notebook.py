import json

def add_code_cell(notebook_path, source):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # prepare source as list of strings
    source_lines = [line + '\n' for line in source.split('\n')]
    if source_lines:
        source_lines[-1] = source_lines[-1].strip('\n') # remove last newline

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }
    
    nb['cells'].append(new_cell)
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

data_prep_code = """import json
import numpy as np
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

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

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""

train_code = """history = cnn_lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)"""

save_code = """cnn_lstm_model.save('stroke_model.h5')
print("Model saved to stroke_model.h5")"""

add_code_cell('model.ipynb', data_prep_code)
add_code_cell('model.ipynb', train_code)
add_code_cell('model.ipynb', save_code)

print("Notebook updated successfully.")
