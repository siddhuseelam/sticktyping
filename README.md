# StickTyping: Dual-Stick AI Keyboard 🎮⌨️

Welcome to **StickTyping**, an experimental unistroke smart-keyboard that brings fast and fluid text input to game controllers using a custom deep-learning hybrid engine! By combining sequential Neural Network classification with English Dictionary Context, you can naturally sketch letters and type words using only two thumbsticks.

![Typing Demo](./screenrecording-2026-04-05_02-29-13.mp4)

---

## 🚀 How It Works

This project functions as a standalone ecosystem split across data-collection, offline-training, and live prediction loops. 

1. **Draw with Right Stick**: Tracing the rough shape of a letter on your right stick generates a geometric `x`, `y` spatial array sequence.
2. **AI Inference (`backend.py`)**: As soon as the stick rests to the center deadzone, the engine dynamically resamples the captured vector into a fixed length of 60 points, extracts `dx`/`dy` momentum derivatives, and predicts the character against a `Bidirectional LSTM` (Keras) model trained to understand unistrokes.
3. **Context Fusion**: The Neural Network's top probabilities are blended intelligently via an `ALPHA` parameter with a pre-loaded 10,000-word trie-based dictionary probability map. (Is "t-h-_" more likely to be an *'e'* or a *'q'*?)
4. **Flick to Select (Left Stick)**: The top 4 final candidate characters are mapped to the D-Pad/Left-Stick radial menu. Flicking the Left Stick instantly enters the character!

---

## 🛠 Project Architecture

- **`xbox_draw/collector.py`**: A PyGame utility that lets you connect an Xbox-style controller to manually build arrays of training data (`stroke_data.json`).
- **`train_and_save.py`** & **`model.ipynb`**: Keras-based deep-learning pipelines that take the raw stroke `.json` data, geometrically scale and normalize the distances, compute derivative offsets, and train complex Sequence-to-Class Bidirectional LSTMs mapping `[num_points, 4] (x, y, dx, dy)` inputs to `softmax` prediction nodes.
- **`backend.py`**: The headless Hybrid Core. It merges the neural classifications loaded from `unistroke_hybrid_model.keras` with PyGtrie's frequency distribution logic structure. Adjust the `ALPHA` constant to tweak how much gravity the raw neural predictions have over contextual auto-correct!
- **`type.py`**: The primary Frontend UI and Event-Loop application built using PyGame. Uses Dual Sticks and Bumpers (`LB = Backspace`, `RB = Space`) to simulate a rapid text-typing ecosystem. Contains a visual Debug window for inspecting metric fusion arrays in real time.

## 💾 Running the project natively

### Requirements
You will need an environment utilizing:
- `pygame`, `numpy`, `scipy`
- `tensorflow`
- `pygtrie`, `wordfreq`

### Hardware
* Any standard XInput (Xbox Series, One, 360) or general Generic Gamepad that supplies 2 thumbsticks.

### 1. Launching the Type Platform:
Make sure your controller is connected before launch!
```bash
python3 type.py
```

### 2. Updating the Artificial Intelligence model:
If you capture custom stroke datasets inside `xbox_draw/`, re-train the underlying recognizer by spinning up the LSTM scripts:
```bash
python3 train_and_save.py
```

---

*Authored iteratively tracking sequential Unistroke recognition via PyGame and DL metrics.*
