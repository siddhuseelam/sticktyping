import pygame
import pygame.font
import time
import string
import numpy as np
import tensorflow as tf
from scipy import interpolate

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FPS = 60
DEADZONE = 0.15
SEGMENT_DELAY = 0.15
MODEL_PATH = 'unistroke_hybrid_model.keras' 

# AXIS CONFIGURATION
# Left Stick: 0, 1 | Right Stick: 3, 4 (Usually)
HORIZ_AXIS = 3 
VERT_AXIS = 4

BG_COLOR = (30, 30, 34)
TEXT_COLOR = (220, 220, 220)
STROKE_COLOR = (100, 150, 255)
DEADZONE_COLOR = (50, 50, 55)
PRED_COLOR = (80, 220, 120)

int_to_char = {i: c for i, c in enumerate(string.ascii_lowercase)}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def resample_and_extract_features(sequence, target_len=60):
    seq = np.array(sequence)
    if len(seq) < 2: return np.zeros((target_len, 4)) 
    
    distance = np.cumsum(np.sqrt(np.sum(np.diff(seq, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    if distance[-1] == 0: return np.zeros((target_len, 4))
    distance /= distance[-1]

    interpolator = interpolate.interp1d(distance, seq, axis=0)
    resampled = interpolator(np.linspace(0, 1, target_len))
    
    resampled -= np.mean(resampled, axis=0)
    max_val = np.max(np.abs(resampled))
    if max_val > 0: resampled /= max_val
        
    dx_dy = np.gradient(resampled, axis=0)
    return np.hstack((resampled, dx_dy))

def predict_stroke(stroke):
    if not model or len(stroke) < 2: return []
    raw_coords = [[pt["x"], pt["y"]] for pt in stroke]
    features = resample_and_extract_features(raw_coords, target_len=60)
    features_array = np.array([features], dtype='float32') 
    preds = model.predict(features_array, verbose=0)[0]
    top_5_indices = np.argsort(preds)[-5:][::-1]
    return [(int_to_char[idx], preds[idx]) for idx in top_5_indices]

def main():
    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Right Stick Predictor")
    clock = pygame.time.Clock()
    msg_font = pygame.font.SysFont("consolas", 24)

    if pygame.joystick.get_count() == 0:
        print("No controller found.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    current_stroke = []
    is_drawing = False
    center_return_time = None
    top_5_predictions = []

    running = True
    while running:
        screen.fill(BG_COLOR)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # --- RIGHT STICK INPUT ---
        try:
            x_axis = joystick.get_axis(HORIZ_AXIS)
            y_axis = joystick.get_axis(VERT_AXIS)
        except:
            # Fallback if axis index is out of range for your specific driver
            x_axis, y_axis = 0.0, 0.0

        if abs(x_axis) < DEADZONE: x_axis = 0.0
        if abs(y_axis) < DEADZONE: y_axis = 0.0
        is_in_center = (x_axis == 0.0 and y_axis == 0.0)

        if not is_in_center:
            if not is_drawing:
                is_drawing = True
                current_stroke = []
                top_5_predictions = []
            current_stroke.append({"x": x_axis, "y": y_axis})
            center_return_time = None
        else:
            if is_drawing:
                if center_return_time is None:
                    center_return_time = time.time()
                elif time.time() - center_return_time >= SEGMENT_DELAY:
                    top_5_predictions = predict_stroke(current_stroke)
                    is_drawing = False
                    center_return_time = None

        # Rendering Logic
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        scale = 150 
        pygame.draw.circle(screen, DEADZONE_COLOR, (center_x, center_y), int(DEADZONE * scale), 2)

        if current_stroke:
            points = [(int(center_x + (pt["x"] * scale)), int(center_y + (pt["y"] * scale))) for pt in current_stroke]
            if len(points) > 1:
                pygame.draw.lines(screen, STROKE_COLOR, False, points, 4)

        if top_5_predictions and not is_drawing:
            for i, (char, conf) in enumerate(top_5_predictions):
                txt = msg_font.render(f"{i+1}. {char.upper()} ({conf*100:.1f}%)", True, PRED_COLOR)
                screen.blit(txt, (50, 150 + (i*30)))

        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()