import pygame
import pygame.font
import time
import string
import numpy as np
import tensorflow as tf

# Load the trained model
try:
    model = tf.keras.models.load_model('stroke_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model. Be sure 'stroke_model.h5' is created.", e)
    model = None

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FPS = 60
DEADZONE = 0.15
SEGMENT_DELAY = 0.15

BG_COLOR = (30, 30, 34)
TEXT_COLOR = (220, 220, 220)
STROKE_COLOR = (100, 150, 255)
DEADZONE_COLOR = (50, 50, 55)
PRED_COLOR = (80, 220, 120)

int_to_char = {i: c for i, c in enumerate(string.ascii_lowercase)}

def predict_stroke(stroke):
    if not model or len(stroke) == 0:
        return []
    
    # Format for model: sequence shape should be (1, N, 2)
    features = [[pt["x"], pt["y"]] for pt in stroke]
    features_array = np.array([features], dtype='float32') # Expand dims to batch size 1
    
    # Predict
    preds = model.predict(features_array, verbose=0)[0]
    
    # Get top 5 indices
    top_5_indices = np.argsort(preds)[-5:][::-1]
    
    results = []
    for idx in top_5_indices:
        results.append((int_to_char[idx], preds[idx]))
    return results

def main():
    pygame.init()
    pygame.font.init()
    pygame.joystick.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Live Prediction App")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("consolas", 36, bold=True)
    msg_font = pygame.font.SysFont("consolas", 24)

    if pygame.joystick.get_count() == 0:
        print("No Xbox controller detected!")
        screen.fill(BG_COLOR)
        error_text = msg_font.render("No Xbox controller detected! Please plug one in.", True, (255, 100, 100))
        screen.blit(error_text, (WIDTH//2 - error_text.get_width()//2, HEIGHT//2))
        pygame.display.flip()
        time.sleep(3)
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected to: {joystick.get_name()}")

    current_stroke = []
    is_drawing = False
    center_return_time = None
    
    top_5_predictions = []

    running = True
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        x_axis = joystick.get_axis(3)
        y_axis = joystick.get_axis(4)

        if abs(x_axis) < DEADZONE: x_axis = 0.0
        if abs(y_axis) < DEADZONE: y_axis = 0.0
        is_in_center = (x_axis == 0.0 and y_axis == 0.0)

        if not is_in_center:
            if not is_drawing:
                is_drawing = True
                current_stroke = [] # Start a new stroke
                top_5_predictions = [] # Clear previous predictions when drawing
                
            current_stroke.append({
                "x": round(x_axis, 4),
                "y": round(y_axis, 4),
                "t": time.time()
            })
            center_return_time = None
        else:
            if is_drawing:
                if center_return_time is None:
                    center_return_time = time.time()
                elif time.time() - center_return_time >= SEGMENT_DELAY:
                    # Stroke ended
                    top_5_predictions = predict_stroke(current_stroke)
                    is_drawing = False
                    center_return_time = None

        # Rendering
        title_surf = title_font.render("Live Stroke Predictor", True, TEXT_COLOR)
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 30))

        center_x, center_y = WIDTH // 2, HEIGHT // 2
        scale = 150 
        pygame.draw.circle(screen, DEADZONE_COLOR, (center_x, center_y), int(DEADZONE * scale), 2)

        # Draw stroke
        if current_stroke:
            points = [(int(center_x + (pt["x"] * scale)), int(center_y + (pt["y"] * scale))) for pt in current_stroke]
            if len(points) > 1:
                pygame.draw.lines(screen, STROKE_COLOR, False, points, 4)
            else:
                pygame.draw.circle(screen, STROKE_COLOR, points[0], 4)
        
        # Display instructions or predictions
        if not is_drawing and not current_stroke:
            inst_text = msg_font.render("Draw a letter to begin predicting...", True, (150, 150, 150))
            screen.blit(inst_text, (WIDTH//2 - inst_text.get_width()//2, HEIGHT - 50))

        # Show Predictions
        if top_5_predictions and not is_drawing:
            pred_y = HEIGHT // 2 - 100
            pred_x = 50
            header_text = msg_font.render("Top 5 Predictions:", True, TEXT_COLOR)
            screen.blit(header_text, (pred_x, pred_y))
            
            for i, (char, conf) in enumerate(top_5_predictions):
                text_str = f"{i+1}. {char.upper()} - {conf*100:.1f}%"
                text_surf = msg_font.render(text_str, True, PRED_COLOR)
                screen.blit(text_surf, (pred_x, pred_y + 30 + (i*30)))
                
        elif is_drawing:
            inst_text = msg_font.render("Drawing...", True, STROKE_COLOR)
            screen.blit(inst_text, (WIDTH//2 - inst_text.get_width()//2, HEIGHT - 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
