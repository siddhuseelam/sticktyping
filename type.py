import pygame
import pygame.font
import time
import math
import numpy as np
from backend import UnistrokeEngine, ALPHA

# --- Configuration ---
WIDTH, HEIGHT = 1000, 600 # Wider to fit the debug menu
FPS = 60
DEADZONE = 0.2
SEGMENT_DELAY = 0.15

# Axis Definitions (Change if your controller maps differently)
LS_X, LS_Y = 0, 1 # Left Stick
RS_X, RS_Y = 3, 4 # Right Stick

# Colors
BG_COLOR = (30, 30, 34)
PANEL_COLOR = (45, 45, 50)
TEXT_COLOR = (220, 220, 220)
STROKE_COLOR = (100, 150, 255)
RADIAL_COLOR = (80, 220, 120)
DEBUG_TEXT_COLOR = (150, 150, 160)

def get_stick_direction(x, y, deadzone=DEADZONE):
    """Converts stick x,y input into a discrete 4-way direction."""
    if abs(x) < deadzone and abs(y) < deadzone:
        return "CENTER"
    
    angle = math.atan2(y, x) # Returns -pi to pi
    
    if -math.pi/4 <= angle <= math.pi/4:
        return "RIGHT"
    elif math.pi/4 < angle <= 3*math.pi/4:
        return "DOWN"
    elif angle > 3*math.pi/4 or angle <= -3*math.pi/4:
        return "LEFT"
    else:
        return "UP"

def main():
    pygame.init()
    pygame.joystick.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dual-Stick AI Keyboard")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 48, bold=True)
    font_med = pygame.font.SysFont("consolas", 28)
    font_small = pygame.font.SysFont("consolas", 18)

    # Initialize Backend
    print("Initializing Backend Engine...")
    engine = UnistrokeEngine()
    
    if pygame.joystick.get_count() == 0:
        print("No controller found. Plug in an Xbox controller and restart.")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # App State
    state = "DRAWING" # States: "DRAWING", "SELECTING"
    text_buffer = ""
    current_word = ""
    
    current_stroke = []
    center_return_time = None
    
    # Prediction Data
    top_4_chars = [] # Maps to [UP, RIGHT, DOWN, LEFT]
    debug_stroke = []
    debug_context = []

    running = True
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                # Handle physical buttons for typing flow
                if event.button == 4: # Left Bumper -> Backspace
                    text_buffer = text_buffer[:-1]
                    current_word = current_word[:-1]
                elif event.button == 5: # Right Bumper -> Space
                    text_buffer += " "
                    current_word = "" # Reset current word
                    state = "DRAWING"
                    current_stroke = []

        # Read Joysticks
        rs_x = joystick.get_axis(RS_X)
        rs_y = joystick.get_axis(RS_Y)
        ls_x = joystick.get_axis(LS_X)
        ls_y = joystick.get_axis(LS_Y)

        # ==========================================
        # STATE MACHINE LOGIC
        # ==========================================
        if state == "DRAWING":
            if abs(rs_x) > DEADZONE or abs(rs_y) > DEADZONE:
                # User is drawing
                current_stroke.append({"x": rs_x, "y": rs_y})
                center_return_time = None
            else:
                # User returned to center
                if len(current_stroke) > 0:
                    if center_return_time is None:
                        center_return_time = time.time()
                    elif time.time() - center_return_time >= SEGMENT_DELAY:
                        # Stroke Finished! Process predictions.
                        final_preds = engine.get_top_4_predictions(current_stroke, current_word)
                        
                        # Populate UI variables
                        # Map to [UP, RIGHT, DOWN, LEFT]
                        top_4_chars = [p[0] for p in final_preds]
                        
                        # -- Debug Extraction (Directly querying engine) --
                        if engine.model:
                            raw_coords = [[pt["x"], pt["y"]] for pt in current_stroke]
                            feats = engine._resample_and_extract_features(raw_coords)
                            feats_array = np.array([feats], dtype='float32')
                            s_preds = engine.model.predict(feats_array, verbose=0)[0]
                            top_s_idx = np.argsort(s_preds)[-5:][::-1]
                            debug_stroke = [(engine.int_to_char[i], s_preds[i]) for i in top_s_idx]
                        
                        c_preds = engine.context_engine.get_next_char_probabilities(current_word)
                        debug_context = sorted(c_preds.items(), key=lambda x: x[1], reverse=True)[:5]
                        # -------------------------------------------------
                        
                        state = "SELECTING"
                        center_return_time = None

        elif state == "SELECTING":
            ls_dir = get_stick_direction(ls_x, ls_y)
            
            if ls_dir != "CENTER":
                # User made a selection!
                selected_char = ""
                if ls_dir == "UP": selected_char = top_4_chars[0]
                elif ls_dir == "RIGHT": selected_char = top_4_chars[1]
                elif ls_dir == "DOWN": selected_char = top_4_chars[2]
                elif ls_dir == "LEFT": selected_char = top_4_chars[3]
                
                text_buffer += selected_char
                current_word += selected_char
                
                # Reset back to drawing state
                current_stroke = []
                state = "DRAWING"

        # ==========================================
        # RENDERING LOGIC
        # ==========================================
        
        # 1. Text Input Field (Top)
        pygame.draw.rect(screen, PANEL_COLOR, (20, 20, WIDTH - 40, 80), border_radius=10)
        display_text = text_buffer + ("_" if time.time() % 1 > 0.5 else " ") # Blinking cursor
        txt_surf = font_large.render(display_text, True, TEXT_COLOR)
        screen.blit(txt_surf, (40, 35))

        # 2. Main Interaction Area (Left Side)
        center_x, center_y = 350, 350
        scale = 150
        
        # Background Circle
        pygame.draw.circle(screen, PANEL_COLOR, (center_x, center_y), scale + 50, 2)

        if state == "DRAWING":
            # Render instructions
            inst_surf = font_med.render("Draw with Right Stick", True, DEBUG_TEXT_COLOR)
            screen.blit(inst_surf, (center_x - inst_surf.get_width()//2, center_y + 220))
            
            # Render the stroke being drawn
            if current_stroke:
                points = [(int(center_x + (pt["x"] * scale)), int(center_y + (pt["y"] * scale))) for pt in current_stroke]
                if len(points) > 1:
                    pygame.draw.lines(screen, STROKE_COLOR, False, points, 5)
                else:
                    pygame.draw.circle(screen, STROKE_COLOR, points[0], 5)

        elif state == "SELECTING":
            # Render the Radial Menu
            inst_surf = font_med.render("Select with Left Stick", True, RADIAL_COLOR)
            screen.blit(inst_surf, (center_x - inst_surf.get_width()//2, center_y + 220))
            
            if len(top_4_chars) == 4:
                positions = [
                    (center_x, center_y - 120), # UP
                    (center_x + 120, center_y), # RIGHT
                    (center_x, center_y + 120), # DOWN
                    (center_x - 120, center_y)  # LEFT
                ]
                for idx, pos in enumerate(positions):
                    char_surf = font_large.render(top_4_chars[idx].upper(), True, RADIAL_COLOR)
                    # Center the text on the position
                    rect = char_surf.get_rect(center=pos)
                    
                    # Highlight if Left Stick is pointing at it
                    ls_dir = get_stick_direction(ls_x, ls_y)
                    is_active = (ls_dir == ["UP", "RIGHT", "DOWN", "LEFT"][idx])
                    
                    if is_active:
                        pygame.draw.circle(screen, (100, 255, 150), pos, 40)
                        char_surf = font_large.render(top_4_chars[idx].upper(), True, BG_COLOR)
                    else:
                        pygame.draw.circle(screen, PANEL_COLOR, pos, 40)
                        
                    screen.blit(char_surf, rect)

        # 3. Debug Menu (Right Side)
        debug_x = 700
        pygame.draw.rect(screen, PANEL_COLOR, (debug_x, 120, 280, 450), border_radius=10)
        
        d_title = font_med.render("DEBUG MENU", True, TEXT_COLOR)
        screen.blit(d_title, (debug_x + 20, 130))
        
        alpha_lbl = font_small.render(f"Fusion Alpha: {ALPHA}", True, DEBUG_TEXT_COLOR)
        screen.blit(alpha_lbl, (debug_x + 20, 170))
        
        # Stroke Model Predictions
        s_title = font_small.render("1. Stroke Model (CNN):", True, STROKE_COLOR)
        screen.blit(s_title, (debug_x + 20, 210))
        for i, (char, prob) in enumerate(debug_stroke):
            lbl = font_small.render(f"   {char.upper()}: {prob*100:5.1f}%", True, TEXT_COLOR)
            screen.blit(lbl, (debug_x + 20, 235 + (i * 20)))

        # Context Engine Predictions
        c_title = font_small.render(f"2. Dictionary ('{current_word}'):", True, (200, 180, 100))
        screen.blit(c_title, (debug_x + 20, 360))
        for i, (char, prob) in enumerate(debug_context):
            lbl = font_small.render(f"   {char.upper()}: {prob*100:5.1f}%", True, TEXT_COLOR)
            screen.blit(lbl, (debug_x + 20, 385 + (i * 20)))

        # Controls hint
        hint_txt = font_small.render("LB: Backspace | RB: Space", True, DEBUG_TEXT_COLOR)
        screen.blit(hint_txt, (debug_x + 10, 530))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()