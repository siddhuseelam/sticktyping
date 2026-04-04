import pygame
import pygame.font
import time
import math
import numpy as np
from backend import UnistrokeEngine, ALPHA

# --- Configuration ---
WIDTH, HEIGHT = 1000, 600 
FPS = 60
DEADZONE = 0.2
SEGMENT_DELAY = 0.15

# Axis Definitions 
LS_X, LS_Y = 0, 1 
RS_X, RS_Y = 3, 4 

# Standard Xbox Button Mapping
BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5

# Colors
BG_COLOR = (30, 30, 34)
PANEL_COLOR = (45, 45, 50)
TEXT_COLOR = (220, 220, 220)
STROKE_COLOR = (100, 150, 255)
RADIAL_COLOR = (80, 220, 120)
DEBUG_TEXT_COLOR = (150, 150, 160)

# Xbox Face Button Colors
COLOR_Y = (240, 200, 50)  # Yellow
COLOR_X = (50, 150, 255)  # Blue
COLOR_A = (80, 220, 100)  # Green
COLOR_B = (240, 80, 80)   # Red

def get_stick_direction(x, y, deadzone=DEADZONE):
    if abs(x) < deadzone and abs(y) < deadzone:
        return "CENTER"
    angle = math.atan2(y, x) 
    if -math.pi/4 <= angle <= math.pi/4: return "RIGHT"
    elif math.pi/4 < angle <= 3*math.pi/4: return "DOWN"
    elif angle > 3*math.pi/4 or angle <= -3*math.pi/4: return "LEFT"
    else: return "UP"

def main():
    pygame.init()
    pygame.joystick.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dual-Stick AI Keyboard")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 48, bold=True)
    font_med = pygame.font.SysFont("consolas", 28)
    font_small = pygame.font.SysFont("consolas", 18)

    print("Initializing Backend Engine...")
    engine = UnistrokeEngine()
    
    if pygame.joystick.get_count() == 0:
        print("No controller found. Plug in an Xbox controller and restart.")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    state = "DRAWING" 
    text_buffer = ""
    current_word = ""
    current_stroke = []
    center_return_time = None
    
    # Predictions
    top_4_chars = [] 
    top_4_words = []
    debug_stroke = []
    debug_context = []

    # Non-blocking Rumble Variables
    pending_second_rumble = False
    second_rumble_time = 0

    running = True
    while running:
        screen.fill(BG_COLOR)
        
        # --- Handle Haptic Feedback Timer ---
        if pending_second_rumble and time.time() >= second_rumble_time:
            joystick.rumble(0.8, 0.8, 150) # Second heavy vibrate
            pending_second_rumble = False
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                
                # Handling Backspace and Space
                if event.button == BTN_LB:
                    text_buffer = text_buffer[:-1]
                    current_word = current_word[:-1]
                    top_4_words = engine.get_word_suggestions(current_word)
                elif event.button == BTN_RB:
                    text_buffer += " "
                    current_word = "" 
                    state = "DRAWING"
                    current_stroke = []
                    top_4_words = []

                # Handling Word Selection (Y, X, B, A)
                elif event.button in [BTN_Y, BTN_X, BTN_B, BTN_A] and top_4_words:
                    selected_word = ""
                    if event.button == BTN_Y and len(top_4_words) > 0: selected_word = top_4_words[0]
                    elif event.button == BTN_X and len(top_4_words) > 1: selected_word = top_4_words[1]
                    elif event.button == BTN_B and len(top_4_words) > 2: selected_word = top_4_words[2]
                    elif event.button == BTN_A and len(top_4_words) > 3: selected_word = top_4_words[3]
                    
                    if selected_word:
                        # Remove the partial word and inject the full word
                        if len(current_word) > 0:
                            text_buffer = text_buffer[:-len(current_word)]
                        
                        text_buffer += selected_word + " "
                        current_word = ""
                        top_4_words = []
                        state = "DRAWING"
                        
                        # Double Vibration for Word Success
                        joystick.rumble(0.8, 0.8, 150) # First heavy vibrate
                        pending_second_rumble = True
                        second_rumble_time = time.time() + 0.25 # Trigger second vibrate in 250ms

        # Read Joysticks
        rs_x = joystick.get_axis(RS_X)
        rs_y = joystick.get_axis(RS_Y)
        ls_x = joystick.get_axis(LS_X)
        ls_y = joystick.get_axis(LS_Y)

        if state == "DRAWING":
            if abs(rs_x) > DEADZONE or abs(rs_y) > DEADZONE:
                current_stroke.append({"x": rs_x, "y": rs_y})
                center_return_time = None
            else:
                if len(current_stroke) > 0:
                    if center_return_time is None:
                        center_return_time = time.time()
                    elif time.time() - center_return_time >= SEGMENT_DELAY:
                        final_preds = engine.get_top_4_predictions(current_stroke, current_word)
                        top_4_chars = [p[0] for p in final_preds]
                        
                        # Debug info
                        if engine.model:
                            raw_coords = [[pt["x"], pt["y"]] for pt in current_stroke]
                            feats = engine._resample_and_extract_features(raw_coords)
                            feats_array = np.array([feats], dtype='float32')
                            s_preds = engine.model.predict(feats_array, verbose=0)[0]
                            top_s_idx = np.argsort(s_preds)[-5:][::-1]
                            debug_stroke = [(engine.int_to_char[i], s_preds[i]) for i in top_s_idx]
                        
                        c_preds = engine.context_engine.get_next_char_probabilities(current_word)
                        debug_context = sorted(c_preds.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        state = "SELECTING"
                        center_return_time = None

        elif state == "SELECTING":
            ls_dir = get_stick_direction(ls_x, ls_y)
            if ls_dir != "CENTER":
                selected_char = ""
                if ls_dir == "UP": selected_char = top_4_chars[0]
                elif ls_dir == "RIGHT": selected_char = top_4_chars[1]
                elif ls_dir == "DOWN": selected_char = top_4_chars[2]
                elif ls_dir == "LEFT": selected_char = top_4_chars[3]
                
                text_buffer += selected_char
                current_word += selected_char
                
                # Fetch new word predictions based on the new character
                top_4_words = engine.get_word_suggestions(current_word)
                
                # Single Vibration for Character Success
                joystick.rumble(0.5, 0.5, 100)
                
                current_stroke = []
                state = "DRAWING"

        # ==========================================
        # RENDERING LOGIC
        # ==========================================
        
        # 1. Text Input Field
        pygame.draw.rect(screen, PANEL_COLOR, (20, 20, WIDTH - 40, 80), border_radius=10)
        display_text = text_buffer + ("_" if time.time() % 1 > 0.5 else " ")
        txt_surf = font_large.render(display_text, True, TEXT_COLOR)
        screen.blit(txt_surf, (40, 35))

        # 2. Main Interaction Area (Left Side)
        center_x, center_y = 250, 350
        scale = 150
        
        pygame.draw.circle(screen, PANEL_COLOR, (center_x, center_y), scale + 50, 2)

        if state == "DRAWING":
            inst_surf = font_med.render("Draw with Right Stick", True, DEBUG_TEXT_COLOR)
            screen.blit(inst_surf, (center_x - inst_surf.get_width()//2, center_y + 220))
            if current_stroke:
                points = [(int(center_x + (pt["x"] * scale)), int(center_y + (pt["y"] * scale))) for pt in current_stroke]
                if len(points) > 1: pygame.draw.lines(screen, STROKE_COLOR, False, points, 5)

        elif state == "SELECTING":
            inst_surf = font_med.render("Select with Left Stick", True, RADIAL_COLOR)
            screen.blit(inst_surf, (center_x - inst_surf.get_width()//2, center_y + 220))
            if len(top_4_chars) == 4:
                positions = [(center_x, center_y - 120), (center_x + 120, center_y), (center_x, center_y + 120), (center_x - 120, center_y)]
                for idx, pos in enumerate(positions):
                    is_active = (get_stick_direction(ls_x, ls_y) == ["UP", "RIGHT", "DOWN", "LEFT"][idx])
                    if is_active:
                        pygame.draw.circle(screen, (100, 255, 150), pos, 40)
                        char_surf = font_large.render(top_4_chars[idx].upper(), True, BG_COLOR)
                    else:
                        pygame.draw.circle(screen, PANEL_COLOR, pos, 40)
                        char_surf = font_large.render(top_4_chars[idx].upper(), True, RADIAL_COLOR)
                    screen.blit(char_surf, char_surf.get_rect(center=pos))

        # 3. Word Suggestions (Middle-Right Diamond Layout)
        word_center_x, word_center_y = 600, 350
        w_title = font_med.render("Word Suggestions", True, TEXT_COLOR)
        screen.blit(w_title, (word_center_x - w_title.get_width()//2, 120))
        
        if top_4_words:
            w_positions = [
                (word_center_x, word_center_y - 80, "(Y)", COLOR_Y),  # Top
                (word_center_x - 120, word_center_y, "(X)", COLOR_X), # Left
                (word_center_x + 120, word_center_y, "(B)", COLOR_B), # Right
                (word_center_x, word_center_y + 80, "(A)", COLOR_A)   # Bottom
            ]
            for idx, (wx, wy, btn_lbl, color) in enumerate(w_positions):
                if idx < len(top_4_words):
                    # Draw button prompt
                    btn_surf = font_med.render(btn_lbl, True, color)
                    screen.blit(btn_surf, btn_surf.get_rect(center=(wx, wy - 20)))
                    # Draw word
                    word_surf = font_med.render(top_4_words[idx], True, TEXT_COLOR)
                    screen.blit(word_surf, word_surf.get_rect(center=(wx, wy + 15)))
        else:
            none_surf = font_small.render("Keep typing...", True, DEBUG_TEXT_COLOR)
            screen.blit(none_surf, none_surf.get_rect(center=(word_center_x, word_center_y)))

        # 4. Debug Menu (Far Right Side)
        debug_x = 800
        pygame.draw.rect(screen, PANEL_COLOR, (debug_x, 120, 180, 450), border_radius=10)
        
        d_title = font_small.render("DEBUG MENU", True, TEXT_COLOR)
        screen.blit(d_title, (debug_x + 10, 130))
        
        s_title = font_small.render("1. CNN Model:", True, STROKE_COLOR)
        screen.blit(s_title, (debug_x + 10, 170))
        for i, (char, prob) in enumerate(debug_stroke):
            lbl = font_small.render(f" {char.upper()}: {prob*100:4.0f}%", True, TEXT_COLOR)
            screen.blit(lbl, (debug_x + 10, 195 + (i * 20)))

        c_title = font_small.render(f"2. Context:", True, (200, 180, 100))
        screen.blit(c_title, (debug_x + 10, 320))
        for i, (char, prob) in enumerate(debug_context):
            lbl = font_small.render(f" {char.upper()}: {prob*100:4.0f}%", True, TEXT_COLOR)
            screen.blit(lbl, (debug_x + 10, 345 + (i * 20)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()