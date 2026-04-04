import pygame
import pygame.font
import time
import json
import os
import random
import string

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FPS = 60
DEADZONE = 0.15
SEGMENT_DELAY = 0.15
OUTPUT_FILE = "stroke_data.json"

BG_COLOR = (30, 30, 34)
SUCCESS_COLOR = (80, 220, 120)
STROKE_COLOR = (100, 150, 255)
DEADZONE_COLOR = (50, 50, 55)

def load_dataset():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_dataset(dataset):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=4)

def main():
    # Initialize everything at the global level
    pygame.init()
    pygame.font.init()
    pygame.joystick.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Unistroke Data Collector")
    clock = pygame.time.Clock()

    letter_font = pygame.font.SysFont("consolas", 120, bold=True)
    msg_font = pygame.font.SysFont("consolas", 24)

    if pygame.joystick.get_count() == 0:
        print("No Xbox controller detected!")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected to: {joystick.get_name()}")

    dataset = load_dataset()
    
    target_letter = random.choice(string.ascii_lowercase)
    current_stroke = []
    is_drawing = False
    center_return_time = None
    
    show_success = False
    success_timer = 0
    SUCCESS_DURATION = 0.6  

    running = True
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        x_axis = joystick.get_axis(0)
        y_axis = joystick.get_axis(1)

        if abs(x_axis) < DEADZONE: x_axis = 0.0
        if abs(y_axis) < DEADZONE: y_axis = 0.0
        is_in_center = (x_axis == 0.0 and y_axis == 0.0)

        if show_success:
            if time.time() - success_timer > SUCCESS_DURATION:
                show_success = False
                target_letter = random.choice(string.ascii_lowercase)
                current_stroke = []
            else:
                success_text = msg_font.render(f"Saved '{target_letter}'!", True, SUCCESS_COLOR)
                screen.blit(success_text, (WIDTH//2 - success_text.get_width()//2, HEIGHT//2 + 80))
        else:
            if not is_in_center:
                if not is_drawing:
                    is_drawing = True
                
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
                        dataset.append({
                            "label": target_letter,
                            "sequence": current_stroke
                        })
                        save_dataset(dataset) 
                        
                        is_drawing = False
                        center_return_time = None
                        show_success = True
                        success_timer = time.time()

        # Rendering
        color = SUCCESS_COLOR if show_success else (220, 220, 220)
        char_surface = letter_font.render(target_letter.upper(), True, color)
        screen.blit(char_surface, (WIDTH//2 - char_surface.get_width()//2, 80))

        center_x, center_y = WIDTH // 2, HEIGHT // 2 + 50
        scale = 150 
        pygame.draw.circle(screen, DEADZONE_COLOR, (center_x, center_y), int(DEADZONE * scale), 2)

        if current_stroke:
            points = [(int(center_x + (pt["x"] * scale)), int(center_y + (pt["y"] * scale))) for pt in current_stroke]
            if len(points) > 1:
                pygame.draw.lines(screen, STROKE_COLOR, False, points, 4)
            else:
                pygame.draw.circle(screen, STROKE_COLOR, points[0], 4)
        
        if not is_drawing and not show_success:
            inst_text = msg_font.render("Draw the letter to begin...", True, (150, 150, 150))
            screen.blit(inst_text, (WIDTH//2 - inst_text.get_width()//2, HEIGHT - 50))

        pygame.display.flip()
        clock.tick(FPS)

    save_dataset(dataset)
    print(f"Exited cleanly. Dataset has {len(dataset)} total strokes.")
    pygame.quit()

if __name__ == "__main__":
    main()