import pygame
import numpy as np
import time

# Initialize Pygame
pygame.init()

# Window Settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Mountain Field Test: Signal Flow")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
MOUNTAIN_COLOR = (100, 150, 100)  # Dark green for mountains
SKY_COLOR = (135, 206, 235)  # Light blue for sky

# Fonts
font = pygame.font.SysFont("Arial", 30)
small_font = pygame.font.SysFont("Arial", 20)

# Button Settings
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
button_rect = pygame.Rect(WINDOW_WIDTH // 2 - BUTTON_WIDTH // 2, WINDOW_HEIGHT - 100, BUTTON_WIDTH, BUTTON_HEIGHT)

# Simulation Parameters (Matching Your Script)
initial_snr = 6.8
distance_factor = 0.5
t = np.linspace(0, 1, 1000)
interference = 0.1 * np.sin(100 * t)
gain = 1.0
max_snr_display = 25  # For scaling the bar and graph

# Simulation Stages
stages = [
    "Starting Signal",
    "Terrain Effects",
    "Ad-Hoc Boost",
    "Optimizing Signal",
    "Final Result"
]
current_stage = 0
snr_values = [initial_snr]
current_snr = initial_snr
stage_duration = 2  # Seconds per stage
last_stage_time = 0
simulation_running = False
modem_gain = 0

# Signal Optimization Logic
def simulate_terrain_effects(snr):
    snr_over_time = snr * np.exp(-distance_factor * t) + interference
    return snr_over_time[-1]

def apply_adhoc_boost(snr):
    return snr * 1.2

def optimize_signal(final_snr):
    def objective(gain):
        return -final_snr * gain[0]
    
    gain = np.array([final_snr * 0.01])
    learning_rate = 0.01
    max_iter = 100
    for _ in range(max_iter):
        grad = (objective(gain + 0.01) - objective(gain - 0.01)) / 0.02
        gain -= learning_rate * grad
        if np.abs(grad) < 1e-4:
            break
    optimized_snr = final_snr * gain[0]
    modem_gain = gain[0] * final_snr * 0.01 * 4
    return optimized_snr, modem_gain

# Draw Mountain Background
def draw_background():
    # Draw sky
    pygame.draw.rect(window, SKY_COLOR, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT // 2))
    # Draw ground and mountains
    pygame.draw.rect(window, MOUNTAIN_COLOR, (0, WINDOW_HEIGHT // 2, WINDOW_WIDTH, WINDOW_HEIGHT // 2))
    # Draw two mountain peaks
    pygame.draw.polygon(window, MOUNTAIN_COLOR, [(0, WINDOW_HEIGHT), (200, 300), (400, WINDOW_HEIGHT)])
    pygame.draw.polygon(window, MOUNTAIN_COLOR, [(400, WINDOW_HEIGHT), (600, 300), (800, WINDOW_HEIGHT)])

# Main Loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos) and not simulation_running:
                simulation_running = True
                current_stage = 0
                snr_values = [initial_snr]
                current_snr = initial_snr
                modem_gain = 0
                last_stage_time = time.time()

    # Update Simulation
    if simulation_running:
        current_time = time.time()
        if current_time - last_stage_time >= stage_duration:
            if current_stage < len(stages) - 1:
                current_stage += 1
                last_stage_time = current_time

                # Process Each Stage
                if current_stage == 1:  # Terrain Effects
                    current_snr = simulate_terrain_effects(initial_snr)
                elif current_stage == 2:  # Ad-Hoc Boost
                    current_snr = apply_adhoc_boost(current_snr)
                elif current_stage == 3:  # Optimizing Signal
                    current_snr, modem_gain = optimize_signal(current_snr)
                elif current_stage == 4:  # Final Result
                    simulation_running = False

                snr_values.append(current_snr)

    # Draw Background
    draw_background()

    # Draw Title
    title_text = font.render("Mountain Field Test: Signal Flow", True, BLACK)
    window.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))

    # Draw Stage Text
    stage_text = font.render(f"Stage: {stages[current_stage]}", True, BLACK)
    window.blit(stage_text, (WINDOW_WIDTH // 2 - stage_text.get_width() // 2, 80))

    # Draw SNR Bar
    bar_max_height = 300
    bar_width = 50
    bar_x = WINDOW_WIDTH // 2 - bar_width // 2
    bar_y = 150
    bar_height = (current_snr / max_snr_display) * bar_max_height
    pygame.draw.rect(window, GRAY, (bar_x, bar_y, bar_width, bar_max_height), 2)
    pygame.draw.rect(window, GREEN, (bar_x, bar_y + bar_max_height - bar_height, bar_width, bar_height))

    # Draw SNR Value
    snr_text = font.render(f"SNR: {current_snr:.2f} dB", True, BLACK)
    window.blit(snr_text, (WINDOW_WIDTH // 2 - snr_text.get_width() // 2, bar_y + bar_max_height + 20))

    # Draw Modem Gain (if applicable)
    if current_stage == 4 and modem_gain > 0:
        gain_text = small_font.render(f"Modem Gain: {modem_gain:.4f}", True, BLUE)
        window.blit(gain_text, (WINDOW_WIDTH // 2 - gain_text.get_width() // 2, bar_y + bar_max_height + 60))

    # Draw SNR Line Graph
    if len(snr_values) > 1:
        graph_y_base = 500  # Base Y position for the graph
        graph_height = 200  # Height of the graph area
        points = [(i * (WINDOW_WIDTH - 100) / (len(snr_values) - 1) + 50,
                   graph_y_base - (snr / max_snr_display) * graph_height) for i, snr in enumerate(snr_values)]
        pygame.draw.lines(window, RED, False, points, 2)

    # Draw Simulate Button
    pygame.draw.rect(window, BLUE if not simulation_running else GRAY, button_rect)
    button_text = small_font.render("Simulate", True, WHITE)
    window.blit(button_text, (button_rect.x + button_rect.width // 2 - button_text.get_width() // 2,
                              button_rect.y + button_rect.height // 2 - button_text.get_height() // 2))

    # Update Display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
