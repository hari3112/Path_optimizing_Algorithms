
import numpy as np
import random
import pygame
import time
import logging
import traceback
import csv
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------- 
# Environment Setup
# -------------------------

GRID_SIZE = 30
START = (1, 1)
GOAL = (29, 29)

# Static obstacles
obstacles = {(10, 10), (15, 20), (5, 25), (12, 18), (20, 5)}

# Actions: Up, Down, Left, Right, Diagonals
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Up-Left", "Up-Right", "Down-Left", "Down-Right"]

# Global variables for GUI and data collection
agent_trail = []
q_table = {}
total_reward = 0
animation_delay = 0.4
selected_mode = "Train"  # Train or Simulate
online_learning = True
show_q_values = False
use_maze = True
num_obstacles = 5
sensor_range = 3
path_length = 0
replan_count = 0
grid = None  # Initialize global grid
training_data = []  # Store episode, success_rate, avg_reward
simulation_data = []  # Store run, maze, obstacles, online_learning, path_length, avg_reward, replan_count
selected_state = START  # State for Q-table display

# Window dimensions
WIDTH = 680
UI_WIDTH = 200
HEIGHT = 680
WIN = pygame.display.set_mode((WIDTH + UI_WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Gridworld - Warehouse Robot")
pygame.font.init()
FONT = pygame.font.SysFont("arial", 16)  # Smaller font for Q-table

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
TURQUOISE = (64, 224, 208)
GREY = (128, 128, 128)
DARK_GREY = (100, 100, 100)
CYAN = (0, 255, 255)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.color = WHITE
        self.base_color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.visited = False
        self.q_values = [0.0] * len(ACTIONS)

    def get_pos(self): return (self.row, self.col)
    def is_open(self): return self.base_color == GREEN
    def is_barrier(self): return self.base_color == YELLOW
    def is_start(self): return self.base_color == ORANGE
    def is_end(self): return self.base_color == TURQUOISE
    def is_path(self): return self.base_color == PURPLE
    def reset(self):
        self.color = WHITE
        self.base_color = WHITE
        self.visited = False
    def make_start(self):
        self.color = ORANGE
        self.base_color = ORANGE
    def make_open(self):
        self.color = GREEN
        self.base_color = GREEN
    def make_barrier(self):
        self.color = YELLOW
        self.base_color = YELLOW
    def make_end(self):
        self.color = TURQUOISE
        self.base_color = TURQUOISE
    def make_path(self):
        self.color = PURPLE
        self.base_color = PURPLE
    def make_visited(self): 
        self.visited = True
        self.color = BLUE

    def draw(self, win, show_q=False):
        try:
            rect = pygame.Rect(self.x, self.y, self.width, self.width)
            pygame.draw.rect(win, self.color, rect)
            if self.visited:
                pygame.draw.circle(win, BLUE, (self.x + self.width // 2, self.y + self.width // 2), self.width // 3)
            if show_q and self.base_color == WHITE:
                max_q = max(self.q_values)
                if max_q > 0:
                    intensity = min(255, int(100 + (max_q * 155)))
                    overlay = (intensity, intensity, intensity)
                    pygame.draw.rect(win, overlay, rect, 1)
        except Exception as e:
            logging.error(f"Error drawing spot at ({self.row}, {self.col}): {str(e)}")

def is_valid(pos, obstacles_set):
    x, y = pos
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in obstacles_set:
        return True
    return False

def step(state, action, obstacles_set):
    new_state = (state[0] + action[0], state[1] + action[1])
    if not is_valid(new_state, obstacles_set):
        return state, -100, False
    if new_state == GOAL:
        return new_state, 100, True
    return new_state, -1, False

def get_q(state_pos):
    global q_table
    if state_pos not in q_table:
        q_table[state_pos] = [0.0] * len(ACTIONS)
    return q_table[state_pos]

def get_policy_path(grid, start, end):
    path = []
    current = start.get_pos()
    path.append(current)
    steps = 0
    while current != end.get_pos():
        q_vals = get_q(current)
        best_action = ACTIONS[np.argmax(q_vals)]
        next_pos = (current[0] + best_action[0], current[1] + best_action[1])
        if not is_valid(next_pos, obstacles):
            break
        path.append(next_pos)
        current = next_pos
        steps += 1
        if steps > 1000:
            break
    return path

def generate_maze(grid, rows):
    try:
        for row in grid:
            for spot in row:
                if spot.is_barrier():
                    spot.reset()
                    obstacles.discard((spot.row, spot.col))

        num_clusters = max(3, rows // 8)
        max_cluster_size = 5
        protected = {(1, 1), (rows-1, rows-1)}
        buffer = 3
        protected_zone = set()
        for r in range(rows):
            for c in range(rows):
                if (abs(r - 1) + abs(c - 1) <= buffer or abs(r - (rows-1)) + abs(c - (rows-1)) <= buffer):
                    protected_zone.add((r, c))
        
        def place_cluster(center_r, center_c, size):
            for dr in range(-size // 2, size // 2 + 1):
                for dc in range(-size // 2, size // 2 + 1):
                    r, c = center_r + dr, center_c + dc
                    if (0 <= r < rows and 0 <= c < rows and (r, c) not in protected_zone and random.random() < 0.7):
                        grid[r][c].make_barrier()
                        obstacles.add((r, c))

        available_centers = [(r, c) for r in range(3, rows-3) for c in range(3, rows-3) if (r, c) not in protected_zone]
        random.shuffle(available_centers)
        for i in range(min(num_clusters, len(available_centers))):
            r, c = available_centers[i]
            cluster_size = random.randint(2, max_cluster_size)
            place_cluster(r, c, cluster_size)

        logging.info(f"Maze generated with {num_clusters} clusters")
    except Exception as e:
        logging.error(f"Error generating maze: {str(e)}")

def initialize_obstacles(grid, num_obs, rows):
    global obstacles
    try:
        for row in grid:
            for spot in row:
                if spot.is_barrier() and (spot.row, spot.col) not in {START, GOAL} | set(agent_trail):
                    spot.reset()
                    obstacles.discard((spot.row, spot.col))

        used_positions = {START, GOAL} | set(agent_trail)
        available_positions = [(r, c) for r in range(rows) for c in range(rows)
                              if not grid[r][c].is_barrier() and (r, c) not in used_positions]
        
        random.shuffle(available_positions)
        num_obs = min(num_obs, len(available_positions))
        for i in range(num_obs):
            r, c = available_positions[i]
            grid[r][c].make_barrier()
            obstacles.add((r, c))
        
        logging.info(f"Initialized {num_obs} static obstacles")
    except Exception as e:
        logging.error(f"Error initializing obstacles: {str(e)}")

def make_grid(rows, width):
    try:
        grid = []
        gap = width // rows
        for i in range(rows):
            grid.append([])
            for j in range(rows):
                spot = Spot(i, j, gap, rows)
                pos = (i, j)
                if pos in q_table:
                    spot.q_values = q_table[pos]
                grid[i].append(spot)
        return grid
    except Exception as e:
        logging.error(f"Error creating grid: {str(e)}")
        return []

def draw_grid(win, rows, width):
    try:
        gap = width // rows
        for i in range(rows):
            pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
            for j in range(rows):
                pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))
    except Exception as e:
        logging.error(f"Error drawing grid: {str(e)}")

def draw_sensor_range(win, grid, agent_pos, rows, width):
    try:
        gap = width // rows
        r, c = agent_pos
        if 0 <= r < rows and 0 <= c < rows:
            spot = grid[r][c]
            x, y = spot.x, spot.y
            range_rect = pygame.Rect(
                x - sensor_range * gap, y - sensor_range * gap,
                (2 * sensor_range + 1) * gap, (2 * sensor_range + 1) * gap
            )
            pygame.draw.rect(win, CYAN, range_rect, 2)
    except Exception as e:
        logging.error(f"Error drawing sensor range: {str(e)}")

def draw_ui(win):
    try:
        pygame.draw.rect(win, DARK_GREY, (WIDTH, 0, UI_WIDTH, HEIGHT))
        buttons = [
            {"text": "Reset Grid", "rect": pygame.Rect(WIDTH + 20, 20, 160, 40), "action": "reset"},
            {"text": f"Mode: {selected_mode}", "rect": pygame.Rect(WIDTH + 20, 70, 160, 40), "action": "switch_mode"},
            {"text": "Train", "rect": pygame.Rect(WIDTH + 20, 120, 160, 40), "action": "train"},
            {"text": "Start/Stop Sim", "rect": pygame.Rect(WIDTH + 20, 170, 160, 40), "action": "toggle_sim"},
            {"text": f"Online Learn: {'On' if online_learning else 'Off'}", "rect": pygame.Rect(WIDTH + 20, 220, 160, 40), "action": "toggle_online"},
            {"text": f"{'Hide' if show_q_values else 'Show'} Q-Values", "rect": pygame.Rect(WIDTH + 20, 270, 160, 40), "action": "toggle_q"},
            {"text": f"{'Disable' if use_maze else 'Enable'} Maze", "rect": pygame.Rect(WIDTH + 20, 320, 160, 40), "action": "toggle_maze"},
            {"text": "Plot Results", "rect": pygame.Rect(WIDTH + 20, 360, 160, 40), "action": "plot_results"},
        ]
        for btn in buttons:
            pygame.draw.rect(win, GREY, btn["rect"])
            text = FONT.render(btn["text"], True, WHITE)
            win.blit(text, (btn["rect"].x + 10, btn["rect"].y + 10))

        sliders = [
            {"text": f"Sensor Range: {sensor_range}", "rect": pygame.Rect(WIDTH + 20, 410, 160, 20), "range": (1, 5), "value": sensor_range, "id": "sensor_range"},
            {"text": f"Obstacles: {num_obstacles}", "rect": pygame.Rect(WIDTH + 20, 450, 160, 20), "range": (0, 20), "value": num_obstacles, "id": "num_obstacles"},
        ]
        for slider in sliders:
            pygame.draw.rect(win, GREY, slider["rect"])
            min_val, max_val = slider["range"]
            val = slider["value"]
            handle_x = slider["rect"].x + ((val - min_val) / (max_val - min_val)) * slider["rect"].width
            pygame.draw.circle(win, WHITE, (int(handle_x), slider["rect"].y + 10), 8)
            text = FONT.render(slider["text"], True, WHITE)
            win.blit(text, (slider["rect"].x, slider["rect"].y - 20))

        # Metrics
        text = FONT.render(f"Replan Count: {replan_count}", True, WHITE)
        win.blit(text, (WIDTH + 20, 490))
        text = FONT.render(f"Path Length: {path_length}", True, WHITE)
        win.blit(text, (WIDTH + 20, 520))
        text = FONT.render(f"Avg Reward: {total_reward / max(1, path_length):.1f}", True, WHITE)
        win.blit(text, (WIDTH + 20, 550))

        # Q-table display for selected state
        text = FONT.render(f"Q-Values @ ({selected_state[0]}, {selected_state[1]}):", True, WHITE)
        win.blit(text, (WIDTH + 20, 580))
        q_vals = get_q(selected_state)
        for i, (action_name, q_val) in enumerate(zip(ACTION_NAMES, q_vals)):
            text = FONT.render(f"{action_name}: {q_val:.3f}", True, WHITE)
            win.blit(text, (WIDTH + 20, 600 + i * 20))

        return buttons, sliders
    except Exception as e:
        logging.error(f"Error drawing UI: {str(e)}")
        return [], []

def draw_grid_only(win, grid, width, rows, agent_pos=None):
    try:
        win.fill(WHITE)
        for row in grid:
            for spot in row:
                spot.draw(win, show_q_values)
        for (ox, oy) in obstacles:
            if 0 <= ox < rows and 0 <= oy < rows:
                pygame.draw.rect(win, BLACK, (oy * (width // rows), ox * (width // rows), width // rows, width // rows))
        for r, c in agent_trail:
            spot = grid[r][c]
            pygame.draw.circle(win, BLUE, (spot.x + spot.width // 2, spot.y + spot.width // 2), spot.width // 3)
        draw_grid(win, rows, width)
        if agent_pos:
            draw_sensor_range(win, grid, agent_pos, rows, width)
        draw_ui(win)
        pygame.display.update()
    except Exception as e:
        logging.error(f"Error drawing grid: {str(e)}")

def save_training_data():
    try:
        with open('training_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Success_Rate', 'Avg_Episode_Reward'])
            for episode, success_rate, avg_reward in training_data:
                writer.writerow([episode, success_rate, avg_reward])
        logging.info("Training data saved to training_data.csv")
    except Exception as e:
        logging.error(f"Error saving training data: {str(e)}")

def save_simulation_data():
    try:
        with open('simulation_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'Maze', 'Obstacles', 'Online_Learning', 'Path_Length', 'Avg_Reward', 'Replan_Count'])
            for run, maze, obstacles, online, path_len, avg_reward, replan in simulation_data:
                writer.writerow([run, maze, obstacles, online, path_len, avg_reward, replan])
        logging.info("Simulation data saved to simulation_data.csv")
    except Exception as e:
        logging.error(f"Error saving simulation data: {str(e)}")

def plot_results():
    try:
        if training_data:
            episodes, success_rates, avg_rewards = zip(*training_data)
            plt.figure(figsize=(10, 5))
            plt.plot(episodes, success_rates, marker='o', label='Success Rate', color='blue')
            plt.xlabel('Episode')
            plt.ylabel('Success Rate')
            plt.title('Convergence Rate: Success Rate vs. Episode')
            plt.grid(True)
            plt.legend()
            plt.savefig('success_rate.png')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(episodes, avg_rewards, marker='o', label='Average Episode Reward', color='green')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title('Convergence Rate: Average Reward vs. Episode')
            plt.grid(True)
            plt.legend()
            plt.savefig('avg_reward.png')
            plt.show()

        if simulation_data:
            runs = [data[0] for data in simulation_data]
            path_lengths = [data[4] for data in simulation_data]
            avg_rewards = [data[5] for data in simulation_data]
            replan_counts = [data[6] for data in simulation_data]

            plt.figure(figsize=(10, 5))
            plt.plot(runs, path_lengths, marker='o', label='Path Length', color='purple')
            plt.xlabel('Simulation Run')
            plt.ylabel('Path Length')
            plt.title('Path Length vs. Simulation Run')
            plt.grid(True)
            plt.legend()
            plt.savefig('path_length.png')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(runs, avg_rewards, marker='o', label='Average Reward', color='red')
            plt.xlabel('Simulation Run')
            plt.ylabel('Average Reward')
            plt.title('Average Reward vs. Simulation Run')
            plt.grid(True)
            plt.legend()
            plt.savefig('avg_reward_sim.png')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(runs, replan_counts, marker='o', label='Replan Count', color='orange')
            plt.xlabel('Simulation Run')
            plt.ylabel('Replan Count')
            plt.title('Replan Count vs. Simulation Run')
            plt.grid(True)
            plt.legend()
            plt.savefig('replan_count.png')
            plt.show()

        logging.info("Plots generated and saved")
    except Exception as e:
        logging.error(f"Error plotting results: {str(e)}")

def train_agent(grid, episodes_count=20000):
    global q_table, total_reward, training_data, selected_state
    try:
        success_count = 0
        total_reward = 0
        epsilon = 1.0
        alpha = 0.3
        gamma = 0.95
        decay = 0.999
        max_steps = GRID_SIZE * GRID_SIZE
        training_data = []

        for ep in range(episodes_count):
            state = START
            episode_reward = 0
            done = False

            for step_count in range(max_steps):
                if random.uniform(0, 1) < epsilon:
                    action_idx = random.randint(0, len(ACTIONS)-1)
                else:
                    q_vals = get_q(state)
                    action_idx = int(np.argmax(q_vals))

                action = ACTIONS[action_idx]
                next_state, reward, done = step(state, action, obstacles)

                if not done and next_state != state:
                    dist_old = abs(GOAL[0]-state[0]) + abs(GOAL[1]-state[1])
                    dist_new = abs(GOAL[0]-next_state[0]) + abs(GOAL[1]-next_state[1])
                    if dist_new < dist_old:
                        reward += 5
                    else:
                        reward -= 5

                old_q = get_q(state)[action_idx]
                next_max = max(get_q(next_state))
                new_q = old_q + alpha * (reward + gamma * next_max - old_q)
                get_q(state)[action_idx] = new_q

                state = next_state
                episode_reward += reward

                if done:
                    if next_state == GOAL:
                        success_count += 1
                    break

            total_reward += episode_reward
            epsilon = max(0.05, epsilon * decay)

            # Sync Q-table to grid every 100 episodes
            if (ep + 1) % 100 == 0:
                for row in grid:
                    for spot in row:
                        pos = spot.get_pos()
                        if pos in q_table:
                            spot.q_values = q_table[pos]
                # Update display for START state
                selected_state = START
                draw_grid_only(WIN, grid, WIDTH, GRID_SIZE, START)
                pygame.display.update()
                time.sleep(0.01)  # Brief pause to show update

            if (ep + 1) % 1000 == 0:
                success_rate = success_count / (ep + 1)
                training_data.append((ep + 1, success_rate, episode_reward))
                logging.info(f"Episode {ep+1}, epsilon={epsilon:.3f}, success={success_rate:.2f}, avg_reward={episode_reward:.1f}")

        # Final sync
        for row in grid:
            for spot in row:
                pos = spot.get_pos()
                if pos in q_table:
                    spot.q_values = q_table[pos]
        
        save_training_data()
        logging.info("Training finished")
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")

def simulate_episode(grid, online=True):
    global agent_trail, path_length, replan_count, total_reward, simulation_data, selected_state
    try:
        agent_trail = []
        state = START
        done = False
        episode_reward = 0
        path_length = 0
        replan_count = 0
        stuck_counter = 0
        epsilon_test = 0.1
        alpha = 0.3
        gamma = 0.9
        max_steps = 500
        
        while not done and path_length < max_steps:
            if random.uniform(0, 1) < epsilon_test:
                action_idx = random.randint(0, len(ACTIONS)-1)
            else:
                q_vals = get_q(state)
                action_idx = int(np.argmax(q_vals))
            
            action = ACTIONS[action_idx]
            next_state, reward, done = step(state, action, obstacles)
            
            if reward == -100:
                stuck_counter += 1
                if stuck_counter >= 10:
                    logging.info("Agent stuck, resetting to start")
                    state = START
                    stuck_counter = 0
                    agent_trail = [state]
                    path_length = 1
                    continue
            else:
                stuck_counter = 0
            
            if online:
                old_q = get_q(state)[action_idx]
                next_max = max(get_q(next_state))
                new_q = old_q + alpha * (reward + gamma * next_max - old_q)
                get_q(state)[action_idx] = new_q
                replan_count += 1
                grid[state[0]][state[1]].q_values = q_table[state]
            
            state = next_state
            episode_reward += reward
            agent_trail.append(state)
            path_length += 1
            
            # Update selected state for Q-table display
            selected_state = state
            draw_grid_only(WIN, grid, WIDTH, GRID_SIZE, state)
            time.sleep(animation_delay)
            yield
            
        total_reward += episode_reward
        run_number = len(simulation_data) + 1
        obstacles_count = num_obstacles if not use_maze else 'N/A'
        simulation_data.append((run_number, use_maze, obstacles_count, online, path_length, episode_reward / max(1, path_length), replan_count))
        save_simulation_data()
        
        if state == GOAL:
            logging.info("Reached goal")
        logging.info(f"Simulation episode complete. Reward: {episode_reward}, Reached goal: {done}")
        yield "DONE"
    except Exception as e:
        logging.error(f"Error in simulation: {str(e)}")
        yield "DONE"

def reset_grid(grid, rows):
    global q_table, agent_trail, total_reward, path_length, replan_count, selected_state
    try:
        q_table = {}
        agent_trail = []
        total_reward = 0
        path_length = 0
        replan_count = 0
        selected_state = START
        grid = make_grid(rows, WIDTH)
        grid[1][1].make_start()
        grid[rows-1][rows-1].make_end()
        if use_maze:
            generate_maze(grid, rows)
        else:
            initialize_obstacles(grid, num_obstacles, rows)
        return grid
    except Exception as e:
        logging.error(f"Error resetting grid: {str(e)}")
        return grid

def pygame_loop(win, width):
    global GRID_SIZE, num_obstacles, sensor_range, selected_mode, online_learning, show_q_values, use_maze, animation_delay, agent_trail, total_reward, path_length, replan_count, grid, selected_state
    try:
        dragging_slider = None
        grid = make_grid(GRID_SIZE, width)
        grid[1][1].make_start()
        grid[GRID_SIZE-1][GRID_SIZE-1].make_end()
        if use_maze:
            generate_maze(grid, GRID_SIZE)
        else:
            initialize_obstacles(grid, num_obstacles, GRID_SIZE)

        clock = pygame.time.Clock()
        running = True
        simulator = None
        setting_start = False
        setting_end = False
        training = False

        gap = width // GRID_SIZE
        while running:
            clock.tick(10)

            buttons, sliders = draw_ui(win)
            agent_pos = agent_trail[-1] if agent_trail else START
            draw_grid_only(win, grid, width, GRID_SIZE, agent_pos)

            if simulator:
                try:
                    result = next(simulator)
                    if result == "DONE":
                        logging.info("Animation finished")
                        simulator = None
                except StopIteration:
                    logging.info("Animation iterator exhausted")
                    simulator = None
                except Exception as e:
                    logging.error(f"Animation error: {str(e)}\n{traceback.format_exc()}")
                    simulator = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for slider in sliders:
                        if slider["rect"].collidepoint(pos):
                            dragging_slider = slider["id"]
                            logging.info(f"Started dragging slider: {dragging_slider}")
                            break
                    for btn in buttons:
                        if btn["rect"].collidepoint(pos):
                            if btn["action"] == "reset":
                                grid = reset_grid(grid, GRID_SIZE)
                                if simulator:
                                    simulator = None
                                logging.info("Grid reset")
                            elif btn["action"] == "switch_mode":
                                selected_mode = "Simulate" if selected_mode == "Train" else "Train"
                                logging.info(f"Mode switched to {selected_mode}")
                            elif btn["action"] == "train":
                                if not training:
                                    training = True
                                    train_agent(grid)
                                    training = False
                                    logging.info("Training initiated")
                            elif btn["action"] == "toggle_sim":
                                if simulator:
                                    simulator = None
                                    logging.info("Simulation stopped")
                                else:
                                    simulator = simulate_episode(grid, online_learning)
                                    logging.info("Simulation started")
                            elif btn["action"] == "toggle_online":
                                online_learning = not online_learning
                                logging.info(f"Online learning: {online_learning}")
                            elif btn["action"] == "toggle_q":
                                show_q_values = not show_q_values
                                logging.info(f"Q-values viz: {show_q_values}")
                            elif btn["action"] == "toggle_maze":
                                use_maze = not use_maze
                                grid = reset_grid(grid, GRID_SIZE)
                                if simulator:
                                    simulator = None
                                logging.info(f"Maze: {use_maze}")
                            elif btn["action"] == "plot_results":
                                plot_results()
                                logging.info("Plotting results")
                            break
                    else:
                        if pos[0] < WIDTH:
                            col = pos[0] // gap
                            row = pos[1] // gap
                            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                                spot = grid[row][col]
                                if event.button == 1:
                                    if not spot.is_start() and not spot.is_end() and (row, col) not in agent_trail:
                                        spot.make_barrier()
                                        obstacles.add((row, col))
                                        logging.info(f"Barrier added at ({row}, {col}) during {'simulation' if simulator else 'idle'}")
                                elif event.button == 3:
                                    spot.reset()
                                    obstacles.discard((row, col))
                                    logging.info(f"Barrier removed at ({row}, {col}) during {'simulation' if simulator else 'idle'}")
                                elif event.button == 2:  # Middle click to select state for Q-table
                                    selected_state = (row, col)
                                    logging.info(f"Selected state for Q-table: {selected_state}")

                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_slider = None
                    logging.info("Stopped dragging slider")

                if event.type == pygame.MOUSEMOTION and dragging_slider:
                    pos = pygame.mouse.get_pos()
                    for slider in sliders:
                        if slider["id"] == dragging_slider:
                            min_val, max_val = slider["range"]
                            x = max(slider["rect"].x, min(slider["rect"].x + slider["rect"].width, pos[0]))
                            value = min_val + (x - slider["rect"].x) / slider["rect"].width * (max_val - min_val)
                            value = int(round(value))
                            if slider["id"] == "sensor_range":
                                sensor_range = value
                                logging.info(f"Sensor range updated to {sensor_range}")
                            elif slider["id"] == "num_obstacles":
                                num_obstacles = value
                                if not use_maze:
                                    initialize_obstacles(grid, num_obstacles, GRID_SIZE)
                                    draw_grid_only(win, grid, width, GRID_SIZE, agent_pos)
                                logging.info(f"Obstacles updated to {num_obstacles}")
                            slider["value"] = value

                if event.type == pygame.MOUSEMOTION and not dragging_slider and not simulator:
                    pos = pygame.mouse.get_pos()
                    if pos[0] < WIDTH:
                        col = pos[0] // gap
                        row = pos[1] // gap
                        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                            selected_state = (row, col)
                            draw_grid_only(win, grid, width, GRID_SIZE, agent_pos)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        setting_start = True
                        logging.info("Set start")
                    elif event.key == pygame.K_e:
                        setting_end = True
                        logging.info("Set end")
                    elif event.key == pygame.K_SPACE:
                        start_pos = START
                        end_pos = GOAL
                        policy_path = get_policy_path(grid, grid[start_pos[0]][start_pos[1]], grid[end_pos[0]][end_pos[1]])
                        for pos in policy_path:
                            if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE:
                                grid[pos[0]][pos[1]].make_path()
                        path_length = len(policy_path)
                        logging.info(f"Policy path length: {path_length}")

        pygame.quit()
        logging.info("Program terminated")

    except Exception as e:
        logging.error(f"Main loop error: {str(e)}\n{traceback.format_exc()}")
    finally:
        pygame.quit()

if __name__ == '__main__':
    pygame_loop(WIN, WIDTH)
