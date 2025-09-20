import pygame
import math
from queue import PriorityQueue
import time
import logging
import traceback
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Window dimensions
WIDTH = 680
UI_WIDTH = 200
HEIGHT = 680
WIN = pygame.display.set_mode((WIDTH + UI_WIDTH, HEIGHT))
pygame.display.set_caption("A* Path Finding Algorithm - Warehouse Robot")
pygame.font.init()
FONT = pygame.font.SysFont("arial", 20)

# Colors
RED = (255, 0, 0)  # Unused
GREEN = (0, 255, 0)  # Unused for open nodes
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
TURQUOISE = (64, 224, 208)
GREY = (128, 128, 128)
DARK_GREY = (100, 100, 100)
CYAN = (0, 255, 255)  # For sensor range visualization

# Global variables
agent_trail = []
num_obstacles = 10
moving_obstacles = []
dynamic_positions = set()
rows = 30
animation_delay = 0.3
speed_scale = 1.0
selected_algorithm = "A*"
dragging_slider = None
replan_count = 0
path_positions = []
SENSOR_RANGE = 3  # Manhattan distance for sensor range
show_sensor_range = True  # Toggle for sensor range visualization
use_maze = True  # Toggle for maze generation
path_length = 0  # Track path length

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

    def get_pos(self): return self.row, self.col
    def is_open(self): return self.base_color == GREEN
    def is_barrier(self): return self.base_color == YELLOW
    def is_start(self): return self.base_color == ORANGE
    def is_end(self): return self.base_color == TURQUOISE
    def is_path(self): return self.base_color == PURPLE
    def reset(self):
        self.color = WHITE
        self.base_color = WHITE
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
    def make_visited(self): self.visited = True
    def set_dynamic_obstacle(self):
        self.color = YELLOW  # Dynamic obstacles drawn in draw_grid_only
    def clear_dynamic_obstacle(self):
        self.color = self.base_color

    def draw(self, win):
        try:
            rect = pygame.Rect(self.x, self.y, self.width, self.width)
            pygame.draw.rect(win, self.color, rect)
            if self.visited:
                pygame.draw.circle(win, BLUE, (self.x + self.width // 2, self.y + self.width // 2), self.width // 3)
        except Exception as e:
            logging.error(f"Error drawing spot at ({self.row}, {self.col}): {str(e)}")

    def update_neighbors(self, grid, detected_obstacles):
        try:
            self.neighbors = []
            directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
            for dr, dc in directions:
                r, c = self.row + dr, self.col + dc
                if 0 <= r < self.total_rows and 0 <= c < self.total_rows and not grid[r][c].is_barrier():
                    if detected_obstacles and (r, c) in detected_obstacles:
                        continue
                    if abs(dr) == 1 and abs(dc) == 1:
                        if grid[self.row + dr][self.col].is_barrier() or grid[self.row][self.col + dc].is_barrier():
                            continue
                    self.neighbors.append(grid[r][c])
            logging.debug(f"Neighbors updated for spot at ({self.row}, {self.col}): {[(n.row, n.col) for n in self.neighbors]}")
        except Exception as e:
            logging.error(f"Error updating neighbors for spot at ({self.row}, {self.col}): {str(e)}")

    def __lt__(self, other): return False

def h(p1, p2, algorithm):
    try:
        if algorithm == "Dijkstra": return 0
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    except Exception as e:
        logging.error(f"Error calculating heuristic: {str(e)}")
        return 0

def get_obstacles_in_range(agent_pos, obstacles):
    try:
        detected = set()
        for obs in obstacles:
            r, c = obs["pos"]
            if h(agent_pos, (r, c), "A*") <= SENSOR_RANGE:
                detected.add((r, c))
        logging.debug(f"Detected obstacles in range {SENSOR_RANGE} of {agent_pos}: {detected}")
        return detected
    except Exception as e:
        logging.error(f"Error detecting obstacles: {str(e)}")
        return set()

def reconstruct_path(came_from, current, grid):
    global path_positions, path_length
    try:
        path = []
        while current in came_from:
            current = came_from[current]
            pos = current.get_pos()
            path.append(pos)
            if not current.is_start() and not current.is_end() and pos not in dynamic_positions:
                current.make_path()
        path.reverse()
        path_length = len(path)
        logging.debug(f"Path reconstructed: {path}, Length: {path_length}")
        path_positions = path
        for r, c in dynamic_positions:
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                grid[r][c].set_dynamic_obstacle()
            else:
                logging.warning(f"Invalid dynamic obstacle position: ({r}, {c})")
        return path
    except Exception as e:
        logging.error(f"Error reconstructing path: {str(e)}")
        return []

def predict_obstacle_positions(grid, obstacles, current_time, steps, grid_size, agent_pos):
    try:
        predicted = []
        detected_obstacles = get_obstacles_in_range(agent_pos, obstacles)
        used_positions = set()
        for obs in obstacles:
            r, c = obs["pos"]
            if (r, c) not in detected_obstacles:
                predicted.append((r, c))
                used_positions.add((r, c))
                continue
            speed = obs["speed"] * speed_scale
            move_interval = 1.0 / speed if speed > 0 else 1.0
            if current_time + animation_delay >= obs["last_moved"] + move_interval:
                directions = [
                    (-3,0), (3,0), (0,-3), (0,3),
                    (-2,0), (2,0), (0,-2), (0,2),
                    (-1,0), (1,0), (0,-1), (0,1),
                    (-3,-3), (-3,3), (3,-3), (3,3),
                    (-2,-2), (-2,2), (2,-2), (2,2),
                    (-1,-1), (-1,1), (1,-1), (1,1),
                    (-3,-2), (-3,2), (3,-2), (3,2),
                    (-2,-3), (-2,3), (2,-3), (2,3),
                    (-3,-1), (-3,1), (3,-1), (3,1),
                    (-1,-3), (-1,3), (1,-3), (1,3)
                ]
                random.shuffle(directions)
                for dr, dc in directions[:5]:
                    new_r, new_c = r + dr, c + dc
                    if (0 <= new_r < grid_size and 0 <= new_c < grid_size and
                        (new_r, new_c) not in used_positions and
                        not grid[new_r][new_c].is_barrier() and
                        not grid[new_r][new_c].is_start() and
                        not grid[new_r][new_c].is_end() and
                        is_clear_path(grid, r, c, new_r, new_c)):
                        predicted.append((new_r, new_c))
                        used_positions.add((new_r, new_c))
                        break
                else:
                    predicted.append((r, c))
                    used_positions.add((r, c))
            else:
                predicted.append((r, c))
                used_positions.add((r, c))
        logging.debug(f"Predicted obstacle positions: {predicted}")
        return predicted
    except Exception as e:
        logging.error(f"Error predicting obstacle positions: {str(e)}")
        return []

def calculate_penalty(current, neighbor, obstacles, detected_obstacles):
    try:
        if not detected_obstacles:
            return 1
        penalty = 1
        for obs in obstacles:
            obs_pos = tuple(obs["pos"])
            if obs_pos not in detected_obstacles:
                continue
            distance = h(current.get_pos(), obs_pos, selected_algorithm)
            if distance <= 2:
                penalty += 5 * (1 / max(0.1, distance)) * obs["speed"] * speed_scale
        return penalty
    except Exception as e:
        logging.error(f"Error calculating penalty: {str(e)}")
        return 1

def is_clear_path(grid, x1, y1, x2, y2):
    try:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
            if not (0 <= x1 < len(grid) and 0 <= y1 < len(grid[0])) or grid[x1][y1].is_barrier():
                return False
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return True
    except Exception as e:
        logging.error(f"Error in is_clear_path: {str(e)}")
        return False

def generate_maze(grid, rows):
    try:
        for row in grid:
            for spot in row:
                if spot.is_barrier():
                    spot.reset()

        num_clusters = max(3, rows // 8)
        max_cluster_size = 5
        protected = {(1, 1), (rows-1, rows-1)}
        buffer = 3
        protected_zone = set()
        for r in range(rows):
            for c in range(rows):
                if (h((r, c), (1, 1), "A*") <= buffer or
                    h((r, c), (rows-1, rows-1), "A*") <= buffer):
                    protected_zone.add((r, c))
        
        def place_cluster(center_r, center_c, size):
            for dr in range(-size // 2, size // 2 + 1):
                for dc in range(-size // 2, size // 2 + 1):
                    r, c = center_r + dr, center_c + dc
                    if (0 <= r < rows and 0 <= c < rows and
                        (r, c) not in protected_zone and
                        random.random() < 0.7):
                        grid[r][c].make_barrier()
                        logging.debug(f"Placed cluster barrier at ({r}, {c})")

        available_centers = [(r, c) for r in range(3, rows-3) for c in range(3, rows-3)
                            if (r, c) not in protected_zone]
        random.shuffle(available_centers)
        for i in range(min(num_clusters, len(available_centers))):
            r, c = available_centers[i]
            cluster_size = random.randint(2, max_cluster_size)
            place_cluster(r, c, cluster_size)

        temp_path = algorithm(lambda: None, grid, grid[1][1], grid[rows-1][rows-1], "A*", set())
        if not temp_path:
            logging.warning("No path exists, clearing some barriers")
            barriers = [(r, c) for r in range(rows) for c in range(rows) if grid[r][c].is_barrier()]
            random.shuffle(barriers)
            for r, c in barriers[:len(barriers)//2]:
                if (r, c) not in protected:
                    grid[r][c].reset()
            temp_path = algorithm(lambda: None, grid, grid[1][1], grid[rows-1][rows-1], "A*", set())
            if not temp_path:
                logging.error("Still no path after clearing, using empty grid")
                for row in grid:
                    for spot in row:
                        if spot.is_barrier():
                            spot.reset()

        logging.info(f"Unstructured maze generated with {num_clusters} shelf clusters")
    except Exception as e:
        logging.error(f"Error generating maze: {str(e)}")

def initialize_obstacles(grid, num_obstacles, rows, bias_to_path=False, static_ratio=0.5):
    try:
        obstacles = []
        new_static_positions = []
        used_positions = set([(1,1), (rows-1, rows-1)])
        available_positions = [(r, c) for r in range(rows) for c in range(rows)
                              if not grid[r][c].is_barrier() and (r, c) not in used_positions]
        
        if bias_to_path and path_positions:
            path_adjacent = set()
            for r, c in path_positions:
                path_adjacent.add((r, c))
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < rows and
                        not grid[nr][nc].is_barrier() and
                        not grid[nr][nc].is_start() and
                        not grid[nr][nc].is_end()):
                        path_adjacent.add((nr, nc))
            path_adjacent = list(path_adjacent - used_positions)
            random.shuffle(path_adjacent)
            path_obstacles = min(num_obstacles // 2, len(path_adjacent))
            for i in range(path_obstacles):
                if i < len(path_adjacent):
                    r, c = path_adjacent[i]
                    used_positions.add((r, c))
                    if random.random() < static_ratio:
                        grid[r][c].make_barrier()
                        new_static_positions.append((r, c))
                        logging.debug(f"Placed static obstacle on/near path at ({r}, {c})")
                    else:
                        obstacles.append({
                            "pos": [r, c],
                            "speed": random.uniform(0.5, 2.0),
                            "last_moved": 0
                        })
                        logging.debug(f"Placed dynamic obstacle on/near path at ({r}, {c})")
            available_positions = [p for p in available_positions if p not in used_positions]
        
        random.shuffle(available_positions)
        remaining_obstacles = num_obstacles - len(obstacles)
        for i in range(min(remaining_obstacles, len(available_positions))):
            r, c = available_positions[i]
            used_positions.add((r, c))
            if random.random() < static_ratio:
                grid[r][c].make_barrier()
                new_static_positions.append((r, c))
                logging.debug(f"Placed static obstacle at ({r}, {c})")
            else:
                obstacles.append({
                    "pos": [r, c],
                    "speed": random.uniform(0.5, 2.0),
                    "last_moved": 0
                })
                logging.debug(f"Placed dynamic obstacle at ({r}, {c})")
        
        logging.info(f"Initialized {len(obstacles)} dynamic and {len(new_static_positions)} static obstacles{' with path bias' if bias_to_path else ''}")
        return obstacles, new_static_positions
    except Exception as e:
        logging.error(f"Error initializing obstacles: {str(e)}")
        return [], []

def update_moving_obstacles(grid, obstacles, current_time):
    global dynamic_positions
    try:
        for r, c in dynamic_positions:
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                grid[r][c].clear_dynamic_obstacle()
        dynamic_positions.clear()

        directions = [
            (-3,0), (3,0), (0,-3), (0,3),
            (-2,0), (2,0), (0,-2), (0,2),
            (-1,0), (1,0), (0,-1), (0,1),
            (-3,-3), (-3,3), (3,-3), (3,3),
            (-2,-2), (-2,2), (2,-2), (2,2),
            (-1,-1), (-1,1), (1,-1), (1,1),
            (-3,-2), (-3,2), (3,-2), (3,2),
            (-2,-3), (-2,3), (2,-3), (2,3),
            (-3,-1), (-3,1), (3,-1), (3,1),
            (-1,-3), (-1,3), (1,-3), (1,3)
        ]
        for obs in obstacles:
            r, c = obs["pos"]
            speed = obs["speed"] * speed_scale
            last_moved = obs["last_moved"]
            move_interval = 1.0 / speed if speed > 0 else 1.0

            if current_time - last_moved < move_interval:
                dynamic_positions.add((r, c))
                if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                    grid[r][c].set_dynamic_obstacle()
                continue

            random.shuffle(directions)
            moved = False
            for dr, dc in directions[:5]:
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < len(grid) and 0 <= new_c < len(grid[0]) and
                    not grid[new_r][new_c].is_barrier() and
                    not grid[new_r][new_c].is_start() and
                    not grid[new_r][new_c].is_end() and
                    (new_r, new_c) not in dynamic_positions and
                    is_clear_path(grid, r, c, new_r, new_c)):
                    obs["pos"] = [new_r, new_c]
                    obs["last_moved"] = current_time
                    dynamic_positions.add((new_r, new_c))
                    grid[new_r][new_c].set_dynamic_obstacle()
                    moved = True
                    logging.debug(f"Obstacle moved from ({r},{c}) to ({new_r},{new_c})")
                    break

            if not moved:
                dynamic_positions.add((r, c))
                if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                    grid[r][c].set_dynamic_obstacle()

    except Exception as e:
        logging.error(f"Error updating obstacles: {str(e)}")

def make_grid(rows, width):
    try:
        grid = []
        gap = width // rows
        for i in range(rows):
            grid.append([])
            for j in range(rows):
                grid[i].append(Spot(i, j, gap, rows))
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

def draw_ui(win):
    try:
        pygame.draw.rect(win, DARK_GREY, (WIDTH, 0, UI_WIDTH, HEIGHT))
        buttons = [
            {"text": "Reset Grid", "rect": pygame.Rect(WIDTH + 20, 20, 160, 40), "action": "reset"},
            {"text": "Start/Stop", "rect": pygame.Rect(WIDTH + 20, 70, 160, 40), "action": "toggle_anim"},
            {"text": f"Algo: {selected_algorithm}", "rect": pygame.Rect(WIDTH + 20, 120, 160, 40), "action": "switch_algo"},
            {"text": f"{'Hide' if show_sensor_range else 'Show'} Sensor", "rect": pygame.Rect(WIDTH + 20, 170, 160, 40), "action": "toggle_sensor"},
            {"text": f"{'Disable' if use_maze else 'Enable'} Maze", "rect": pygame.Rect(WIDTH + 20, 220, 160, 40), "action": "toggle_maze"},
            {"text": "Add Obs to Path", "rect": pygame.Rect(WIDTH + 20, 270, 160, 40), "action": "add_obstacles_to_path"},
        ]
        for btn in buttons:
            pygame.draw.rect(win, GREY, btn["rect"])
            text = FONT.render(btn["text"], True, WHITE)
            win.blit(text, (btn["rect"].x + 10, btn["rect"].y + 10))

        sliders = [
            {"text": f"Grid Size: {rows}", "rect": pygame.Rect(WIDTH + 20, 320, 160, 20), "range": (10, 50), "value": rows, "id": "grid_size"},
            {"text": f"Obs Speed: {speed_scale:.1f}x", "rect": pygame.Rect(WIDTH + 20, 360, 160, 20), "range": (0.5, 3.0), "value": speed_scale, "id": "speed_scale"},
            {"text": f"Anim Delay: {animation_delay:.1f}s", "rect": pygame.Rect(WIDTH + 20, 400, 160, 20), "range": (0.1, 1.0), "value": animation_delay, "id": "anim_delay"},
            {"text": f"Obstacles: {num_obstacles}", "rect": pygame.Rect(WIDTH + 20, 440, 160, 20), "range": (1, 20), "value": num_obstacles, "id": "num_obstacles"},
            {"text": f"Sensor Range: {SENSOR_RANGE}", "rect": pygame.Rect(WIDTH + 20, 480, 160, 20), "range": (1, 5), "value": SENSOR_RANGE, "id": "sensor_range"},
        ]
        for slider in sliders:
            pygame.draw.rect(win, GREY, slider["rect"])
            min_val, max_val = slider["range"]
            val = slider["value"]
            handle_x = slider["rect"].x + ((val - min_val) / (max_val - min_val)) * slider["rect"].width
            pygame.draw.circle(win, WHITE, (int(handle_x), slider["rect"].y + 10), 8)
            text = FONT.render(slider["text"], True, WHITE)
            win.blit(text, (slider["rect"].x, slider["rect"].y - 20))

        text = FONT.render(f"Replans: {replan_count}", True, WHITE)
        win.blit(text, (WIDTH + 20, 520))
        text = FONT.render(f"Path Length: {path_length}", True, WHITE)
        win.blit(text, (WIDTH + 20, 540))

        return buttons, sliders
    except Exception as e:
        logging.error(f"Error drawing UI: {str(e)}")
        return [], []

def draw_sensor_range(win, grid, agent_pos, rows, width):
    try:
        if not show_sensor_range:
            return
        gap = width // rows
        r, c = agent_pos
        if 0 <= r < rows and 0 <= c < rows:
            spot = grid[r][c]
            x, y = spot.x, spot.y
            range_rect = pygame.Rect(
                x - SENSOR_RANGE * gap, y - SENSOR_RANGE * gap,
                (2 * SENSOR_RANGE + 1) * gap, (2 * SENSOR_RANGE + 1) * gap
            )
            pygame.draw.rect(win, CYAN, range_rect, 2)
    except Exception as e:
        logging.error(f"Error drawing sensor range: {str(e)}")

def algorithm(draw, grid, start, end, algorithm, detected_obstacles, avoid_dir=None):
    global path_positions
    try:
        for r, c in path_positions:
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                spot = grid[r][c]
                if not (spot.is_start() or spot.is_end() or spot.visited or (r, c) in dynamic_positions):
                    spot.reset()
        path_positions = []

        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, start))
        came_from = {}
        g_score = {spot: float("inf") for row in grid for spot in row}
        f_score = {spot: float("inf") for row in grid for spot in row}
        g_score[start] = 0
        f_score[start] = h(start.get_pos(), end.get_pos(), algorithm)
        open_set_hash = {start}

        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("QUIT event received during algorithm")
                    return []

            current = open_set.get()[2]
            open_set_hash.remove(current)

            if current == end:
                start.make_start()
                path = reconstruct_path(came_from, end, grid)
                end.make_end()
                logging.info("Path found")
                return path

            for neighbor in current.neighbors:
                if (neighbor.row, neighbor.col) in detected_obstacles:
                    continue

                penalty = calculate_penalty(current, neighbor, moving_obstacles, detected_obstacles)
                temp_g_score = g_score[current] + penalty
                if avoid_dir:
                    nr, nc = neighbor.get_pos()
                    dr, dc = avoid_dir
                    if (nr - current.row, nc - current.col) == (dr, dc):
                        temp_g_score += 10
                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos(), algorithm)
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        logging.debug(f"Added neighbor ({neighbor.row}, {neighbor.col}) to open set")
            draw()
        logging.warning("No path found")
        return []
    except Exception as e:
        logging.error(f"Error in algorithm: {str(e)}\n{traceback.format_exc()}")
        return []

def animate_path_on_grid_gen_with_replanning(win, grid, path_positions):
    global agent_trail, animation_delay, replan_count
    try:
        path = path_positions[:]
        last_replan_check = time.time()
        while path:
            current_time = time.time()
            current_pos = agent_trail[-1] if agent_trail else (1, 1)
            detected_obstacles = get_obstacles_in_range(current_pos, moving_obstacles)

            # Check next position for static or dynamic obstacles
            if path:
                next_r, next_c = path[0]
                if (next_r, next_c) in detected_obstacles or (h(current_pos, (next_r, next_c), "A*") <= SENSOR_RANGE and grid[next_r][next_c].is_barrier()):
                    logging.info(f"{'Dynamic' if (next_r, next_c) in detected_obstacles else 'Static'} obstacle at next position ({next_r}, {next_c}), replanning...")
                    replan_count += 1
                    start_pos = current_pos
                    if not (0 <= start_pos[0] < len(grid) and 0 <= start_pos[1] < len(grid[0])):
                        logging.error(f"Invalid start position: {start_pos}")
                        yield "DONE"
                    start = grid[start_pos[0]][start_pos[1]]
                    end = grid[rows-1][rows-1]
                    avoid_dir = None
                    min_dist = float('inf')
                    for obs in moving_obstacles:
                        obs_pos = tuple(obs["pos"])
                        if obs_pos in detected_obstacles:
                            dist = h(start_pos, obs_pos, selected_algorithm)
                            if dist < min_dist:
                                min_dist = dist
                                dr = start_pos[0] - obs_pos[0]
                                dc = start_pos[1] - obs_pos[1]
                                avoid_dir = (0 if dr == 0 else (-1 if dr > 0 else 1),
                                             0 if dc == 0 else (-1 if dc > 0 else 1))
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid, detected_obstacles)
                    path_positions = algorithm(
                        lambda: (draw_grid_only(win, grid, WIDTH, rows, current_pos), pygame.display.update()),
                        grid, start, end, selected_algorithm, detected_obstacles, avoid_dir
                    )
                    if not path_positions:
                        logging.warning("No new path found, stopping animation")
                        yield "DONE"
                        return
                    path = path_positions[:]
                    continue

            # Look-ahead for obstacles within SENSOR_RANGE
            look_ahead = min(len(path), SENSOR_RANGE)
            obstacle_detected = False
            for i in range(look_ahead):
                if i < len(path):
                    r, c = path[i][0], path[i][1]
                    if (r, c) in detected_obstacles or grid[r][c].is_barrier():
                        logging.info(f"{'Dynamic' if (r, c) in detected_obstacles else 'Static'} obstacle detected at path position {i} ({r}, {c}), replanning...")
                        obstacle_detected = True
                        break

            # Check predicted dynamic obstacles
            if not obstacle_detected and (current_time - last_replan_check >= animation_delay):
                predicted_positions = predict_obstacle_positions(grid, moving_obstacles, current_time, 3, rows, current_pos)
                for i in range(look_ahead):
                    if i < len(path) and (path[i][0], path[i][1]) in predicted_positions:
                        logging.info(f"Predicted obstacle at path position {i} ({path[i][0], path[i][1]}), replanning...")
                        obstacle_detected = True
                        break
                last_replan_check = current_time

            if obstacle_detected:
                replan_count += 1
                start_pos = current_pos
                if not (0 <= start_pos[0] < len(grid) and 0 <= start_pos[1] < len(grid[0])):
                    logging.error(f"Invalid start position: {start_pos}")
                    yield "DONE"
                    return
                start = grid[start_pos[0]][start_pos[1]]
                end = grid[rows-1][rows-1]
                avoid_dir = None
                min_dist = float('inf')
                for obs in moving_obstacles:
                    obs_pos = tuple(obs["pos"])
                    if obs_pos in detected_obstacles:
                        dist = h(start_pos, obs_pos, selected_algorithm)
                        if dist < min_dist:
                            min_dist = dist
                            dr = start_pos[0] - obs_pos[0]
                            dc = start_pos[1] - obs_pos[1]
                            avoid_dir = (0 if dr == 0 else (-1 if dr > 0 else 1),
                                         0 if dc == 0 else (-1 if dc > 0 else 1))
                for row in grid:
                    for spot in row:
                        spot.update_neighbors(grid, detected_obstacles)
                path_positions = algorithm(
                    lambda: (draw_grid_only(win, grid, WIDTH, rows, current_pos), pygame.display.update()),
                    grid, start, end, selected_algorithm, detected_obstacles, avoid_dir
                )
                if not path_positions:
                    logging.warning("No new path found, stopping animation")
                    yield "DONE"
                    return
                path = path_positions[:]
                continue

            row, col = path.pop(0)
            if (row, col) not in agent_trail:
                agent_trail.append((row, col))
                grid[row][col].make_visited()

            draw_grid_only(win, grid, WIDTH, rows, current_pos)
            time.sleep(animation_delay)
            yield

        end_row, end_col = rows-1, rows-1
        if (end_row, end_col) not in agent_trail:
            agent_trail.append((end_row, end_col))
            grid[end_row][end_col].make_visited()
            draw_grid_only(win, grid, WIDTH, rows, (end_row, end_col))
            time.sleep(animation_delay)

        logging.info("Animation completed")
        yield "DONE"
    except Exception as e:
        logging.error(f"Error in animation: {str(e)}\n{traceback.format_exc()}")
        yield "DONE"

def draw_grid_only(win, grid, width, rows, agent_pos=None):
    try:
        win.fill(WHITE)
        detected_obstacles = get_obstacles_in_range(agent_pos, moving_obstacles) if agent_pos else set()
        for row in grid:
            for spot in row:
                spot.draw(win)
        for r, c in dynamic_positions:
            if 0 <= r < rows and 0 <= c < rows:
                spot = grid[r][c]
                if not spot.is_barrier() and not spot.is_start() and not spot.is_end():
                    spot.color = YELLOW if (r, c) in detected_obstacles else GREY
                    pygame.draw.rect(win, spot.color, (spot.x, spot.y, spot.width, spot.width))
                    logging.debug(f"Drawing dynamic obstacle at ({r}, {c}) with color {'YELLOW' if (r, c) in detected_obstacles else 'GREY'}")
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

def reset_grid(grid, rows):
    global agent_trail, moving_obstacles, dynamic_positions, replan_count, path_positions, path_length
    try:
        logging.debug(f"Resetting grid with use_maze={use_maze}")
        agent_trail = []
        dynamic_positions = set()
        replan_count = 0
        path_positions = []
        path_length = 0
        grid = make_grid(rows, WIDTH)
        grid[1][1].make_start()
        grid[rows-1][rows-1].make_end()
        if use_maze:
            generate_maze(grid, rows)
        moving_obstacles, _ = initialize_obstacles(grid, num_obstacles, rows, bias_to_path=False, static_ratio=0.0)
        for obs in moving_obstacles:
            r, c = obs["pos"]
            dynamic_positions.add((r, c))
            grid[r][c].set_dynamic_obstacle()
        return grid
    except Exception as e:
        logging.error(f"Error resetting grid: {str(e)}")
        return grid

def pygame_loop(win, width):
    global rows, animation_delay, speed_scale, selected_algorithm, dragging_slider, moving_obstacles, replan_count, path_positions, num_obstacles, SENSOR_RANGE, show_sensor_range, use_maze, path_length
    try:
        grid = make_grid(rows, width)
        start = grid[1][1]
        end = grid[rows-1][rows-1]
        start.make_start()
        end.make_end()
        if use_maze:
            generate_maze(grid, rows)
        moving_obstacles, _ = initialize_obstacles(grid, num_obstacles, rows, bias_to_path=False, static_ratio=0.0)
        for obs in moving_obstacles:
            r, c = obs["pos"]
            dynamic_positions.add((r, c))
            grid[r][c].set_dynamic_obstacle()

        clock = pygame.time.Clock()
        running = True
        agent_animator = None
        setting_start = False
        setting_end = False
        initial_path_computed = False

        gap = width // rows
        while running:
            clock.tick(15 if agent_animator else 10)
            current_time = time.time()

            update_moving_obstacles(grid, moving_obstacles, current_time)
            for row in grid:
                for spot in row:
                    spot.update_neighbors(grid, set() if not initial_path_computed else get_obstacles_in_range(agent_trail[-1] if agent_trail else (1, 1), moving_obstacles))

            buttons, sliders = draw_ui(win)
            agent_pos = agent_trail[-1] if agent_trail else (1, 1)
            draw_grid_only(win, grid, width, rows, agent_pos)

            if agent_animator:
                try:
                    result = next(agent_animator)
                    if result == "DONE":
                        logging.info("Animation finished")
                        agent_animator = None
                except StopIteration:
                    logging.info("Animation iterator exhausted")
                    agent_animator = None
                except Exception as e:
                    logging.error(f"Animation error: {str(e)}\n{traceback.format_exc()}")
                    agent_animator = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("User closed window")
                    running = False
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    logging.debug(f"Mouse clicked at position: {pos}")
                    for btn in buttons:
                        if btn["rect"].collidepoint(pos):
                            if btn["action"] == "reset":
                                grid = reset_grid(grid, rows)
                                start = grid[1][1]
                                end = grid[rows-1][rows-1]
                                agent_animator = None
                                path_positions = []
                                initial_path_computed = False
                                logging.info("Grid reset")
                            elif btn["action"] == "toggle_anim":
                                if agent_animator:
                                    agent_animator = None
                                    logging.info("Animation stopped")
                                elif path_positions:
                                    agent_animator = animate_path_on_grid_gen_with_replanning(win, grid, path_positions)
                                    logging.info("Animation started")
                                else:
                                    logging.warning("No path to animate")
                            elif btn["action"] == "switch_algo":
                                algorithms = ["A*", "Dijkstra"]
                                current_idx = algorithms.index(selected_algorithm)
                                selected_algorithm = algorithms[(current_idx + 1) % len(algorithms)]
                                logging.info(f"Algorithm switched to {selected_algorithm}")
                            elif btn["action"] == "toggle_sensor":
                                show_sensor_range = not show_sensor_range
                                logging.info(f"Sensor range visualization {'enabled' if show_sensor_range else 'disabled'}")
                            elif btn["action"] == "toggle_maze":
                                use_maze = not use_maze
                                logging.info(f"Maze generation {'enabled' if use_maze else 'disabled'}")
                            elif btn["action"] == "add_obstacles_to_path":
                                if path_positions:
                                    num_obstacles += 5
                                    num_obstacles = min(num_obstacles, 20)
                                    moving_obstacles, new_static_positions = initialize_obstacles(grid, num_obstacles, rows, bias_to_path=True, static_ratio=0.5)
                                    dynamic_positions.clear()
                                    for obs in moving_obstacles:
                                        r, c = obs["pos"]
                                        dynamic_positions.add((r, c))
                                        grid[r][c].set_dynamic_obstacle()
                                    for r, c in new_static_positions:
                                        grid[r][c].make_barrier()
                                    logging.info(f"Added obstacles to path without replanning, total: {num_obstacles} ({len(new_static_positions)} static, {len(moving_obstacles)} dynamic)")
                                else:
                                    logging.warning("No path to add obstacles to")

                    for slider in sliders:
                        if slider["rect"].collidepoint(pos):
                            dragging_slider = slider["id"]
                            logging.info(f"Started dragging slider: {dragging_slider}")

                    if pos[0] < WIDTH:
                        col = pos[0] // gap
                        row = pos[1] // gap
                        logging.debug(f"Mapped click to grid cell: ({row}, {col}), gap={gap}")
                        if 0 <= row < rows and 0 <= col < rows:
                            spot = grid[row][col]
                            if event.button == 1:
                                if setting_start:
                                    if not spot.is_end() and not spot.is_barrier() and (row, col) not in dynamic_positions:
                                        start.reset()
                                        start = spot
                                        start.make_start()
                                        setting_start = False
                                        initial_path_computed = False
                                        logging.info(f"Start point set to ({row}, {col})")
                                    else:
                                        logging.warning(f"Cannot set start at ({row}, {col})")
                                elif not spot.is_end() and not spot.is_start() and (row, col) not in dynamic_positions:
                                    spot.make_barrier()
                                    if (row, col) in path_positions:
                                        path_positions.remove((row, col))
                                    initial_path_computed = False
                                    logging.info(f"Barrier placed at ({row}, {col})")
                            elif event.button == 3:
                                if setting_end:
                                    if not spot.is_start() and not spot.is_barrier() and (row, col) not in dynamic_positions:
                                        end.reset()
                                        end = spot
                                        end.make_end()
                                        setting_end = False
                                        initial_path_computed = False
                                        logging.info(f"End point set to ({row}, {col})")
                                    else:
                                        logging.warning(f"Cannot set end at ({row}, {col})")
                                elif not spot.is_start() and not spot.is_end():
                                    spot.reset()
                                    if (row, col) in dynamic_positions:
                                        dynamic_positions.remove((row, col))
                                        moving_obstacles = [obs for obs in moving_obstacles if tuple(obs["pos"]) != (row, col)]
                                    initial_path_computed = False
                                    logging.info(f"Barrier or obstacle removed at ({row}, {col})")
                            for r in grid:
                                for s in r:
                                    s.update_neighbors(grid, set() if not initial_path_computed else get_obstacles_in_range(agent_trail[-1] if agent_trail else (1, 1), moving_obstacles))
                            moving_obstacles, _ = initialize_obstacles(grid, num_obstacles, rows, bias_to_path=initial_path_computed, static_ratio=0.0)
                            dynamic_positions.clear()
                            for obs in moving_obstacles:
                                r, c = obs["pos"]
                                dynamic_positions.add((r, c))
                                grid[r][c].set_dynamic_obstacle()

                if event.type == pygame.MOUSEBUTTONUP:
                    if dragging_slider:
                        logging.info(f"Stopped dragging slider: {dragging_slider}")
                    dragging_slider = None

                if event.type == pygame.MOUSEMOTION and dragging_slider:
                    pos = pygame.mouse.get_pos()
                    for slider in sliders:
                        if slider["id"] == dragging_slider:
                            min_val, max_val = slider["range"]
                            x = max(slider["rect"].x, min(slider["rect"].x + slider["rect"].width, pos[0]))
                            value = min_val + (x - slider["rect"].x) / slider["rect"].width * (max_val - min_val)
                            if slider["id"] == "grid_size":
                                new_rows = int(round(value))
                                if new_rows != rows:
                                    rows = new_rows
                                    grid = reset_grid(grid, rows)
                                    start = grid[1][1]
                                    end = grid[rows-1][rows-1]
                                    gap = WIDTH // rows
                                    agent_animator = None
                                    path_positions = []
                                    initial_path_computed = False
                                    logging.info(f"Grid size changed to {rows}x{rows}")
                            elif slider["id"] == "speed_scale":
                                speed_scale = round(value, 1)
                                logging.info(f"Obstacle speed scale set to {speed_scale}x")
                            elif slider["id"] == "anim_delay":
                                animation_delay = round(value, 1)
                                logging.info(f"Animation delay set to {animation_delay}s")
                            elif slider["id"] == "num_obstacles":
                                new_num = int(round(value))
                                if new_num != num_obstacles:
                                    num_obstacles = new_num
                                    moving_obstacles, new_static_positions = initialize_obstacles(grid, num_obstacles, rows, bias_to_path=initial_path_computed, static_ratio=0.5 if initial_path_computed else 0.0)
                                    dynamic_positions.clear()
                                    for obs in moving_obstacles:
                                        r, c = obs["pos"]
                                        dynamic_positions.add((r, c))
                                        grid[r][c].set_dynamic_obstacle()
                                    for r, c in new_static_positions:
                                        grid[r][c].make_barrier()
                                    logging.info(f"Number of obstacles set to {num_obstacles} without replanning ({len(new_static_positions)} static, {len(moving_obstacles)} dynamic)")
                            elif slider["id"] == "sensor_range":
                                SENSOR_RANGE = int(round(value))
                                logging.info(f"Sensor range set to {SENSOR_RANGE}")
                            slider["value"] = value

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        setting_start = True
                        setting_end = False
                        logging.info("Start point selection enabled")
                    elif event.key == pygame.K_e:
                        setting_end = True
                        setting_start = False
                        logging.info("End point selection enabled")
                    elif event.key == pygame.K_SPACE:
                        for row in grid:
                            for spot in row:
                                spot.update_neighbors(grid, set())
                        path_positions = algorithm(
                            lambda: (draw_grid_only(win, grid, WIDTH, rows, agent_trail[-1] if agent_trail else (1, 1)), pygame.display.update()),
                            grid, start, end, selected_algorithm, set()
                        )
                        if path_positions:
                            draw_grid_only(win, grid, WIDTH, rows, agent_trail[-1] if agent_trail else (1, 1))
                            initial_path_computed = True
                            logging.info("Path computed")
                        else:
                            logging.warning("No path found")

        pygame.display.update()

    except Exception as e:
        logging.error(f"Main loop error: {str(e)}\n{traceback.format_exc()}")
    finally:
        pygame.quit()
        logging.info("Program terminated")

if __name__ == '__main__':
    pygame_loop(WIN, WIDTH)