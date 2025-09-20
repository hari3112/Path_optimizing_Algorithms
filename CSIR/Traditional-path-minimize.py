import pygame
import math
from queue import PriorityQueue
import time
import logging
import traceback
import random
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

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
LIGHT_BLUE = (173, 216, 230)

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
original_path_positions = []
local_detour_path = []
SENSOR_RANGE = 3
show_sensor_range = True
use_maze = True
path_length = 0
total_path_length = 0
global_plan_time = 0.0
local_plan_time = 0.0
global_nodes_expanded = 0
local_nodes_expanded = 0
simulation_time = 0.0
success = False
metrics_store = []

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
    def is_local_path(self): return self.base_color == LIGHT_BLUE
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
    def make_local_path(self):
        self.color = LIGHT_BLUE
        self.base_color = LIGHT_BLUE
    def make_visited(self): self.visited = True
    def set_dynamic_obstacle(self):
        self.color = YELLOW
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

def find_obstacles_in_path_ahead(current_pos, path_ahead, detected_obstacles, grid):
    obstacles_in_path = []
    for i, (r, c) in enumerate(path_ahead):
        if (r, c) in detected_obstacles or grid[r][c].is_barrier():
            obstacles_in_path.append(i)
        if h(current_pos, (r, c), "A*") > SENSOR_RANGE:
            break
    return obstacles_in_path

def find_rejoin_point(current_pos, original_path, obstacle_positions, grid):
    for i, (r, c) in enumerate(original_path):
        if h(current_pos, (r, c), "A*") < 4:
            continue
        position_clear = True
        for obs_r, obs_c in obstacle_positions:
            if h((r, c), (obs_r, obs_c), "A*") <= 2:
                position_clear = False
                break
        if position_clear and not grid[r][c].is_barrier():
            logging.info(f"Found rejoin point at ({r}, {c}) at original path index {i}")
            return (r, c), original_path[i:]
    end_pos = (rows-1, rows-1)
    logging.warning("No suitable rejoin point found, using end point")
    return end_pos, [end_pos]

def local_replan(grid, start_pos, rejoin_point, detected_obstacles, avoid_dir=None):
    global local_plan_time, local_nodes_expanded
    start_time = time.perf_counter()
    try:
        start = grid[start_pos[0]][start_pos[1]]
        end = grid[rejoin_point[0]][rejoin_point[1]]
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, start))
        came_from = {}
        g_score = {spot: float("inf") for row in grid for spot in row}
        f_score = {spot: float("inf") for row in grid for spot in row}
        g_score[start] = 0
        f_score[start] = h(start.get_pos(), end.get_pos(), selected_algorithm)
        open_set_hash = {start}
        nodes = 0

        if not start.neighbors:
            logging.warning(f"No neighbors for start position {start_pos}")
            replan_time = time.perf_counter() - start_time
            local_plan_time += replan_time
            logging.info(f"Replan {replan_count}: time={replan_time:.6f}s, nodes={nodes}, failed: no neighbors")
            return []

        while not open_set.empty():
            current = open_set.get()[2]
            open_set_hash.remove(current)
            if current == end:
                local_path = []
                while current in came_from:
                    current = came_from[current]
                    pos = current.get_pos()
                    local_path.append(pos)
                local_path.reverse()
                replan_time = time.perf_counter() - start_time
                local_plan_time += replan_time
                local_nodes_expanded += nodes
                logging.info(f"Replan {replan_count}: time={replan_time:.6f}s, nodes={nodes}, path={local_path}")
                return local_path
            for neighbor in current.neighbors:
                if (neighbor.row, neighbor.col) in detected_obstacles:
                    continue
                penalty = calculate_penalty(current, neighbor, moving_obstacles, detected_obstacles)
                temp_g_score = g_score[current] + penalty
                if avoid_dir:
                    nr, nc = neighbor.get_pos()
                    dr, dc = avoid_dir
                    if (nr - current.row, nc - current.col) == (dr, dc):
                        temp_g_score += 50
                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos(), selected_algorithm)
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        nodes += 1
        replan_time = time.perf_counter() - start_time
        local_plan_time += replan_time
        local_nodes_expanded += nodes
        logging.info(f"Replan {replan_count}: time={replan_time:.6f}s, nodes={nodes}, failed: no path to rejoin point {rejoin_point}")
        return []
    except Exception as e:
        replan_time = time.perf_counter() - start_time
        local_plan_time += replan_time
        logging.error(f"Error in local replanning: {str(e)}")
        logging.info(f"Replan {replan_count}: time={replan_time:.6f}s, nodes={nodes}, failed: exception")
        return []

def reconstruct_path(came_from, current, grid):
    global path_positions, original_path_positions, path_length
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
        original_path_positions = path.copy()
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
            move_interval = min(0.5 / speed, animation_delay * 2)
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
            move_interval = min(0.5 / speed, animation_delay * 2)
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
        has_astar = False
        has_dijkstra = False
        # Check CSV
        if os.path.isfile("simulation_metrics.csv"):
            try:
                with open("simulation_metrics.csv", 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("Algorithm") == "A*" and row.get("Success") == "True":
                            has_astar = True
                        elif row.get("Algorithm") == "Dijkstra" and row.get("Success") == "True":
                            has_dijkstra = True
            except Exception as e:
                logging.warning(f"Error reading CSV for button enabling: {str(e)}")
        # Check in-memory store
        for row in metrics_store:
            if row.get("Algorithm") == "A*" and row.get("Success") is True:
                has_astar = True
            elif row.get("Algorithm") == "Dijkstra" and row.get("Success") is True:
                has_dijkstra = True
        logging.info(f"Show Comparison button: has_astar={has_astar}, has_dijkstra={has_dijkstra}, enabled={has_astar and has_dijkstra}")
        buttons = [
            {"text": "Reset Grid", "rect": pygame.Rect(WIDTH + 20, 20, 160, 40), "action": "reset"},
            {"text": "Start/Stop", "rect": pygame.Rect(WIDTH + 20, 70, 160, 40), "action": "toggle_anim"},
            {"text": f"Algo: {selected_algorithm}", "rect": pygame.Rect(WIDTH + 20, 120, 160, 40), "action": "switch_algo"},
            {"text": f"{'Hide' if show_sensor_range else 'Show'} Sensor", "rect": pygame.Rect(WIDTH + 20, 170, 160, 40), "action": "toggle_sensor"},
            {"text": f"{'Disable' if use_maze else 'Enable'} Maze", "rect": pygame.Rect(WIDTH + 20, 220, 160, 40), "action": "toggle_maze"},
            {"text": "Add Obs to Path", "rect": pygame.Rect(WIDTH + 20, 270, 160, 40), "action": "add_obstacles_to_path"},
            {"text": "Clear Metrics", "rect": pygame.Rect(WIDTH + 20, 520, 160, 40), "action": "clear_metrics"},
            {"text": "Show Metrics", "rect": pygame.Rect(WIDTH + 20, 570, 160, 40), "action": "show_metrics"},
            {"text": "Show Comparison", "rect": pygame.Rect(WIDTH + 20, 620, 160, 40), "action": "show_comparison", "enabled": has_astar and has_dijkstra},
        ]
        for btn in buttons:
            color = GREY if btn.get("enabled", True) else DARK_GREY
            pygame.draw.rect(win, color, btn["rect"])
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
        text = FONT.render(f"Total Path Length: {total_path_length}", True, WHITE)
        win.blit(text, (WIDTH + 20, 480))
        text = FONT.render(f"Current Path Len: {path_length}", True, WHITE)
        win.blit(text, (WIDTH + 20, 460))
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

def draw_results_screen(win):
    try:
        win.fill(DARK_GREY)
        title = FONT.render("Simulation Results", True, WHITE)
        win.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
        
        metrics = [
            ("Total Path Length", f"{total_path_length}"),
            ("Global Plan Time (s)", f"{global_plan_time:.6f}"),
            ("Local Plan Time (s)", f"{local_plan_time:.6f}"),
            ("Total Nodes Expanded", f"{global_nodes_expanded + local_nodes_expanded}"),
            ("Success", "Yes" if success else "No"),
            ("Simulation Time (s)", f"{simulation_time:.6f}")
        ]
        
        table_x = 100
        table_y = 100
        row_height = 40
        col1_width = 300
        col2_width = 150
        
        pygame.draw.rect(win, GREY, (table_x, table_y, col1_width + col2_width, row_height))
        header1 = FONT.render("Metric", True, WHITE)
        header2 = FONT.render("Value", True, WHITE)
        win.blit(header1, (table_x + 10, table_y + 10))
        win.blit(header2, (table_x + col1_width + 10, table_y + 10))
        
        for i, (metric, value) in enumerate(metrics):
            y = table_y + (i + 1) * row_height
            pygame.draw.rect(win, GREY if i % 2 == 0 else DARK_GREY, (table_x, y, col1_width + col2_width, row_height))
            metric_text = FONT.render(metric, True, WHITE)
            value_text = FONT.render(value, True, WHITE)
            win.blit(metric_text, (table_x + 10, y + 10))
            win.blit(value_text, (table_x + col1_width + 10, y + 10))
        
        back_button = {"text": "Back", "rect": pygame.Rect(WIDTH // 2 - 80, HEIGHT - 100, 160, 40), "action": "back"}
        pygame.draw.rect(win, GREY, back_button["rect"])
        text = FONT.render(back_button["text"], True, WHITE)
        win.blit(text, (back_button["rect"].x + 10, back_button["rect"].y + 10))
        
        pygame.display.update()
        return [back_button]
    except Exception as e:
        logging.error(f"Error drawing results screen: {str(e)}")
        return []

def save_metrics_to_csv():
    try:
        filename = "simulation_metrics.csv"
        file_exists = os.path.isfile(filename)
        headers = [
            "Timestamp", "Algorithm", "Grid Size", "Num Obstacles", "Sensor Range", 
            "Obstacle Speed", "Total Path Length", "Global Plan Time", 
            "Local Plan Time", "Total Nodes Expanded", "Success", "Simulation Time"
        ]
        data = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Algorithm": selected_algorithm,
            "Grid Size": rows,
            "Num Obstacles": num_obstacles,
            "Sensor Range": SENSOR_RANGE,
            "Obstacle Speed": speed_scale,
            "Total Path Length": total_path_length,
            "Global Plan Time": global_plan_time,
            "Local Plan Time": local_plan_time,
            "Total Nodes Expanded": global_nodes_expanded + local_nodes_expanded,
            "Success": success,
            "Simulation Time": simulation_time
        }
        metrics_store.append(data)
        logging.info(f"Metrics stored in memory: Algorithm={selected_algorithm}, Success={success}")
        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerow([data[h] for h in headers])
            logging.info(f"Metrics saved to {filename}: Algorithm={selected_algorithm}, Success={success}")
        except Exception as e:
            logging.warning(f"Failed to save metrics to CSV, using in-memory store: {str(e)}")
    except Exception as e:
        logging.error(f"Error in save_metrics_to_csv: {str(e)}")

def generate_comparison_plot():
    try:
        astar_data = None
        dijkstra_data = None
        # Try CSV first
        if os.path.isfile("simulation_metrics.csv"):
            with open("simulation_metrics.csv", 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                logging.info(f"Read {len(rows)} rows from simulation_metrics.csv")
                for row in reversed(rows):
                    logging.debug(f"Processing row: {row}")
                    if row.get("Algorithm") == "A*" and row.get("Success") == "True" and astar_data is None:
                        astar_data = row
                        logging.info(f"Selected A* data from CSV: {astar_data}")
                    elif row.get("Algorithm") == "Dijkstra" and row.get("Success") == "True" and dijkstra_data is None:
                        dijkstra_data = row
                        logging.info(f"Selected Dijkstra data from CSV: {dijkstra_data}")
                    if astar_data and dijkstra_data:
                        break
        # Fallback to in-memory store
        if not astar_data or not dijkstra_data:
            logging.info("Checking in-memory metrics store")
            for row in reversed(metrics_store):
                logging.debug(f"Processing in-memory row: {row}")
                if row.get("Algorithm") == "A*" and row.get("Success") is True and astar_data is None:
                    astar_data = row
                    logging.info(f"Selected A* data from memory: {astar_data}")
                elif row.get("Algorithm") == "Dijkstra" and row.get("Success") is True and dijkstra_data is None:
                    dijkstra_data = row
                    logging.info(f"Selected Dijkstra data from memory: {dijkstra_data}")
                if astar_data and dijkstra_data:
                    break
        
        if not astar_data or not dijkstra_data:
            logging.warning(f"Insufficient data: has_astar={bool(astar_data)}, has_dijkstra={bool(dijkstra_data)}")
            return None
        
        metrics = [
            "Total Path Length", "Global Plan Time", 
            "Local Plan Time", "Total Nodes Expanded", "Simulation Time"
        ]
        astar_values = [
            float(astar_data[m]) if m in ["Global Plan Time", "Local Plan Time", "Simulation Time"] 
            else int(astar_data[m]) 
            for m in metrics
        ]
        dijkstra_values = [
            float(dijkstra_data[m]) if m in ["Global Plan Time", "Local Plan Time", "Simulation Time"] 
            else int(dijkstra_data[m]) 
            for m in metrics
        ]
        
        logging.info(f"A* values: {dict(zip(metrics, astar_values))}")
        logging.info(f"Dijkstra values: {dict(zip(metrics, dijkstra_values))}")
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, astar_values, width, label="A*", color="#FF9999")
        ax.bar(x + width/2, dijkstra_values, width, label="Dijkstra", color="#66B2FF")
        
        ax.set_ylabel("Value")
        ax.set_title("A* vs Dijkstra Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.legend()
        
        for i, v in enumerate(astar_values):
            ax.text(i - width/2, v, f"{v:.4f}" if isinstance(v, float) else str(v), 
                    ha="center", va="bottom")
        for i, v in enumerate(dijkstra_values):
            ax.text(i + width/2, v, f"{v:.4f}" if isinstance(v, float) else str(v), 
                    ha="center", va="bottom")
        
        plt.tight_layout()
        plot_file = "comparison_plot.png"
        try:
            plt.savefig(plot_file)
            logging.info(f"Comparison plot saved to {plot_file}")
        except Exception as e:
            logging.warning(f"Failed to save plot, using in-memory buffer: {str(e)}")
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_file = buf
            logging.info("Plot stored in memory buffer")
        plt.close()
        return plot_file
    except Exception as e:
        logging.error(f"Error generating comparison plot: {str(e)}")
        return None

def draw_comparison_screen(win):
    try:
        plot_file = generate_comparison_plot()
        if not plot_file:
            logging.warning("No comparison plot available")
            win.fill(DARK_GREY)
            error_text = FONT.render("No comparison data available", True, WHITE)
            win.blit(error_text, (WIDTH // 2 - error_text.get_width() // 2, HEIGHT // 2))
        else:
            if isinstance(plot_file, str):
                plot_image = pygame.image.load(plot_file)
            else:  # BytesIO buffer
                plot_image = pygame.image.load(plot_file)
            plot_image = pygame.transform.scale(plot_image, (WIDTH + UI_WIDTH, HEIGHT - 100))
            win.fill(DARK_GREY)
            win.blit(plot_image, (0, 0))
        
        back_button = {"text": "Back", "rect": pygame.Rect(WIDTH // 2 - 80, HEIGHT - 80, 160, 40), "action": "back"}
        pygame.draw.rect(win, GREY, back_button["rect"])
        text = FONT.render(back_button["text"], True, WHITE)
        win.blit(text, (back_button["rect"].x + 10, back_button["rect"].y + 10))
        
        pygame.display.update()
        return [back_button]
    except Exception as e:
        logging.error(f"Error drawing comparison screen: {str(e)}")
        return []

def algorithm(draw, grid, start, end, algorithm, detected_obstacles, avoid_dir=None):
    global path_positions, global_plan_time, global_nodes_expanded
    try:
        start_time = time.perf_counter()
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
        nodes = 0

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
                global_plan_time = time.perf_counter() - start_time
                global_nodes_expanded = nodes
                logging.info(f"Path found, time: {global_plan_time:.3f}s, nodes: {nodes}")
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
                        temp_g_score += 50
                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos(), algorithm)
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
                        nodes += 1
            draw()
        global_plan_time = time.perf_counter() - start_time
        global_nodes_expanded = nodes
        logging.warning(f"No path found, time: {global_plan_time:.3f}s, nodes: {nodes}")
        return []
    except Exception as e:
        logging.error(f"Error in algorithm: {str(e)}\n{traceback.format_exc()}")
        return []

def animate_path_with_local_replanning(win, grid, path_positions):
    global agent_trail, animation_delay, replan_count, local_detour_path, original_path_positions, total_path_length, simulation_time, success
    try:
        simulation_start = time.perf_counter()
        current_path = path_positions[:]
        using_detour = False
        remaining_original_path = original_path_positions[:]
        last_replan_pos = None
        last_obstacle_pos = None
        max_retries = 3
        while current_path:
            current_time = time.time()
            current_pos = agent_trail[-1] if agent_trail else (1, 1)
            detected_obstacles = get_obstacles_in_range(current_pos, moving_obstacles)
            look_ahead = min(len(current_path), SENSOR_RANGE + 2)
            path_ahead = current_path[:look_ahead]
            obstacles_in_path = find_obstacles_in_path_ahead(current_pos, path_ahead, detected_obstacles, grid)
            predicted_positions = predict_obstacle_positions(grid, moving_obstacles, current_time, 3, rows, current_pos)
            for i, (r, c) in enumerate(path_ahead):
                if (r, c) in predicted_positions and i not in obstacles_in_path:
                    obstacles_in_path.append(i)
            if obstacles_in_path:
                obstacle_pos = path_ahead[obstacles_in_path[0]] if obstacles_in_path else None
                if current_pos == last_replan_pos and obstacle_pos == last_obstacle_pos:
                    logging.info("Skipping repeated replan for same obstacle")
                    time.sleep(animation_delay)
                    yield
                    continue
                last_replan_pos = current_pos
                last_obstacle_pos = obstacle_pos
                logging.info(f"Obstacles detected at path positions: {obstacles_in_path}, doing local replan")
                replan_count += 1
                all_obstacle_positions = set(detected_obstacles)
                all_obstacle_positions.update(predicted_positions)
                rejoin_point, remaining_path = find_rejoin_point(
                    current_pos, remaining_original_path, all_obstacle_positions, grid
                )
                avoid_dir = None
                min_dist = float('inf')
                for obs_pos in all_obstacle_positions:
                    dist = h(current_pos, obs_pos, selected_algorithm)
                    if dist < min_dist:
                        min_dist = dist
                        dr = current_pos[0] - obs_pos[0]
                        dc = current_pos[1] - obs_pos[1]
                        avoid_dir = (0 if dr == 0 else (-1 if dr > 0 else 1),
                                   0 if dc == 0 else (-1 if dc > 0 else 1))
                for row in grid:
                    for spot in row:
                        spot.update_neighbors(grid, detected_obstacles)
                detour_path = local_replan(grid, current_pos, rejoin_point, detected_obstacles, avoid_dir)
                if detour_path:
                    for r, c in local_detour_path:
                        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                            spot = grid[r][c]
                            if not (spot.is_start() or spot.is_end() or spot.visited or (r, c) in dynamic_positions):
                                spot.reset()
                    local_detour_path = detour_path[:]
                    for r, c in local_detour_path:
                        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                            spot = grid[r][c]
                            if not (spot.is_start() or spot.is_end() or spot.visited or (r, c) in dynamic_positions):
                                spot.make_local_path()
                    current_path = detour_path + remaining_path
                    using_detour = True
                    logging.info(f"Detour path: {detour_path}")
                    logging.info(f"Current path after detour: {current_path}")
                    logging.info(f"Detected obstacles: {detected_obstacles}")
                    logging.info(f"Predicted obstacles: {predicted_positions}")
                    logging.info(f"Using local detour with {len(detour_path)} steps to rejoin original path")
                else:
                    logging.warning(f"No local detour found, attempting global replan from {current_pos}")
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid, detected_obstacles)
                    retry_count = 0
                    new_path = []
                    while retry_count < max_retries and not new_path:
                        new_path = algorithm(
                            lambda: (draw_grid_only(win, grid, WIDTH, rows, current_pos), pygame.display.update()),
                            grid, grid[current_pos[0]][current_pos[1]], grid[rows-1][rows-1], selected_algorithm,
                            detected_obstacles if retry_count == 0 else set()
                        )
                        retry_count += 1
                        if not new_path:
                            logging.warning(f"Global replan attempt {retry_count} failed, retrying with relaxed constraints")
                            time.sleep(animation_delay)
                            update_moving_obstacles(grid, moving_obstacles, time.time())
                            for row in grid:
                                for spot in row:
                                    spot.update_neighbors(grid, set())
                    if new_path:
                        for r, c in path_positions:
                            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                                spot = grid[r][c]
                                if not (spot.is_start() or spot.is_end() or spot.visited or (r, c) in dynamic_positions):
                                    spot.reset()
                        current_path = new_path
                        path_positions = new_path
                        original_path_positions = new_path[:]
                        remaining_original_path = new_path[:]
                        using_detour = False
                        logging.info(f"Global replan successful after {retry_count} attempts, new path: {new_path}")
                    else:
                        logging.error(f"No global path found from {current_pos} after {max_retries} retries, stopping animation")
                        simulation_time = time.perf_counter() - simulation_start
                        success = (rows-1, rows-1) in agent_trail
                        save_metrics_to_csv()
                        yield "DONE"
                        return
            else:
                if using_detour:
                    if current_path and current_path[0] in remaining_original_path:
                        try:
                            rejoin_index = remaining_original_path.index(current_path[0])
                            remaining_original_path = remaining_original_path[rejoin_index:]
                            using_detour = False
                            logging.info("Rejoined original path")
                        except ValueError:
                            pass
            if current_path:
                row, col = current_path.pop(0)
                if (row, col) not in agent_trail:
                    agent_trail.append((row, col))
                    grid[row][col].make_visited()
                    total_path_length += 1
                if not using_detour and remaining_original_path:
                    if (row, col) in remaining_original_path:
                        idx = remaining_original_path.index((row, col))
                        remaining_original_path = remaining_original_path[idx + 1:]
            draw_grid_only(win, grid, WIDTH, rows, current_pos)
            time.sleep(animation_delay)
            yield
        end_row, end_col = rows-1, rows-1
        if (end_row, end_col) not in agent_trail:
            agent_trail.append((end_row, end_col))
            grid[end_row][end_col].make_visited()
            total_path_length += 1
            draw_grid_only(win, grid, WIDTH, rows, (end_row, end_col))
            time.sleep(animation_delay)
        simulation_time = time.perf_counter() - simulation_start
        success = (end_row, end_col) in agent_trail
        save_metrics_to_csv()
        logging.info("Animation completed with local replanning")
        yield "SUCCESS" if success else "DONE"
    except Exception as e:
        logging.error(f"Error in animation: {str(e)}\n{traceback.format_exc()}")
        simulation_time = time.perf_counter() - simulation_start
        success = (rows-1, rows-1) in agent_trail
        save_metrics_to_csv()
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
    global agent_trail, moving_obstacles, dynamic_positions, replan_count, path_positions, path_length, original_path_positions, local_detour_path, total_path_length, global_plan_time, local_plan_time, global_nodes_expanded, local_nodes_expanded, simulation_time, success
    try:
        logging.debug(f"Resetting grid with use_maze={use_maze}")
        agent_trail = []
        dynamic_positions = set()
        replan_count = 0
        path_positions = []
        original_path_positions = []
        local_detour_path = []
        path_length = 0
        total_path_length = 0
        global_plan_time = 0.0
        local_plan_time = 0.0
        global_nodes_expanded = 0
        local_nodes_expanded = 0
        simulation_time = 0.0
        success = False
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

def pygame_loop(win, width):
    global rows, animation_delay, speed_scale, selected_algorithm, dragging_slider, moving_obstacles, replan_count, path_positions, num_obstacles, SENSOR_RANGE, show_sensor_range, use_maze, path_length, total_path_length, global_plan_time, local_plan_time, global_nodes_expanded, local_nodes_expanded, simulation_time, success
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
        show_results = False
        show_comparison = False
        gap = width // rows
        while running:
            clock.tick(15 if agent_animator else 10)
            current_time = time.time()
            if not show_results and not show_comparison:
                update_moving_obstacles(grid, moving_obstacles, current_time)
                for row in grid:
                    for spot in row:
                        spot.update_neighbors(grid, set() if not initial_path_computed else get_obstacles_in_range(agent_trail[-1] if agent_trail else (1, 1), moving_obstacles))
                buttons, sliders = draw_ui(win)
                agent_pos = agent_trail[-1] if agent_trail else (1, 1)
                draw_grid_only(win, grid, width, rows, agent_pos)
            elif show_results:
                buttons = draw_results_screen(win)
                sliders = []
            else:
                buttons = draw_comparison_screen(win)
                sliders = []
            if agent_animator:
                try:
                    result = next(agent_animator)
                    if result == "SUCCESS" or result == "DONE":
                        logging.info("Animation terminated")
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
                    if show_results or show_comparison:
                        for btn in buttons:
                            if btn["rect"].collidepoint(pos) and btn["action"] == "back":
                                show_results = False
                                show_comparison = False
                                grid = reset_grid(grid, rows)
                                start = grid[1][1]
                                end = grid[rows-1][rows-1]
                                agent_animator = None
                                path_positions = []
                                initial_path_computed = False
                                logging.info("Returned to simulation from results or comparison")
                        continue
                    for btn in buttons:
                        if btn["rect"].collidepoint(pos) and btn.get("enabled", True):
                            logging.info(f"Button clicked: {btn['action']}")
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
                                    agent_animator = animate_path_with_local_replanning(win, grid, path_positions)
                                    logging.info("Animation started with local replanning")
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
                            elif btn["action"] == "clear_metrics":
                                metrics_store.clear()
                                if os.path.isfile("simulation_metrics.csv"):
                                    os.remove("simulation_metrics.csv")
                                logging.info("Metrics cleared")
                            elif btn["action"] == "show_metrics":
                                if success:
                                    show_results = True
                                    logging.info("Showing metrics results")
                                else:
                                    logging.warning("Simulation not completed successfully yet. Metrics not available.")
                            elif btn["action"] == "show_comparison":
                                show_comparison = True
                                logging.info("Show Comparison button activated")
                    for slider in sliders:
                        if slider["rect"].collidepoint(pos):
                            dragging_slider = slider["id"]
                            logging.info(f"Started dragging slider: {dragging_slider}")
                    if pos[0] < WIDTH and not show_results and not show_comparison:
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

import asyncio
import platform
FPS = 60

async def main():
    pygame_loop(WIN, WIDTH)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())