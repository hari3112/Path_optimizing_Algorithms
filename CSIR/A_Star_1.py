import pygame
import math
from queue import PriorityQueue
import numpy
import time

WIDTH = 680
HEIGHT = 680
WIN = pygame.display.set_mode((WIDTH * 2, HEIGHT))  # Double width for two grids
pygame.display.set_caption("A* Path Finding Algorithm - Dual Grid View")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
TURQUOISE = (64, 224, 208)
GREY = (128, 128, 128)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE
    

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win, offset_x=0):
        pygame.draw.rect(win, self.color, (self.x + offset_x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])
        
        

    def __lt__(self, other):
        return False

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    path = []
    while current in came_from:
        current = came_from[current]
        path.append(current.get_pos())
        current.make_path()
        draw()
    path.reverse()
    b=numpy.array(path)
    return b

def animate_path_on_grid(win, path_positions, grid, delay=150, offset_x=WIDTH):
    for row, col in path_positions:
        pygame.time.delay(delay)
        spot = grid[row][col]
        pygame.draw.circle(win, BLUE, (spot.x + spot.width // 2 + offset_x, spot.y + spot.width // 2), spot.width // 3)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pygame.time.delay(10000)
    print('Goal Reached'.center(100,'-'))
    
    

def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            a=reconstruct_path(came_from, end, draw)
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return a

def silent_algorithm(grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            start.make_start()
            a=reconstruct_path(came_from, end, lambda: None)
            end.make_end()

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
        

    return a

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid

def draw_grid(win, rows, width, offset_x=0):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (offset_x, i * gap), (offset_x + width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (offset_x + j * gap, 0), (offset_x + j * gap, width))

def draw_both(win, grid1, grid2, rows, width):
    win.fill(WHITE)
    for row in grid1:
        for spot in row:
            spot.draw(win)
    for row in grid2:
        for spot in row:
            spot.draw(win, offset_x=WIDTH)
    
    # Grid lines
    draw_grid(win, rows, width)
    draw_grid(win, rows, width, offset_x=WIDTH)

    # Divider line
    pygame.draw.line(win, BLACK, (WIDTH, 0), (WIDTH, HEIGHT), 10)

    pygame.display.update()
        

def main(win, width):
    ROWS = 30
    grid1 = make_grid(ROWS, width)
    grid2 = make_grid(ROWS, width)
     # right at the top of main()


    # Define start and end positions
    start_pos = (1, 1)
    end_pos = (29, 29)

    start1 = grid1[start_pos[0]][start_pos[1]]
    end1 = grid1[end_pos[0]][end_pos[1]]
    start2 = grid2[start_pos[0]][start_pos[1]]
    end2 = grid2[end_pos[0]][end_pos[1]]

    start1.make_start()
    end1.make_end()
    start2.make_start()
    end2.make_end()

    # Define obstacles (e.g., a vertical wall at col = 15 from row 0 to 29)

    for i1 in range(9,14):
        for j1 in range(0,3):
            grid1[i1][j1].make_barrier()
            grid2[i1][j1].make_barrier()

    for i2 in range(21,24):
        for j2 in range(2,8):
            grid1[i2][j2].make_barrier()
            grid2[i2][j2].make_barrier()

    for j3 in range(6,24,4):
        grid1[6][j3].make_barrier()
        grid2[6][j3].make_barrier()

    for i3 in range(6,17,5):
        grid1[i3][6].make_barrier()
        grid2[i3][6].make_barrier()

    for goods_2 in range(12,29,10):
        for goods_1 in range(11,30,8):
            for i4 in range(goods_1,goods_1+3):
                for j4 in range(goods_2,goods_2+6):
                    grid1[i4][j4].make_barrier()
                    grid2[i4][j4].make_barrier()

    

    for row in grid1:
        for spot in row:
            spot.update_neighbors(grid1)
    for row in grid2:
        for spot in row:
            spot.update_neighbors(grid2)

    

    run = True
    started = False
    draw_both(win, grid1, grid2, ROWS, width)
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start1 and end1:
                    for row in grid1:
                        for spot in row:
                            spot.update_neighbors(grid1)
                    algorithm(lambda: draw_both(win, grid1, grid2, ROWS, width), grid1, start1, end1)
                    path_positions = silent_algorithm( grid2, start2, end2)
                    print(path_positions)

                if event.key == pygame.K_g:
                    animate_path_on_grid(win, path_positions, grid2)
                    
                    

                if event.key == pygame.K_c:
                    grid1 = make_grid(ROWS, width)
                    grid2 = make_grid(ROWS, width)

                    start1 = grid1[start_pos[0]][start_pos[1]]
                    end1 = grid1[end_pos[0]][end_pos[1]]
                    start2 = grid2[start_pos[0]][start_pos[1]]
                    end2 = grid2[end_pos[0]][end_pos[1]]

                    start1.make_start()
                    end1.make_end()
                    start2.make_start()
                    end2.make_end()

                    for i1 in range(9,14):
                        for j1 in range(0,3):
                            grid1[i1][j1].make_barrier()
                            grid2[i1][j1].make_barrier()
                    for i2 in range(21,24):
                        for j2 in range(2,8):
                            grid1[i2][j2].make_barrier()
                            grid2[i2][j2].make_barrier()
                    for j3 in range(6,24,4):
                        grid1[6][j3].make_barrier()
                        grid2[6][j3].make_barrier()
                    for i3 in range(6,17,5):
                        grid1[i3][6].make_barrier()
                        grid2[i3][6].make_barrier()
                    for goods_2 in range(12,29,10):
                        for goods_1 in range(11,30,8):
                            for i4 in range(goods_1,goods_1+3):
                                for j4 in range(goods_2,goods_2+6):
                                    grid1[i4][j4].make_barrier()
                                    grid2[i4][j4].make_barrier()

                    for row in grid1:
                        for spot in row:
                            spot.update_neighbors(grid1)
                    for row in grid2:
                        for spot in row:
                            spot.update_neighbors(grid2)

                    started = False
                    path_positions=[]
                    
        

    pygame.quit()

main(WIN, WIDTH)
