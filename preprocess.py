from collections import deque
import numpy as np


def get_areas(grid, district):
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j]==district:
            grid[i][j] = -1
            return 1 + dfs(i - 1, j) + dfs(i, j + 1) + \
                dfs(i + 1, j) + dfs(i, j - 1)
        return 0

    areas = [dfs(i, j) for i in range(m) for j in range(n) if grid[i][j]==district]
    return areas

def upper_lower_district_size(grid, district, min_max):
    areas = get_areas(grid.copy(), district)
    return min_max(areas) if areas else 0

def max_district_sizes(grid):
    return [upper_lower_district_size(grid, i, max) for i in range(0, 5)]

def min_district_sizes(grid):
    return [upper_lower_district_size(grid, i, min) for i in range(0, 5)]

def avg_district_sizes(grid):
    return [np.average(get_areas(grid, i)) for i in range(5)]

def num_districts(grid):
    return [len(get_areas(grid, i)) for i in range(5)]




def isValid(vis, row, col, m, n):
    if (row < 0 or col < 0 or row >= m or col >= n):
        return False
    if (vis[row][col] == -1):
        return False
 
    # Otherwise
    return True
 
# Function to perform the BFS traversal
def BFS(grid, vis, row, col, goal):
    dRow = [ -1, 0, 1, 0]
    dCol = [ 0, 1, 0, -1]
   
    q = deque()
    m, n  = len(grid), len(grid[0])
    q.append(( row, col ))
    vis[row][col] = -1
 
    # Iterate while the queue
    # is not empty
    count = 0
    while (len(q) > 0):
        for i in range(len(q)):
            cell = q.popleft()
            x = cell[0]
            y = cell[1]

            if grid[x][y] == goal:
                return count
    
            # Go to the adjacent cells
            for i in range(4):
                adjx = x + dRow[i]
                adjy = y + dCol[i]
                if (isValid(vis, adjx, adjy, m, n)):
                    q.append((adjx, adjy))
                    vis[adjx][adjy] = True
        count += 1
    return count


def get_distances(grid, district_1, district_2):
    distances = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != district_1:
                continue
            distances.append(BFS(grid.copy(), grid.copy(), i, j, district_2))
    return distances



def get_avg_min_distance(grid, district_1, district_2):
    dists = get_distances(grid, district_1, district_2)
    return 0 if not dists else np.average(dists)


def find_avg_min_distances(grid):
    distances = []
    for i in range(5):
        for j in range(i+1, 5):
            dist = get_avg_min_distance(grid.copy(), i, j)
            distances.append(dist)
    return distances


def preprocess_data(grid, preprocess_fxns):
    vector = grid.reshape(grid.shape[0], 49)
    for fxn in preprocess_fxns:
        vector = np.concatenate(vector, fxn(grid))
    return vector
