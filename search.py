# search.py - Updated with proper cost differentiation

from grid import DANGER, BLOCKED, EMPTY
import heapq
import math
import random

def path_cost(path, grid):
    """Calculate path cost with DANGER cells costing 5x more"""
    if not path:
        return float('inf')
    cost = 0
    for r, c in path:
        if grid[r][c] == DANGER:
            cost += 5  # ← Increased from 1 to 5 (was too small)
        else:
            cost += 1
    return cost

def path_risk(path, grid):
    """Count danger cells on path"""
    if not path:
        return float('inf')
    return sum(1 for r, c in path if grid[r][c] == DANGER)

def path_length(path):
    return len(path) if path else float('inf')

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ============================================================
# BFS - Unweighted, shortest steps only
# ============================================================
def bfs(grid, start, goal, blocked_set):
    if start == goal:
        return [start]
    
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        (r, c), path = queue.popleft()
        
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) == goal:
                    return path + [(nr, nc)]
                if (nr, nc) not in visited and (nr, nc) not in blocked_set:
                    if grid[nr][nc] != BLOCKED:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
    return None

# ============================================================
# A* - Weighted: Avoids danger cells (higher cost)
# ============================================================
def astar(grid, start, goal, blocked_set):
    if start == goal:
        return [start]
    
    rows, cols = len(grid), len(grid[0])
    
    def get_cost(cell):
        r, c = cell
        return 5 if grid[r][c] == DANGER else 1
    
    open_set = [(0 + manhattan(start, goal), 0, start, [start])]
    g_score = {start: 0}
    
    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = (nr, nc)
                if neighbor in blocked_set or grid[nr][nc] == BLOCKED:
                    continue
                
                new_cost = cost + get_cost(neighbor)
                
                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    g_score[neighbor] = new_cost
                    f_score = new_cost + manhattan(neighbor, goal)
                    heapq.heappush(open_set, (f_score, new_cost, neighbor, path + [neighbor]))
    
    return None

# ============================================================
# Greedy BFS - Only heuristic, ignores path cost (may take danger)
# ============================================================
def greedy_bfs(grid, start, goal, blocked_set):
    """Greedy: Only uses heuristic, may choose dangerous but heuristically good paths"""
    if start == goal:
        return [start]
    
    rows, cols = len(grid), len(grid[0])
    
    # Greedy picks cell with smallest heuristic, regardless of danger
    heap = [(manhattan(start, goal), start, [start])]
    visited = {start}
    
    while heap:
        _, current, path = heapq.heappop(heap)
        
        if current == goal:
            return path
        
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = (nr, nc)
                if neighbor not in visited and neighbor not in blocked_set:
                    if grid[nr][nc] != BLOCKED:
                        visited.add(neighbor)
                        heapq.heappush(heap, (manhattan(neighbor, goal), neighbor, path + [neighbor]))
    
    return None

# ============================================================
# Hill Climbing - Can get stuck in local optima
# ============================================================
def hill_climbing(grid, start, goal, blocked_set):
    """Hill climbing: Always moves to best neighbor, can get stuck"""
    if start == goal:
        return [start]
    
    rows, cols = len(grid), len(grid[0])
    current = start
    path = [current]
    visited = {current}
    
    def evaluate(cell):
        # Lower is better
        r, c = cell
        danger_penalty = 10 if grid[r][c] == DANGER else 0
        return manhattan(cell, goal) + danger_penalty
    
    stuck_count = 0
    while current != goal and stuck_count < 50:
        neighbors = []
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = (nr, nc)
                if neighbor not in blocked_set and grid[nr][nc] != BLOCKED:
                    if neighbor not in visited:
                        neighbors.append((evaluate(neighbor), neighbor))
        
        if not neighbors:
            # Stuck! Return what we have (may not reach goal)
            return path if path[-1] == goal else None
        
        neighbors.sort(key=lambda x: x[0])
        best_score, best_neighbor = neighbors[0]
        
        # If no improvement, stuck
        if best_score >= evaluate(current) and current != goal:
            stuck_count += 1
        
        current = best_neighbor
        path.append(current)
        visited.add(current)
    
    return path if path[-1] == goal else None