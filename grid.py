# grid.py - 6x6 version with 5 victims

EMPTY    = 0
START    = 1
VICTIM   = 2
HOSPITAL = 3
BLOCKED  = 4
DANGER   = 5

ROWS = 6
COLS = 6

SYMBOLS = {
    EMPTY:    ".",
    START:    "S",
    VICTIM:   "V",
    HOSPITAL: "H",
    BLOCKED:  "X",
    DANGER:   "!"
}

def create_grid():
    grid = [[EMPTY] * COLS for _ in range(ROWS)]
    
    # Start position
    grid[0][0] = START
    
    # Hospitals (2)
    grid[0][5] = HOSPITAL
    grid[5][5] = HOSPITAL
    
    # Victims (5 victims) - ALL 5 should be in 6x6 grid
    grid[1][2] = VICTIM   # Victim 1
    grid[2][4] = VICTIM   # Victim 2
    grid[3][1] = VICTIM   # Victim 3
    grid[4][3] = VICTIM   # Victim 4
    grid[5][2] = VICTIM   # Victim 5
    
    # Danger cells
    grid[1][1] = DANGER
    grid[1][3] = DANGER
    grid[2][2] = DANGER
    grid[3][3] = DANGER
    grid[4][2] = DANGER
    grid[4][4] = DANGER
    
    # Blocked roads (4 blockages)
    grid[0][3] = BLOCKED
    grid[1][4] = BLOCKED
    grid[3][4] = BLOCKED
    grid[5][4] = BLOCKED
    
    return grid

def print_grid(grid, title="DISASTER GRID (6x6)"):
    print(f"\n{'='*34}")
    print(f"  {title}")
    print(f"{'='*34}")
    print("     " + " ".join(str(c) for c in range(COLS)))
    print("     " + "─"*(COLS*2-1))
    for r, row in enumerate(grid):
        print(f"  {r} │ " + " ".join(SYMBOLS[c] for c in row))
    print()

def extract_positions(grid):
    start, victims, hospitals, blocked, danger = None, [], [], [], []
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r][c]
            if   v == START:    start = (r, c)
            elif v == VICTIM:   victims.append((r, c))
            elif v == HOSPITAL: hospitals.append((r, c))
            elif v == BLOCKED:  blocked.append((r, c))
            elif v == DANGER:   danger.append((r, c))
    return start, victims, hospitals, blocked, danger

def block_road(grid, blocked_set, pos):
    r, c = pos
    if grid[r][c] == EMPTY:
        grid[r][c] = BLOCKED
        blocked_set.add(pos)
        return True
    return False