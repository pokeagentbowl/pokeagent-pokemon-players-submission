"""
A* pathfinding system with Pokemon-specific terrain handling.

This module provides:
- Game-independent A* pathfinding on cost grids
- TerrainGrid helper for managing terrain costs
- Main pathfinder that attempts A* first, then falls back to simple path
"""
import heapq
import logging

from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

TILE_SYMBOLS = {
    '.', '#', '~', '^', 'W', 'I', 's', 'D', 'S', 'C',
    '→', '←', '↑', '↓', '↗', '↖', '↘', '↙', 'L', 'T',
    'G', 'K', 'B', '?', 'P', 'N'
}

# treat P player as walkable to prevent validation errors when starting at player position
WALKABLE_SYMBOLS = {
    '.', 'D', 'S', 'P', '~'
}

# TODO: handle water which is conditional on surf


class TerrainGrid:
    """Manages 15x15 traversability grid for pathfinding."""
    
    def __init__(self, traversability_map: List[List[str]]):
        """
        Initialize terrain grid.
        
        Args:
            traversability_map: 15x15 grid of traversability symbols
        """
        self.grid = traversability_map
    
    def is_walkable(self, local_tile_x: int, local_tile_y: int) -> bool:
        """
        Check if a local tile position is traversable.
        
        Args:
            local_tile_x: X position in local tile grid (0-14)
            local_tile_y: Y position in local tile grid (0-14)
            
        Returns:
            True if walkable, False if blocked
        """
        pass
        # if not self.is_within_bounds(local_tile_x, local_tile_y):
        #     return False
        
        # cell = self.grid[local_tile_y][local_tile_x]
        # # '#' represents blocked/wall tiles
        # return cell != '#'
    
    def get_terrain_cost(self, local_tile_x: int, local_tile_y: int) -> float:
        """
        Get movement cost for a local tile position.
        TODO: currently naive implementation that looks for directly traversable only
        
        Args:
            local_tile_x: X position in local tile grid (0-14)
            local_tile_y: Y position in local tile grid (0-14)
            
        Returns:
            Movement cost (1 for normal, 100 for water, etc)
        """
        # if not self.is_within_bounds(local_tile_x, local_tile_y):
        #     return float('inf')
        
        cell = self.grid[local_tile_y][local_tile_x]
        
        if cell in WALKABLE_SYMBOLS:
            return 1
        else:
            return float('inf')
    
    def is_within_bounds(self, local_tile_x: int, local_tile_y: int) -> bool:
        """
        Check if position is within local tile grid bounds.
        
        Args:
            local_tile_x: X position in local tile grid
            local_tile_y: Y position in local tile grid
            
        Returns:
            True if within 15x15 bounds
        """
        pass
        # if not self.grid or not self.grid[0]:
        #     return False
        # height = len(self.grid)
        # width = len(self.grid[0])
        # return 0 <= local_tile_x < width and 0 <= local_tile_y < height


class AStarPathfinder:
    """A* pathfinding implementation with direction-aware state space."""
    
    def __init__(self):
        """Initialize pathfinder."""
        self.terrain: Optional[TerrainGrid] = None
    
    def _game_to_cost_grid(
        self,
        terrain: TerrainGrid,
        obstacles: Optional[List[Tuple[int, int]]] = None
    ) -> List[List[float]]:
        """
        NODE-BASED: Transform game information to cost grid.
        DEPRECATED: Will be replaced by _game_to_cost_grid_edges for proper jump tile handling.

        This is a placeholder for future implementation that will convert
        game-specific terrain and obstacle information into a simple 2D cost grid
        suitable for the game-independent A* algorithm.

        Args:
            terrain: TerrainGrid with game-specific terrain information
            obstacles: Optional list of obstacle positions

        Returns:
            2D cost grid where grid[y][x] is the cost to enter cell (x, y)
            Use float('inf') for blocked cells
        """
        # TODO: Implement proper transformation from game terrain to cost grid
        # For now, return a simple placeholder that uses terrain costs
        if not terrain.grid or not terrain.grid[0]:
            logger.error("Cannot build cost grid: terrain grid is empty or invalid")
            return [[]]
        
        height = len(terrain.grid)
        width = len(terrain.grid[0])
        cost_grid = []
        
        obstacle_set = set(obstacles) if obstacles else set()
        
        for y in range(height):
            row = []
            for x in range(width):
                if (x, y) in obstacle_set:
                    row.append(float('inf'))
                else:
                    row.append(terrain.get_terrain_cost(x, y))
            cost_grid.append(row)
        
        return cost_grid

    def _game_to_cost_grid_edges(
        self,
        terrain: TerrainGrid,
        obstacles: Optional[List[Tuple[int, int]]] = None
    ) -> dict:
        """
        EDGE-BASED: Transform game grid to directed edge representation.

        Builds a graph where:
        - Nodes are positions (x, y)
        - Edges represent valid movements between positions
        - Jump tiles create directed edges that skip over the jump tile

        Jump tile behavior:
        - '↓' at (x, y): Edge from (x, y-1) to (x, y+1) [can only approach from above, jumps down]
        - '↑' at (x, y): Edge from (x, y+1) to (x, y-1) [can only approach from below, jumps up]
        - '←' at (x, y): Edge from (x+1, y) to (x-1, y) [can only approach from right, jumps left]
        - '→' at (x, y): Edge from (x-1, y) to (x+1, y) [can only approach from left, jumps right]

        Args:
            terrain: TerrainGrid with game-specific terrain information
            obstacles: Optional list of obstacle positions

        Returns:
            Dict mapping (x, y) positions to list of (neighbor_pos, cost) tuples
        """
        if not terrain.grid or not terrain.grid[0]:
            logger.error("Cannot build edge graph: terrain grid is empty or invalid")
            return {}

        height = len(terrain.grid)
        width = len(terrain.grid[0])
        obstacle_set = set(obstacles) if obstacles else set()

        # Build edge graph
        edges = {}

        # Direction mappings for jump tiles
        # from_dir: the movement direction needed to approach this jump tile
        # to_dir: the direction the jump takes you (landing offset from jump tile)
        JUMP_TILES = {
            '↓': {'from_dir': (0, 1), 'to_dir': (0, 1)},   # Down: approach by moving DOWN, land below
            '↑': {'from_dir': (0, -1), 'to_dir': (0, -1)},   # Up: approach by moving UP, land above
            '←': {'from_dir': (-1, 0), 'to_dir': (-1, 0)},   # Left: approach by moving LEFT, land left
            '→': {'from_dir': (1, 0), 'to_dir': (1, 0)},   # Right: approach by moving RIGHT, land right
        }

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height

        def is_passable(x: int, y: int) -> bool:
            """Check if a position is passable (not blocked, not obstacle)."""
            if not in_bounds(x, y):
                return False
            if (x, y) in obstacle_set:
                return False
            cell = terrain.grid[y][x]
            # Passable if walkable or a jump tile
            return cell in WALKABLE_SYMBOLS or cell in JUMP_TILES

        # Build edges for each position
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                cell = terrain.grid[y][x]

                # Skip blocked cells and obstacles
                # TODO: this should be is_passable(x, y) right?
                if (x, y) in obstacle_set or cell == '#':
                    continue

                edges[pos] = []

                # Check if this cell is a jump tile
                if cell in JUMP_TILES:
                    # Jump tiles don't have regular outgoing edges
                    # The edges TO the landing position are added from the approach position
                    continue

                # For walkable cells, add edges to adjacent cells
                if cell in WALKABLE_SYMBOLS:
                    # Standard 4-directional movement
                    moves = [
                        (0, -1, "UP"),
                        (0, 1, "DOWN"),
                        (-1, 0, "LEFT"),
                        (1, 0, "RIGHT")
                    ]

                    for dx, dy, direction in moves:
                        nx, ny = x + dx, y + dy

                        if not in_bounds(nx, ny):
                            continue

                        neighbor_cell = terrain.grid[ny][nx]

                        # Check if neighbor is a jump tile
                        if neighbor_cell in JUMP_TILES:
                            # Need to check if we can approach this jump tile from our direction
                            jump_info = JUMP_TILES[neighbor_cell]
                            expected_from = jump_info['from_dir']

                            # Check if our movement direction matches the required approach direction
                            if (dx, dy) == expected_from:
                                # Valid jump! Edge goes to the landing position
                                to_dir = jump_info['to_dir']
                                land_x = nx + to_dir[0]
                                land_y = ny + to_dir[1]

                                # Verify landing position is valid
                                if in_bounds(land_x, land_y):
                                    land_cell = terrain.grid[land_y][land_x]
                                    # Landing position must be walkable or passable
                                    if land_cell in WALKABLE_SYMBOLS or land_cell in JUMP_TILES:
                                        # Cost is 2: one to reach jump tile, one to land
                                        edges[pos].append(((land_x, land_y), 2))
                            # else: cannot approach this jump tile from our direction, no edge

                        elif neighbor_cell in WALKABLE_SYMBOLS:
                            # Regular walkable neighbor
                            if (nx, ny) not in obstacle_set:
                                cost = terrain.get_terrain_cost(nx, ny)
                                edges[pos].append(((nx, ny), cost))

        return edges

    def _run_astar_edges(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        edges: dict
    ) -> Optional[List[str]]:
        """
        EDGE-BASED: A* pathfinding on directed edge graph.

        This implementation works on a directed edge graph to properly handle
        directional ledges and jump tiles.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            edges: Dict mapping positions to list of (neighbor_pos, cost) tuples

        Returns:
            List of actions ["UP", "DOWN", "LEFT", "RIGHT"] or None if no path exists
        """
        if not edges:
            logger.error("A* failed: edge graph is empty")
            return None

        # Don't validate start since it's the player position (always valid)
        # The start may not be in edges if it's in obstacles list, but we still process it
        # if start not in edges:
        #     logger.error(f"A* failed: start position {start} not in edge graph")
        #     return None

        # Direction mappings
        def get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Optional[str]:
            """Determine direction from one position to another."""
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]

            # Normalize for multi-step jumps
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)

            direction_map = {
                (0, -1): "UP",
                (0, 1): "DOWN",
                (-1, 0): "LEFT",
                (1, 0): "RIGHT"
            }
            return direction_map.get((dx, dy))

        # A* search
        open_heap = []
        heapq.heappush(open_heap, (0 + self._heuristic(start, goal), 0, start))

        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            # Goal check
            if current == goal:
                # Reconstruct path
                path = []
                pos = current
                while pos in came_from:
                    prev_pos = came_from[pos]
                    direction = get_direction(prev_pos, pos)
                    if direction:
                        path.append(direction)
                    pos = prev_pos

                path.reverse()
                logger.debug(f"A* found path: {len(path)} steps, cost: {g}")
                return path

            # Check if goal is adjacent (for unwalkable goals)
            if goal not in edges:
                if abs(current[0] - goal[0]) + abs(current[1] - goal[1]) == 1:
                    # Adjacent to goal
                    path = []
                    pos = current
                    while pos in came_from:
                        prev_pos = came_from[pos]
                        direction = get_direction(prev_pos, pos)
                        if direction:
                            path.append(direction)
                        pos = prev_pos

                    path.reverse()
                    logger.debug(f"A* reached adjacent to unwalkable goal: {len(path)} steps, cost: {g}")
                    return path

            # Explore neighbors via edges
            if current not in edges:
                continue

            for neighbor, edge_cost in edges[current]:
                if neighbor in closed:
                    continue

                # Calculate tentative g score
                tentative_g = g + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        # No path found
        logger.warning(f"A* no path found: explored {len(closed)} nodes from {start} to {goal}")
        logger.debug(f"Closed set size: {len(closed)}, g_score size: {len(g_score)}")
        return None

    def find_path(
        self,
        start_local_tile: Tuple[int, int],
        goal_local_tile: Tuple[int, int],
        start_facing: str,
        terrain: TerrainGrid,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        movement_mode: str = "facing_aware"
    ) -> List[str]:
        """
        Find path from start to goal using A* algorithm with fallback.

        Main pathfinder method that:
        1. Transforms game information (terrain, obstacles) to edge graph
        2. Attempts A* pathfinding on the edge graph (supports jump tiles)
        3. Falls back to simple path if A* fails

        Args:
            start_local_tile: Starting local tile position (x, y)
            goal_local_tile: Goal local tile position (x, y)
            start_facing: Initial facing direction ("North", "South", "East", "West")
            terrain: TerrainGrid instance
            obstacles: Optional list of obstacle positions to avoid
            movement_mode: Movement mode ("naive" or "facing_aware")

        Returns:
            List of action strings: ["UP", "DOWN", "LEFT", "RIGHT"]
            Includes both turns and moves
        """
        # Use edge-based pathfinding for proper jump tile support
        edges = self._game_to_cost_grid_edges(terrain, obstacles)

        # Run edge-based A* pathfinding
        astar_result = self._run_astar_edges(start_local_tile, goal_local_tile, edges)

        # If A* fails, fallback to simple path
        if astar_result is None:
            logger.warning(f"A* pathfinding failed from {start_local_tile} to {goal_local_tile}, using simple fallback")
            astar_result = simple_path_fallback(start_local_tile, goal_local_tile)
            logger.info(f"Fallback path: {astar_result}")
        else:
            logger.info(f"A* path found: {astar_result} ({len(astar_result)} steps)")

        # If facing_aware mode, post-process the path to add turn actions
        if movement_mode == "facing_aware":
            # NOTE: we dont have player facing from game, need manual probe!
            astar_result = add_turn_actions(astar_result, start_facing)

        return astar_result
    
    def _run_astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_grid: List[List[float]]
    ) -> Optional[List[str]]:
        """
        NODE-BASED: Game-independent A* pathfinding on a cost grid.
        DEPRECATED: Will be replaced by _run_astar_edges for proper jump tile handling.

        This is a pure A* implementation that only knows about:
        - Start position (x, y)
        - Goal position (x, y)
        - Cost grid (2D array of movement costs)

        No game-specific logic or terrain types. Costs determine traversability:
        - Cost < infinity: walkable with that cost
        - Cost = infinity: blocked/unwalkable

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            cost_grid: 2D array where cost_grid[y][x] is the cost to enter cell (x, y)
                      Use float('inf') for blocked cells

        Returns:
            List of actions ["UP", "DOWN", "LEFT", "RIGHT"] or None if no path exists
        """
        if not cost_grid or not cost_grid[0]:
            logger.error("A* failed: cost grid is empty or invalid")
            return None
        
        height = len(cost_grid)
        width = len(cost_grid[0])
        
        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height
        
        def is_walkable(x: int, y: int) -> bool:
            return in_bounds(x, y) and cost_grid[y][x] < float('inf')
        
        # Validate start and goal
        # dont validate start since its the player but we will do so if we extend beyond that
        # if not is_walkable(start[0], start[1]):
        #     logger.error(f"A* failed: start position {start} is not walkable (cost: {cost_grid[start[1]][start[0]] if in_bounds(start[0], start[1]) else 'out of bounds'})")
        #     return None
        # if not is_walkable(goal[0], goal[1]):
        #     logger.error(f"A* failed: goal position {goal} is not walkable (cost: {cost_grid[goal[1]][goal[0]] if in_bounds(goal[0], goal[1]) else 'out of bounds'})")
        #     return None
        
        # Direction mappings
        MOVES = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
        
        # A* search
        open_heap = []
        start_state = start
        heapq.heappush(open_heap, (0 + self._heuristic(start, goal), 0, start_state))
        
        came_from = {}
        came_action = {}
        g_score = {start_state: 0}
        closed = set()
        
        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            
            if current in closed:
                continue
            closed.add(current)
            
            # Goal check
            if current == goal:
                path = self._reconstruct_path(came_from, came_action, start_state, current)
                logger.debug(f"A* found path: {len(path)} steps, cost: {g}")
                return path

            if not is_walkable(goal[0], goal[1]):
                if abs(current[0] - goal[0]) + abs(current[1] - goal[1]) == 1:
                    # Adjacent to goal
                    path = self._reconstruct_path(came_from, came_action, start_state, current)
                    logger.debug(f"A* reached adjacent to unwalkable goal: {len(path)} steps, cost: {g}")
                    return path
            
            # Explore neighbors
            x, y = current
            for direction, (dx, dy) in MOVES.items():
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if not is_walkable(nx, ny):
                    continue
                
                if neighbor in closed:
                    continue
                
                # Calculate tentative g score
                tentative_g = g + cost_grid[ny][nx]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    came_action[neighbor] = direction
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))
        
        # No path found
        logger.warning(f"A* no path found: explored {len(closed)} nodes from {start} to {goal}")
        logger.debug(f"Closed set size: {len(closed)}, g_score size: {len(g_score)}")
        return None
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic.
        
        Args:
            pos: Current position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Manhattan distance
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _reconstruct_path(
        self,
        came_from: dict,
        came_action: dict,
        start,
        end
    ) -> List[str]:
        """
        Reconstruct action sequence from A* search.
        
        Works with both simple position tuples (x, y) from game-independent A*
        and direction-aware states (x, y, direction) from direction-aware A*.
        
        Args:
            came_from: Dict mapping states to previous states
            came_action: Dict mapping states to actions taken
            start: Starting state (position or state)
            end: Ending state (position or state)
            
        Returns:
            List of actions from start to end
        """
        actions = []
        current = end
        
        while current != start and current in came_from:
            actions.append(came_action[current])
            current = came_from[current]
        
        actions.reverse()
        return actions


def add_turn_actions(
    path: List[str],
    start_facing: str
) -> List[str]:
    """
    Post-process a naive path to add turn actions for facing-aware movement.
    
    In facing-aware mode:
    - Pressing a button in current facing direction: MOVES
    - Pressing a button in different direction: TURNS (no movement)
    
    This simulates the player's facing direction and inserts turn actions when needed.
    
    Args:
        path: Naive path of actions ["UP", "DOWN", "LEFT", "RIGHT"]
        start_facing: Initial facing direction ("North", "South", "East", "West")
        
    Returns:
        Path with turn actions inserted
    """
    if not path:
        return path
    
    # Map actions to facing directions
    ACTION_TO_DIRECTION = {
        "UP": "North",
        "DOWN": "South",
        "LEFT": "West",
        "RIGHT": "East"
    }
    
    result = []
    current_facing = start_facing
    
    for action in path:
        target_direction = ACTION_TO_DIRECTION.get(action)
        
        if target_direction is None:
            # Unknown action, keep as-is
            result.append(action)
            continue
        
        if target_direction != current_facing:
            # Need to turn first: insert action as turn
            result.append(action)
            current_facing = target_direction
        
        # Now move: insert action as movement
        result.append(action)
    
    return result


def simple_path_fallback(
    start_local_tile: Tuple[int, int],
    goal_local_tile: Tuple[int, int]
) -> List[str]:
    """
    Simple fallback pathfinding - moves in straight lines (X then Y).
    
    Temporary MVP implementation that doesn't use A* or traversability checks.
    Just generates actions to move horizontally first, then vertically.
    
    Args:
        start_local_tile: Starting local tile position (x, y)
        goal_local_tile: Goal local tile position (x, y)
        
    Returns:
        List of action strings: ["UP", "DOWN", "LEFT", "RIGHT"]
    """
    actions = []
    start_x, start_y = start_local_tile
    goal_x, goal_y = goal_local_tile
    
    # Calculate deltas
    delta_x = goal_x - start_x
    delta_y = goal_y - start_y
    
    # Move horizontally first
    if delta_x > 0:
        # Move right
        actions.extend(["RIGHT"] * delta_x)
    elif delta_x < 0:
        # Move left
        actions.extend(["LEFT"] * abs(delta_x))
    
    # Then move vertically
    if delta_y > 0:
        # Move down
        actions.extend(["DOWN"] * delta_y)
    elif delta_y < 0:
        # Move up
        actions.extend(["UP"] * abs(delta_y))
    
    return actions
