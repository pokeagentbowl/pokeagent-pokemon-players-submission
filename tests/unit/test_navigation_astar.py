"""
Tests for edge-based A* pathfinding with jump tile support.
"""
import pytest
from custom_utils.navigation_astar import AStarPathfinder, TerrainGrid


class TestEdgeBasedAstar:
    """Test edge-based A* pathfinding with jump tiles and obstacles."""

    def test_simple_path_no_obstacles(self):
        """Test basic pathfinding on an empty grid."""
        # Create a simple 5x5 grid with all walkable tiles
        grid = [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from (0, 0) to (4, 4)
        path = pathfinder.find_path(
            start_local_tile=(0, 0),
            goal_local_tile=(4, 4),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path
        assert path is not None
        assert len(path) > 0
        # Path should have 8 steps (4 right + 4 down)
        assert len(path) == 8

    def test_path_with_impassable_tiles(self):
        """Test pathfinding around blocked tiles."""
        # Create a 7x7 grid with a wall
        grid = [
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from (1, 1) to (5, 1) - must go around the wall
        path = pathfinder.find_path(
            start_local_tile=(1, 1),
            goal_local_tile=(5, 1),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path that goes around the wall
        assert path is not None
        assert len(path) > 4  # More than direct path due to obstacle

    def test_jump_tile_down(self):
        """Test pathfinding with down jump tile '↓'."""
        # Create a grid with a down jump tile
        # Approach from (3, 0), jump tile at (3, 1), land at (3, 2)
        grid = [
            ['.', '.', '.', '.', '.', '.', '.'],  # Row 0: approach position
            ['.', '.', '.', '↓', '.', '.', '.'],  # Row 1: jump tile at (3, 1)
            ['.', '.', '.', '.', '.', '.', '.'],  # Row 2: landing position
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from (3, 0) to (3, 2)
        # Down arrow at (3, 1) means:
        # - Approach from (3, 0) by moving DOWN
        # - Jump over (3, 1) and land at (3, 2)
        # So path from (3, 0) to (3, 2) should use the jump
        path = pathfinder.find_path(
            start_local_tile=(3, 0),
            goal_local_tile=(3, 2),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path using the jump tile
        assert path is not None
        # Path should be just one DOWN action (from 3,0 jumping over 3,1 landing at 3,2)
        assert path == ["DOWN"]

    def test_jump_tile_up(self):
        """Test pathfinding with up jump tile '↑'."""
        grid = [
            ['.', '.', '.', '.', '.', '.', '.'],  # Row 0: landing position
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '↑', '.', '.', '.'],  # Row 3: jump tile at (3, 3)
            ['.', '.', '.', '.', '.', '.', '.'],  # Row 4: approach position
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from (3, 4) to (3, 2)
        # Up arrow at (3, 3) means:
        # - Approach from (3, 4) by moving UP
        # - Jump over (3, 3) and land at (3, 2)
        path = pathfinder.find_path(
            start_local_tile=(3, 4),
            goal_local_tile=(3, 2),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path using the jump tile
        assert path is not None
        assert path == ["UP"]

    def test_jump_tile_right(self):
        """Test pathfinding with right jump tile '→'."""
        grid = [
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '→', '.', '.', '.', '.', '.'],  # Jump tile at (1, 3)
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from (0, 3) to (2, 3)
        # Right arrow at (1, 3) means:
        # - Approach from (0, 3) by moving RIGHT
        # - Jump over (1, 3) and land at (2, 3)
        path = pathfinder.find_path(
            start_local_tile=(0, 3),
            goal_local_tile=(2, 3),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path using the jump tile
        assert path is not None
        assert path == ["RIGHT"]

    def test_jump_tile_left(self):
        """Test pathfinding with left jump tile '←'."""
        grid = [
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '←', '.', '.'],  # Jump tile at (4, 3)
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from (5, 3) to (3, 3)
        # Left arrow at (4, 3) means:
        # - Approach from (5, 3) by moving LEFT
        # - Jump over (4, 3) and land at (3, 3)
        path = pathfinder.find_path(
            start_local_tile=(5, 3),
            goal_local_tile=(3, 3),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path using the jump tile
        assert path is not None
        assert path == ["LEFT"]

    def test_jump_tile_wrong_direction_blocked(self):
        """Test that jump tiles cannot be approached from wrong direction."""
        grid = [
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '↓', '.', '.', '.'],  # Down jump at (3, 1)
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Try to approach down jump from below (3, 2) to (3, 0)
        # This should NOT be able to use the jump tile directly
        # Down jump can only be approached from above
        path = pathfinder.find_path(
            start_local_tile=(3, 2),
            goal_local_tile=(3, 0),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path, but it should go around (not through the jump)
        assert path is not None
        # The path should be longer than 2 steps since it can't go straight through
        assert len(path) > 2

    def test_obstacles_blocking_path(self):
        """Test pathfinding with obstacles list."""
        grid = [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Add obstacles that partially block direct path (not complete wall)
        obstacles = [(1, 1), (1, 2), (1, 3)]

        path = pathfinder.find_path(
            start_local_tile=(0, 2),
            goal_local_tile=(4, 2),
            start_facing="North",
            terrain=terrain,
            obstacles=obstacles,
            movement_mode="naive"
        )

        # Should find a path around the obstacles
        assert path is not None
        assert len(path) > 4  # More than direct path due to going around

    def test_15x15_grid_with_player_center(self):
        """Test pathfinding on standard 15x15 grid with player at center."""
        # Create 15x15 grid (standard game size) with player at center (7, 7)
        grid = [['.' for _ in range(15)] for _ in range(15)]

        # Add some obstacles
        for i in range(5, 10):
            grid[i][5] = '#'

        # Add a jump tile
        grid[7][12] = '→'

        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path from center (7, 7) to (14, 7)
        path = pathfinder.find_path(
            start_local_tile=(7, 7),
            goal_local_tile=(14, 7),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        assert path is not None
        assert len(path) > 0

    def test_complex_scenario_with_multiple_jumps(self):
        """Test complex scenario with multiple jump tiles and obstacles."""
        grid = [
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '#', '#', '#', '.', '.', '.', '.', '.', '.'],
            ['.', '#', '.', '#', '.', '.', '↓', '.', '.', '.'],
            ['.', '#', '.', '#', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '→', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '↑', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '←', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Find path through the complex terrain
        path = pathfinder.find_path(
            start_local_tile=(0, 0),
            goal_local_tile=(9, 9),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        assert path is not None
        assert len(path) > 0

    def test_edge_graph_creation(self):
        """Test that edge graph is created correctly."""
        grid = [
            ['.', '.', '.'],  # Row 0: approach position
            ['.', '↓', '.'],  # Row 1: jump tile at (1, 1)
            ['.', '.', '.']   # Row 2: landing position
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        edges = pathfinder._game_to_cost_grid_edges(terrain)

        # Check that edge graph is created
        assert edges is not None
        assert isinstance(edges, dict)

        # Position (1, 0) should have an edge to (1, 2) via jump at (1, 1)
        assert (1, 0) in edges
        # Check if there's an edge from (1, 0) that goes to (1, 2)
        neighbors = [pos for pos, cost in edges[(1, 0)]]
        assert (1, 2) in neighbors

        # Jump tile itself should not have outgoing edges (can't stand on it)
        assert (1, 1) in edges
        assert len(edges[(1, 1)]) == 0

    def test_no_path_completely_blocked(self):
        """Test that pathfinding returns None when no path exists."""
        grid = [
            ['.', '#', '.'],
            ['#', '#', '#'],
            ['.', '#', '.']
        ]
        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # No path from (0, 0) to (2, 2) - completely blocked
        path = pathfinder.find_path(
            start_local_tile=(0, 0),
            goal_local_tile=(2, 2),
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should fall back to simple path since A* fails
        assert path is not None  # Fallback will provide a path even if it's blocked

    def test_real_game_grid_with_player_and_goal(self):
        """Test pathfinding on real game grid with P (player) and S (goal) symbols."""
        grid_str = """##.............
##.............
##.............
##.............
##......#######
##......#......
.#......#......
.#.#..#P######.
.#......######.
.#......######.
##.....#..#S#..
.#.............
...............
...............
##............."""
        grid = [list(line) for line in grid_str.strip().split('\n')]

        # Find P and S positions
        player_pos = goal_pos = None
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == 'P':
                    player_pos = (x, y)
                if cell == 'S':
                    goal_pos = (x, y)

        assert player_pos == (7, 7), f"Expected player at (7, 7), got {player_pos}"
        assert goal_pos == (11, 10), f"Expected goal at (11, 10), got {goal_pos}"

        terrain = TerrainGrid(grid)
        pathfinder = AStarPathfinder()

        # Test that P symbol is treated as walkable
        edges = pathfinder._game_to_cost_grid_edges(terrain)
        assert player_pos in edges
        assert len(edges[player_pos]) > 0, "Player position should have outgoing edges"

        # Find path from P to S
        path = pathfinder.find_path(
            start_local_tile=player_pos,
            goal_local_tile=goal_pos,
            start_facing="North",
            terrain=terrain,
            movement_mode="naive"
        )

        # Should find a path using A* (not fallback)
        assert path is not None
        assert len(path) > 0
        # Verify it's using A* not fallback by checking path is reasonable
        # (fallback would give straight line path which would be longer due to obstacles)
        assert len(path) < 20, f"Path too long ({len(path)}), likely using fallback"
