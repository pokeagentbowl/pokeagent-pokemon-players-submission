"""
Tests for navigation target generation from traversability maps.
"""
from custom_utils.navigation_targets import find_known_objects, find_warp_points, generate_navigation_targets


class TestFindKnownObjects:
    """Test finding known objects (PC, TV, Bookshelf, Clock, NPC) in traversability maps."""

    def test_find_single_pc(self):
        """Test finding a single PC in the map."""
        # Create a 15x15 grid with a PC at position (7, 7)
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[7][7] = 'C'  # PC

        objects = find_known_objects(grid)

        assert len(objects) == 1
        assert objects[0] == (7, 7, 'C')

    def test_find_single_television(self):
        """Test finding a single TV in the map."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[5][3] = 'T'  # Television

        objects = find_known_objects(grid)

        assert len(objects) == 1
        assert objects[0] == (3, 5, 'T')

    def test_find_single_bookshelf(self):
        """Test finding a single bookshelf in the map."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[8][10] = 'B'  # Bookshelf

        objects = find_known_objects(grid)

        assert len(objects) == 1
        assert objects[0] == (10, 8, 'B')

    def test_find_single_clock(self):
        """Test finding a single clock in the map."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[4][12] = 'K'  # Clock

        objects = find_known_objects(grid)

        assert len(objects) == 1
        assert objects[0] == (12, 4, 'K')

    def test_find_single_npc(self):
        """Test finding a single NPC in the map."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[6][8] = 'N'  # NPC

        objects = find_known_objects(grid)

        assert len(objects) == 1
        assert objects[0] == (8, 6, 'N')

    def test_find_multiple_objects(self):
        """Test finding multiple objects of different types."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[3][2] = 'C'  # PC
        grid[5][7] = 'T'  # TV
        grid[8][10] = 'B'  # Bookshelf
        grid[4][12] = 'K'  # Clock
        grid[9][5] = 'N'  # NPC

        objects = find_known_objects(grid)

        assert len(objects) == 5
        # Check all objects are found (order may vary)
        object_tuples = set(objects)
        assert (2, 3, 'C') in object_tuples
        assert (7, 5, 'T') in object_tuples
        assert (10, 8, 'B') in object_tuples
        assert (12, 4, 'K') in object_tuples
        assert (5, 9, 'N') in object_tuples

    def test_find_multiple_same_type(self):
        """Test finding multiple objects of the same type."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[3][2] = 'N'  # NPC 1
        grid[5][7] = 'N'  # NPC 2
        grid[8][10] = 'N'  # NPC 3

        objects = find_known_objects(grid)

        assert len(objects) == 3
        # All should be NPCs
        for obj in objects:
            assert obj[2] == 'N'

    def test_empty_map(self):
        """Test map with no known objects."""
        grid = [['.' for _ in range(15)] for _ in range(15)]

        objects = find_known_objects(grid)

        assert len(objects) == 0

    def test_objects_outside_valid_range(self):
        """Test that objects in rows 0-1 and 13-14 are not detected."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[0][5] = 'C'  # Outside range (row 0)
        grid[1][5] = 'T'  # Outside range (row 1)
        grid[13][5] = 'B'  # Outside range (row 13)
        grid[14][5] = 'K'  # Outside range (row 14)
        grid[7][7] = 'N'  # Inside range (row 7)

        objects = find_known_objects(grid)

        # Only the NPC in the valid range should be found
        assert len(objects) == 1
        assert objects[0] == (7, 7, 'N')

    def test_ignores_other_symbols(self):
        """Test that other symbols are not detected as known objects."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[3][2] = '#'  # Wall
        grid[5][7] = 'D'  # Door
        grid[8][10] = 'S'  # Stairs
        grid[4][12] = '~'  # Water
        grid[9][5] = 'N'  # NPC (should be detected)

        objects = find_known_objects(grid)

        # Only the NPC should be detected
        assert len(objects) == 1
        assert objects[0] == (5, 9, 'N')


class TestGenerateNavigationTargets:
    """Test generating navigation targets including known objects."""

    def test_generate_targets_with_single_pc(self):
        """Test generating navigation targets with a single PC."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[7][7] = 'C'  # PC

        player_pos = (100, 100)  # Player at map position (100, 100)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find the PC target
        pc_targets = [t for t in targets if t.entity_type == 'PC/Computer']
        assert len(pc_targets) == 1

        pc_target = pc_targets[0]
        assert pc_target.type == "object"
        assert pc_target.local_tile_position == (7, 7)
        assert "PC/Computer" in pc_target.description
        assert pc_target.tile_size == (1, 1)

    def test_generate_targets_with_multiple_objects(self):
        """Test generating navigation targets with multiple object types."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[3][2] = 'C'  # PC
        grid[5][7] = 'T'  # TV
        grid[8][10] = 'B'  # Bookshelf

        player_pos = (50, 50)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find all known object targets
        pc_targets = [t for t in targets if t.entity_type == 'PC/Computer']
        tv_targets = [t for t in targets if t.entity_type == 'Television']
        bookshelf_targets = [t for t in targets if t.entity_type == 'Bookshelf']

        assert len(pc_targets) == 1
        assert len(tv_targets) == 1
        assert len(bookshelf_targets) == 1

    def test_generate_targets_with_npc(self):
        """Test generating navigation targets with NPCs."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[6][8] = 'N'  # NPC

        player_pos = (0, 0)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find the NPC target
        npc_targets = [t for t in targets if t.entity_type == 'NPC']
        assert len(npc_targets) == 1

        npc_target = npc_targets[0]
        assert npc_target.type == "object"
        assert npc_target.local_tile_position == (8, 6)
        assert "NPC" in npc_target.description

    def test_generate_targets_with_clock(self):
        """Test generating navigation targets with a clock."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[4][12] = 'K'  # Clock

        player_pos = (200, 200)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find the clock target
        clock_targets = [t for t in targets if t.entity_type == 'Clock (Wall)']
        assert len(clock_targets) == 1

        clock_target = clock_targets[0]
        assert clock_target.type == "object"
        assert clock_target.local_tile_position == (12, 4)
        assert "Clock (Wall)" in clock_target.description

    def test_generate_targets_mixed_with_warp_points(self):
        """Test generating targets with both known objects and warp points."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[3][2] = 'C'  # PC
        grid[5][7] = 'D'  # Door
        grid[8][10] = 'T'  # TV
        grid[9][12] = 'S'  # Stairs

        player_pos = (100, 100)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find different target types
        pc_targets = [t for t in targets if t.entity_type == 'PC/Computer']
        tv_targets = [t for t in targets if t.entity_type == 'Television']
        door_targets = [t for t in targets if t.type == 'door']
        stairs_targets = [t for t in targets if t.type == 'stairs']

        assert len(pc_targets) == 1
        assert len(tv_targets) == 1
        assert len(door_targets) == 1
        assert len(stairs_targets) == 1

    def test_target_ids_are_unique(self):
        """Test that target IDs are unique when multiple objects of same type exist."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[3][2] = 'N'  # NPC 1
        grid[5][7] = 'N'  # NPC 2
        grid[8][10] = 'N'  # NPC 3

        player_pos = (0, 0)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find all NPC targets
        npc_targets = [t for t in targets if t.entity_type == 'NPC']
        assert len(npc_targets) == 3

        # Check that all IDs are unique
        ids = [t.id for t in npc_targets]
        assert len(ids) == len(set(ids))

    def test_map_tile_position_conversion(self):
        """Test that local tile positions are correctly converted to map positions."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[7][7] = 'C'  # PC at center of local grid

        player_pos = (100, 100)  # Player at map position (100, 100)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # Find the PC target
        pc_targets = [t for t in targets if t.entity_type == 'PC/Computer']
        assert len(pc_targets) == 1

        pc_target = pc_targets[0]
        # Local position (7, 7) with player at (100, 100) should give map position (100, 100)
        # since (7, 7) is the center of the 15x15 grid
        assert pc_target.map_tile_position == (100, 100)

    def test_empty_map_no_known_objects(self):
        """Test that no known object targets are generated for empty map."""
        grid = [['.' for _ in range(15)] for _ in range(15)]

        player_pos = (0, 0)
        targets = generate_navigation_targets([], grid, player_pos, "TEST_LOCATION")

        # No known object targets should be generated
        known_object_targets = [t for t in targets if t.entity_type in
                                ['PC/Computer', 'Television', 'Bookshelf', 'Clock (Wall)', 'NPC']]
        assert len(known_object_targets) == 0


class TestFindWarpPoints:
    """Test finding warp points for comparison with known objects."""

    def test_find_door_and_stairs(self):
        """Test that warp points are still detected correctly."""
        grid = [['.' for _ in range(15)] for _ in range(15)]
        grid[5][7] = 'D'  # Door
        grid[9][12] = 'S'  # Stairs

        warps = find_warp_points(grid)

        assert len(warps) == 2
        warp_tuples = set(warps)
        assert (7, 5, 'D') in warp_tuples
        assert (12, 9, 'S') in warp_tuples
