# Enhanced Save/Load State API

The save_state and load_state endpoints now support folder-based saves that include all relevant game data including agent memory.

## API Endpoints

### POST /save_state

Save game state with optional agent memory to a folder or single file.

#### Folder-based Save (Recommended)

```json
{
  "folder_path": ".pokeagent_cache/my_save",
  "save_memory": true,
  "memory_data": {
    "location": {
      "LITTLEROOT_TOWN": "Starting town info",
      "ROUTE_101": "Route info"
    },
    "others": {}
  }
}
```

**Saves:**
- `game.state` - Emulator state
- `milestones.json` - Milestone data
- `memory.json` - Agent memory (if provided)

**Response:**
```json
{
  "status": "success",
  "message": "State saved to folder .pokeagent_cache/my_save",
  "files": {
    "state": ".pokeagent_cache/my_save/game.state",
    "milestones": ".pokeagent_cache/my_save/milestones.json",
    "memory": ".pokeagent_cache/my_save/memory.json"
  }
}
```

#### Single File Save (Legacy)

```json
{
  "filepath": ".pokeagent_cache/manual_save.state",
  "save_memory": true,
  "memory_data": {...}
}
```

**Saves:**
- Emulator state to specified file
- `memory.json` in same directory (if memory_data provided)

### POST /load_state

Load game state from a folder or single file, with automatic memory restoration.

#### Folder-based Load (Recommended)

```json
{
  "folder_path": ".pokeagent_cache/my_save",
  "preserve_map_cache": false
}
```

**Loads:**
- `game.state` - Emulator state
- `milestones.json` - Milestone data (if exists)
- `memory.json` - Agent memory (if exists)

**Response:**
```json
{
  "status": "success",
  "message": "State loaded from folder .pokeagent_cache/my_save",
  "memory_data": {
    "location": {
      "LITTLEROOT_TOWN": "Starting town info"
    },
    "others": {}
  },
  "files_loaded": {
    "state": ".pokeagent_cache/my_save/game.state",
    "milestones": ".pokeagent_cache/my_save/milestones.json",
    "memory": ".pokeagent_cache/my_save/memory.json"
  }
}
```

The `memory_data` in the response should be used by the client to restore agent memory.

#### Single File Load (Legacy)

```json
{
  "filepath": ".pokeagent_cache/manual_save.state",
  "preserve_map_cache": false
}
```

## Client Integration

The client (`server/client.py`) automatically handles agent memory when using the LocationSimpleAgent:

### Save State (Press '1' in client)

```python
# Automatically includes agent memory if available
save_request = {
    "folder_path": ".pokeagent_cache/manual_save",
    "save_memory": True,
    "memory_data": agent.agent_impl.memory_manager.to_dict()
}
```

### Load State (Press '2' in client)

```python
# Automatically restores agent memory from response
memory_data = response.get("memory_data")
if memory_data:
    agent.agent_impl.memory_manager.from_dict(memory_data)
```

## Folder Structure

When using folder-based saves, the structure is:

```
.pokeagent_cache/
└── my_save/
    ├── game.state       # Emulator save state
    ├── milestones.json  # Game milestone data
    └── memory.json      # Agent memory (LocationSimpleAgent)
```

## Benefits of Folder-based Saves

1. **All data in one place**: Game state, milestones, and agent memory together
2. **Easy to backup**: Copy entire folder to preserve everything
3. **Easy to share**: Share folder with others to reproduce exact game state
4. **Organized**: Multiple saves in separate folders
5. **Extensible**: Easy to add more data files in the future

## Migration from Legacy

Old code using single file saves still works:

```python
# Still works (legacy mode)
requests.post(f"{server_url}/save_state", 
              json={"filepath": ".pokeagent_cache/manual_save.state"})
```

New code should use folder-based saves:

```python
# Recommended (folder mode)
requests.post(f"{server_url}/save_state", 
              json={"folder_path": ".pokeagent_cache/manual_save"})
```

## Examples

### Python Script Example

```python
import requests

server_url = "http://localhost:8000"

# Save with folder
response = requests.post(f"{server_url}/save_state", json={
    "folder_path": "saves/checkpoint_1",
    "save_memory": True,
    "memory_data": {
        "location": {
            "LITTLEROOT_TOWN": "Visited Prof Birch's lab"
        },
        "others": {}
    }
})

# Load from folder
response = requests.post(f"{server_url}/load_state", json={
    "folder_path": "saves/checkpoint_1"
})

memory_data = response.json().get("memory_data")
print(f"Loaded memory: {memory_data}")
```

### Checkpoint System Integration

The checkpoint system (`utils/checkpoint.py`) already supports agent memory through the `location_simple_agent` parameter. The new API provides the same functionality through HTTP endpoints for client-server architectures.

## Keyboard Shortcuts

When running the client with display:

- **Press '1'**: Save state (folder-based with memory)
- **Press '2'**: Load state (folder-based with memory)

The client automatically detects if the agent has memory capabilities and includes/restores it accordingly.
