# LocationSimpleAgent

A simplified agent focused on location-based memory management for Pokémon Emerald.

## Overview

`LocationSimpleAgent` is a streamlined agent that emphasizes location memory without the complexity of objectives tracking, storyline management, or extensive movement memory found in `SimpleAgent`. It provides a clean, focused approach to building spatial awareness through persistent location memories.

## Key Features

- **Location-Based Memory**: Stores and retrieves text information about each map/location
- **Automatic Memory Updates**: Reads and updates location memory every turn
- **LLM-Driven Memory**: The LLM decides what information to remember about locations
- **Persistent Storage**: Memories saved in `memory.json` within checkpoint directory
- **Simplified Design**: Removed unnecessary features from SimpleAgent for clarity
- **Extensible Memory System**: Built on the modular `agent_memory` framework

## Usage

### Starting with LocationSimpleAgent

```bash
# Run with location_simple scaffold
python run.py --scaffold location_simple --backend gemini --model-name gemini-2.5-flash
```

### Command Line Arguments

The agent supports standard arguments:
- `--scaffold location_simple`: Use LocationSimpleAgent
- `--backend`: VLM backend (gemini, openai, etc.)
- `--model-name`: Specific model to use
- `--load-checkpoint`: Load saved game state and memories

## How It Works

### Each Turn

1. **Get Current Map**: Extracts the current map name from game state
2. **Load Memory Context**: Retrieves stored information about current and nearby locations
3. **Process Frame**: Sends frame and context to LLM
4. **Update Memory**: LLM can update location memory via `LOCATION_MEMORY:` directive
5. **Execute Action**: Chosen action is executed in the game

### Memory Structure

Location memories are stored in `.pokeagent_cache/memory.json`:

```json
{
  "location": {
    "LITTLEROOT_TOWN": "Starting town. Prof Birch's lab is to the north. Rival May lives next door.",
    "ROUTE_101": "Route connecting Littleroot and Oldale. Tall grass with wild Poochyena.",
    "OLDALE_TOWN": "Small town with Pokemon Center and Pokemart. NPCs mention Route 103."
  },
  "others": {}
}
```

### LLM Response Format

The agent expects structured responses from the LLM:

```
ANALYSIS:
[What's happening in the current scene]

LOCATION_MEMORY: [Optional - information to remember about this location]

PLAN:
[Immediate strategy]

ACTION:
[Single action like UP, DOWN, LEFT, RIGHT, A, B, etc.]
```

## Differences from SimpleAgent

| Feature | SimpleAgent | LocationSimpleAgent |
|---------|-------------|---------------------|
| Objectives System | ✅ Full storyline + custom | ❌ Removed |
| Movement Memory | ✅ Failed movements + NPCs | ❌ Removed |
| Location Memory | ❌ No structured system | ✅ Core feature with persistence |
| History Tracking | ✅ Extensive (100 entries) | ✅ Simplified (20 entries) |
| Stuck Detection | ✅ Complex coordinate-based | ❌ Removed |
| Code Complexity | ~1600 lines | ~400 lines |

## Memory Operations

The agent provides methods for memory management:

```python
from agent import LocationSimpleAgent, get_location_simple_agent
from utils.vlm import VLM

vlm = VLM(backend="gemini", model_name="gemini-2.5-flash")
agent = get_location_simple_agent(vlm)

# Option 1: Use agent wrapper methods (recommended for save/load)
agent.save_memory(".pokeagent_cache/memory.json")
agent.load_memory(".pokeagent_cache/memory.json")

# Option 2: Access memory_manager directly for detailed operations
map_name = "LITTLEROOT_TOWN"
memory = agent.memory_manager.get_location_memory(map_name)
agent.memory_manager.set_location_memory(map_name, "Starting town with lab")
```

## Checkpoint Integration

Memories are automatically saved and loaded with checkpoints:

```python
from utils.checkpoint import save_checkpoint, load_checkpoint

# Save (includes memory.json)
save_checkpoint(
    emulator=env,
    llm_logger=llm_logger,
    agent_step_count=step_count,
    memory_agent=agent.agent_impl  # Works with any agent having memory_manager
)

# Load (includes memory.json)
load_checkpoint(
    emulator=env,
    llm_logger=llm_logger,
    memory_agent=agent.agent_impl  # Works with any agent having memory_manager
)
```

The parameter name changed from `location_simple_agent` to `memory_agent` to reflect that it now works with any agent that has a `memory_manager` attribute, not just LocationSimpleAgent.

## Development Tips

### Extending Memory Types

The underlying `MemoryManager` supports additional memory types:

```python
# Add custom memory categories
agent.memory_manager.set_other_memory("npc_dialogue", "prof_birch", "Gave me starter Pokémon")
agent.memory_manager.set_other_memory("items_found", "route_101", "Potion in grass")

# Retrieve custom memories
dialogue = agent.memory_manager.get_other_memory("npc_dialogue", "prof_birch")
```

### Testing

Test the agent with a simple game state:

```python
game_state = {
    'frame': frame,  # PIL Image
    'player': {
        'location': 'LITTLEROOT_TOWN'
    },
    'map': {
        'name': 'LITTLEROOT_TOWN',
        'id': 1
    }
}

result = agent.step(game_state)
print(result['action'])  # e.g., 'UP'
```

## Future Enhancements

Potential additions while maintaining simplicity:
- **Smart Memory Summarization**: Automatically condense verbose memories
- **Memory Relevance Scoring**: Prioritize important location information
- **Cross-Location References**: Link related locations in memory
- **Memory Export/Import**: Share memories between runs or agents

## Architecture

```
LocationSimpleAgent
├── memory_manager (MemoryManager)
│   ├── location memories
│   └── extensible other memories
├── state (LocationSimpleAgentState)
│   ├── history (last 20 entries)
│   ├── recent_actions (last 30 actions)
│   └── step_counter
└── vlm (VLM client)
```

## See Also

- [agent_memory/README.md](agent_memory/README.md) - Memory system documentation
- [agent/simple.py](agent/simple.py) - Full-featured SimpleAgent
- [utils/checkpoint.py](utils/checkpoint.py) - Checkpoint system integration
