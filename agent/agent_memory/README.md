# Agent Memory Module

Extensible memory management system for Pokémon Emerald agents.

## Overview

The `agent_memory` module provides a modular and extensible framework for agents to store and retrieve different types of memories. Currently focused on location-based memory, it's designed to easily accommodate additional memory types in the future.

## Features

- **Location-Based Memory**: Store and retrieve text information about game locations (maps)
- **Extensible Architecture**: Easy to add new memory types beyond locations
- **JSON Persistence**: Save/load memories to/from JSON files
- **Checkpoint Integration**: Integrated with the game's checkpoint system
- **Clean API**: Simple, intuitive methods for memory operations

## Memory Structure

Memories are stored in a JSON structure:

```json
{
  "location": {
    "LITTLEROOT_TOWN": "Starting town with Prof Birch's lab and rival's house",
    "ROUTE_101": "Route between Littleroot and Oldale, encountered wild Pokemon",
    "OLDALE_TOWN": "Small town with Pokemon Center"
  },
  "others": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

## Usage

### Basic Operations

```python
from agent_memory import MemoryManager

# Create memory manager
mm = MemoryManager()

# Store location memory
mm.set_location_memory("LITTLEROOT_TOWN", "Starting town with lab")

# Retrieve location memory
info = mm.get_location_memory("LITTLEROOT_TOWN")
print(info)  # "Starting town with lab"

# Get all location memories
all_locations = mm.get_all_location_memories()

# Clear specific location
mm.clear_location_memory("LITTLEROOT_TOWN")

# Clear all locations
mm.clear_location_memory()
```

### Save and Load

```python
# Save to file
mm.save_to_file(".pokeagent_cache/memory.json")

# Load from file
mm2 = MemoryManager()
mm2.load_from_file(".pokeagent_cache/memory.json")
```

### Extensible Memory Types

```python
# Add custom memory types
mm.set_other_memory("npc_info", "rival", "May lives next door")
mm.set_other_memory("items", "potion_count", 5)

# Retrieve custom memory
rival_info = mm.get_other_memory("npc_info", "rival")
```

## Integration with LocationSimpleAgent

The `LocationSimpleAgent` uses the `MemoryManager` to maintain location memories:

```python
from agent import Agent

# Create agent with location_simple scaffold
args.scaffold = "location_simple"
agent = Agent(args)

# Agent automatically reads and updates location memory each turn
# The LLM can update memory using the LOCATION_MEMORY: directive in responses
```

## Checkpoint Integration

The memory system is integrated with the game's checkpoint system. Any agent with a `memory_manager` attribute can use this:

```python
from utils.checkpoint import save_checkpoint, load_checkpoint

# Save checkpoint (includes memory for any agent with memory_manager)
save_checkpoint(
    emulator=env,
    llm_logger=llm_logger,
    agent_step_count=step_count,
    memory_agent=agent.agent_impl  # Any agent with memory_manager attribute
)

# Load checkpoint (includes memory for any agent with memory_manager)
load_checkpoint(
    emulator=env,
    llm_logger=llm_logger,
    memory_agent=agent.agent_impl  # Any agent with memory_manager attribute
)
```

The checkpoint system automatically detects if the agent has a `memory_manager` attribute and handles saving/loading accordingly. This works with:
- `LocationSimpleAgent` (has built-in memory_manager)
- Any custom agent that includes a `memory_manager` attribute
- Backward compatible with agents that have `save_memory()` and `load_memory()` methods

## Files

- `memory_manager.py`: Core memory management class
- `__init__.py`: Module exports
- `README.md`: This file

## Future Enhancements

The system is designed to easily support additional memory types:

- **Quest/Objective Memory**: Track progress on specific quests
- **NPC Memory**: Remember NPC locations and dialogue
- **Battle Memory**: Record battle strategies and outcomes
- **Item Memory**: Track items found or needed
- **Exploration Memory**: Mark explored vs unexplored areas

Simply add new top-level keys to the memory structure and implement methods as needed.
