"""
Memory Manager Module

Provides a modular and extensible memory management system for agents.
Currently supports location-based memory with the ability to add other memory types.

Memory Structure:
{
    "location": {
        "map_name_1": "text information about this location",
        "map_name_2": "text information about this location"
    },
    "others": {
        "key1": "value1",
        "key2": "value2"
    }
}
"""

import json
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages different types of memories for agents in a modular, extensible way.
    """
    
    def __init__(self):
        """Initialize the memory manager with empty memory stores."""
        self.memories: Dict[str, Dict[str, Any]] = {
            "location": {},
            "others": {}
        }
    
    def set_location_memory(self, map_name: str, information: str) -> None:
        """
        Store or update location-based memory.
        
        Args:
            map_name: Name of the map/location (e.g., "LITTLEROOT_TOWN", "ROUTE_101")
            information: Text information about this location
        """
        self.memories["location"][map_name] = information
        logger.debug(f"Updated location memory for {map_name}")
    
    def get_location_memory(self, map_name: str) -> Optional[str]:
        """
        Retrieve location-based memory.
        
        Args:
            map_name: Name of the map/location
            
        Returns:
            Text information about the location, or None if not found
        """
        return self.memories["location"].get(map_name)
    
    def get_all_location_memories(self) -> Dict[str, str]:
        """
        Get all location memories.
        
        Returns:
            Dictionary mapping map names to their information
        """
        return self.memories["location"].copy()
    
    def clear_location_memory(self, map_name: Optional[str] = None) -> None:
        """
        Clear location memory.
        
        Args:
            map_name: If provided, clear only this location. Otherwise, clear all.
        """
        if map_name:
            if map_name in self.memories["location"]:
                del self.memories["location"][map_name]
                logger.info(f"Cleared location memory for {map_name}")
        else:
            self.memories["location"].clear()
            logger.info("Cleared all location memories")
    
    def set_other_memory(self, memory_type: str, key: str, value: Any) -> None:
        """
        Store other types of memory (extensible).
        
        Args:
            memory_type: Category of memory (will be added to top-level if not exists)
            key: Memory key
            value: Memory value
        """
        if memory_type not in self.memories:
            self.memories[memory_type] = {}
        
        self.memories[memory_type][key] = value
        logger.debug(f"Updated {memory_type} memory: {key}")
    
    def get_other_memory(self, memory_type: str, key: str) -> Optional[Any]:
        """
        Retrieve other types of memory.
        
        Args:
            memory_type: Category of memory
            key: Memory key
            
        Returns:
            Memory value, or None if not found
        """
        return self.memories.get(memory_type, {}).get(key)
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save all memories to a JSON file.
        
        Args:
            filepath: Path to save the memory file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.memories, f, indent=2)
            
            logger.info(f"Saved memories to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memories to {filepath}: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load memories from a JSON file.
        
        Args:
            filepath: Path to the memory file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Memory file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                loaded_memories = json.load(f)
            
            # Validate structure - ensure at minimum 'location' exists
            if "location" not in loaded_memories:
                loaded_memories["location"] = {}
            if "others" not in loaded_memories:
                loaded_memories["others"] = {}
            
            self.memories = loaded_memories
            logger.info(f"Loaded memories from {filepath}")
            logger.info(f"  Location memories: {len(self.memories['location'])}")
            logger.info(f"  Other memory types: {len([k for k in self.memories.keys() if k not in ['location', 'others']])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memories from {filepath}: {e}")
            return False
    
    def get_memory_summary(self) -> str:
        """
        Get a human-readable summary of stored memories.
        
        Returns:
            Formatted string summarizing memory contents
        """
        lines = []
        lines.append("=== Memory Summary ===")
        
        # Location memories
        location_count = len(self.memories["location"])
        lines.append(f"Location memories: {location_count}")
        if location_count > 0:
            for map_name, info in list(self.memories["location"].items())[:5]:
                preview = info[:50] + "..." if len(info) > 50 else info
                lines.append(f"  - {map_name}: {preview}")
            if location_count > 5:
                lines.append(f"  ... and {location_count - 5} more")
        
        # Other memory types
        other_types = [k for k in self.memories.keys() if k not in ["location", "others"]]
        if other_types:
            lines.append(f"Other memory types: {', '.join(other_types)}")
        
        if self.memories.get("others"):
            lines.append(f"Others: {len(self.memories['others'])} entries")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export memories as a dictionary.
        
        Returns:
            Dictionary containing all memories
        """
        return self.memories.copy()
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import memories from a dictionary.
        
        Args:
            data: Dictionary containing memories
        """
        # Ensure required keys exist
        if "location" not in data:
            data["location"] = {}
        if "others" not in data:
            data["others"] = {}
        
        self.memories = data
        logger.info("Imported memories from dictionary")
