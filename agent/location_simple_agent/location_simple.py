"""
Location Simple Agent Module

A simplified agent focused on location-based memory management.
Based on SimpleAgent but streamlined to focus on location memory without
complex objectives, storyline tracking, or movement memory.

Key features:
- Location-based memory that persists across sessions
- Read and update location memory every turn
- Simple history tracking for context
- Direct frame + state -> action processing
"""

import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from utils.state_formatter import format_state_for_llm
from agent.agent_memory import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class LocationSimpleAgentState:
    """Maintains minimal state for the location simple agent"""
    history: deque = None
    recent_actions: deque = None
    step_counter: int = 0
    current_map_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize deques with default values"""
        if self.history is None:
            self.history = deque(maxlen=20)  # Keep last 20 history entries
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=30)  # Keep last 30 actions


class LocationSimpleAgent:
    """
    Simplified agent that focuses on location-based memory management.
    """
    
    def __init__(self, vlm):
        self.vlm = vlm
        self.state = LocationSimpleAgentState()
        self.memory_manager = MemoryManager()
        
        # Auto-save memory file path (like MapStitcher)
        self.memory_file = ".pokeagent_cache/memory.json"
        os.makedirs(".pokeagent_cache", exist_ok=True)
        
        # Load existing memory if available
        if os.path.exists(self.memory_file):
            self.memory_manager.load_from_file(self.memory_file)
            logger.info(f"Loaded existing memory from {self.memory_file}")
        
        logger.info("Initialized LocationSimpleAgent with memory management")
    
    def get_map_name(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Extract map name from game state.
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Map name string or None
        """
        try:
            # Try to get location name from player data
            player = game_state.get("player", {})
            location = player.get("location")
            if location:
                return location
            
            # Fallback: try to get map name from map data
            map_data = game_state.get("map", {})
            map_name = map_data.get("name")
            if map_name:
                return map_name
            
            # Last resort: use map ID
            map_id = map_data.get("id")
            if map_id is not None:
                return f"MAP_{map_id}"
                
        except Exception as e:
            logger.warning(f"Error getting map name: {e}")
        
        return None
    
    def update_location_memory_from_llm(self, llm_response: str, map_name: str) -> None:
        """
        Extract and update location memory from LLM response.
        
        The LLM can update location memory using the format:
        LOCATION_MEMORY: <information about this location>
        
        Args:
            llm_response: LLM response text
            map_name: Current map name
        """
        try:
            for line in llm_response.split('\n'):
                if line.strip().upper().startswith('LOCATION_MEMORY:'):
                    memory_text = line.split(':', 1)[1].strip()
                    if memory_text:
                        self.memory_manager.set_location_memory(map_name, memory_text)
                        logger.info(f"Updated location memory for {map_name}: {memory_text[:50]}...")
                        # Auto-save memory after update (like MapStitcher)
                        self._auto_save_memory()
                        break
        except Exception as e:
            logger.warning(f"Error updating location memory from LLM: {e}")
    
    def _auto_save_memory(self) -> None:
        """Auto-save memory to cache file after updates (like MapStitcher behavior)"""
        try:
            self.memory_manager.save_to_file(self.memory_file)
            logger.debug(f"Auto-saved memory to {self.memory_file}")
        except Exception as e:
            logger.warning(f"Failed to auto-save memory: {e}")
    
    def get_location_memory_context(self, map_name: str) -> str:
        """
        Get location memory context for the current and nearby locations.
        
        Args:
            map_name: Current map name
            
        Returns:
            Formatted string with location memories
        """
        lines = []
        
        # Current location memory
        current_memory = self.memory_manager.get_location_memory(map_name)
        if current_memory:
            lines.append(f"📍 CURRENT LOCATION MEMORY ({map_name}):")
            lines.append(f"   {current_memory}")
        else:
            lines.append(f"📍 CURRENT LOCATION ({map_name}): No memory recorded yet")
        
        # Show a few other location memories for context
        all_memories = self.memory_manager.get_all_location_memories()
        other_memories = {k: v for k, v in all_memories.items() if k != map_name}
        
        if other_memories:
            lines.append("\n🗺️ OTHER LOCATION MEMORIES:")
            for loc_name, loc_info in list(other_memories.items())[:3]:
                preview = loc_info[:60] + "..." if len(loc_info) > 60 else loc_info
                lines.append(f"   - {loc_name}: {preview}")
            if len(other_memories) > 3:
                lines.append(f"   ... and {len(other_memories) - 3} more locations")
        
        return "\n".join(lines) if lines else "No location memories stored yet."
    
    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility method for client that expects agent.step(game_state)
        
        Args:
            game_state: Complete game state dictionary (should include 'frame')
            
        Returns:
            Dictionary with 'action' and optional 'reasoning'
        """
        frame = game_state.get('frame')
        if frame is None:
            logger.error("🚫 No frame in game_state for LocationSimpleAgent.step")
            return {"action": "WAIT", "reasoning": "No frame available"}
        
        action = self.process_step(frame, game_state)
        return {"action": action, "reasoning": "Location simple agent decision"}
    
    def process_step(self, frame, game_state: Dict[str, Any]) -> str:
        """
        Main processing step for location simple mode.
        
        Args:
            frame: Current game frame (PIL Image or similar)
            game_state: Complete game state dictionary
            
        Returns:
            Action string
        """
        # Validate frame
        if frame is None:
            logger.error("🚫 CRITICAL: LocationSimpleAgent.process_step called with None frame")
            return "WAIT"
        
        try:
            # Increment step counter
            self.state.step_counter += 1
            
            # Get current map name
            map_name = self.get_map_name(game_state)
            if map_name:
                self.state.current_map_name = map_name
            
            # Format the current state for LLM
            formatted_state = format_state_for_llm(game_state)
            
            # Get location memory context
            location_memory_context = ""
            if map_name:
                location_memory_context = self.get_location_memory_context(map_name)
            
            # Get recent history
            recent_history = self._format_recent_history()
            recent_actions_str = ', '.join(list(self.state.recent_actions)[-10:]) if self.state.recent_actions else 'None'
            
            # Build prompt
            prompt = f"""You are playing Pokemon Emerald. Make decisions based on the current game state and your location memories.

RECENT ACTIONS (last 10): {recent_actions_str}

RECENT HISTORY:
{recent_history}

{location_memory_context}

CURRENT GAME STATE:
{formatted_state}

Available actions: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT

Instructions:
1. Analyze the current situation and your location memories
2. Decide what action to take
3. Optionally update your memory about this location

Response format:
ANALYSIS:
[What do you see? What's your situation? What do you remember about this place?]

LOCATION_MEMORY: [Optional - Update memory about current location with useful information]

PLAN:
[What's your immediate goal and strategy?]

ACTION:
[Your chosen action - prefer single actions like 'RIGHT' or 'A']

Map name: {map_name} | Step: {self.state.step_counter}"""
            
            # Print prompt for debugging
            print("\n" + "="*100)
            print("🤖 LOCATION SIMPLE AGENT PROMPT:")
            print("="*100)
            sys.stdout.write(prompt)
            sys.stdout.write("\n")
            sys.stdout.flush()
            print("="*100 + "\n")
            
            # Make VLM call with game_state for Langfuse metadata
            response = self.vlm.get_query(frame, prompt, "location_simple_mode", game_state=game_state)
            print(f"🔍 VLM response: {response[:100]}..." if len(response) > 100 else f"🔍 VLM response: {response}")
            
            # Extract action from response
            action = self._parse_action(response)
            
            # Update location memory if provided in response
            if map_name:
                self.update_location_memory_from_llm(response, map_name)
            
            # Record this step in history
            history_entry = f"Map: {map_name}, Action: {action}, Step: {self.state.step_counter}"
            self.state.history.append(history_entry)
            
            # Update recent actions
            if isinstance(action, list):
                self.state.recent_actions.extend(action)
            else:
                self.state.recent_actions.append(action)
            
            return action
            
        except Exception as e:
            logger.error(f"Error in location simple agent processing: {e}")
            return "A"  # Default safe action
    
    def _format_recent_history(self) -> str:
        """Format recent history for LLM"""
        if not self.state.history:
            return "No previous history."
        
        recent = list(self.state.history)[-10:]  # Last 10 entries
        return "\n".join(f"{i+1}. {entry}" for i, entry in enumerate(recent))
    
    def _parse_action(self, response: str) -> str:
        """
        Parse action from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Action string
        """
        valid_actions = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
        
        # Look for ACTION: line
        for line in response.split('\n'):
            if line.strip().upper().startswith('ACTION:'):
                action_text = line.split(':', 1)[1].strip().upper()
                # Extract first valid action
                for token in action_text.replace(',', ' ').split():
                    if token in valid_actions:
                        return token
        
        # Fallback: search entire response
        response_upper = response.upper()
        for action in valid_actions:
            if action in response_upper:
                return action
        
        # Default
        return 'A'
    
    def save_memory(self, filepath: str) -> bool:
        """
        Save location memories to file.
        
        Args:
            filepath: Path to save memory file
            
        Returns:
            True if successful
        """
        return self.memory_manager.save_to_file(filepath)
    
    def load_memory(self, filepath: str) -> bool:
        """
        Load location memories from file.
        
        Args:
            filepath: Path to memory file
            
        Returns:
            True if successful
        """
        return self.memory_manager.load_from_file(filepath)


# Global instance for backward compatibility
_global_location_simple_agent = None


def get_location_simple_agent(vlm) -> LocationSimpleAgent:
    """Get or create the global location simple agent instance"""
    global _global_location_simple_agent
    if _global_location_simple_agent is None:
        _global_location_simple_agent = LocationSimpleAgent(vlm)
    elif _global_location_simple_agent.vlm != vlm:
        _global_location_simple_agent = LocationSimpleAgent(vlm)
    
    return _global_location_simple_agent
