"""
Agent 2: Trainer Agent Module

A training-focused agent that prioritizes Pokemon battles and team development.
Based on LocationSimpleAgent with emphasis on battles and training.

Key features:
- Prioritizes Pokemon battles and training
- Records Pokemon encounters and battle outcomes
- Focus on team development and leveling
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
class TrainerAgentState:
    """Maintains minimal state for the trainer agent"""
    history: deque = None
    recent_actions: deque = None
    step_counter: int = 0
    current_map_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize deques with default values"""
        if self.history is None:
            self.history = deque(maxlen=20)
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=30)


class TrainerAgent:
    """
    Trainer agent that focuses on Pokemon battles and team development.
    """
    
    def __init__(self, vlm):
        self.vlm = vlm
        self.state = TrainerAgentState()
        self.memory_manager = MemoryManager()
        
        self.memory_file = ".pokeagent_cache/memory_trainer.json"
        os.makedirs(".pokeagent_cache", exist_ok=True)
        
        if os.path.exists(self.memory_file):
            self.memory_manager.load_from_file(self.memory_file)
            logger.info(f"Loaded existing memory from {self.memory_file}")
        
        logger.info("Initialized TrainerAgent with memory management")
    
    def get_map_name(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Extract map name from game state."""
        try:
            player = game_state.get("player", {})
            location = player.get("location")
            if location:
                return location
            
            map_data = game_state.get("map", {})
            map_name = map_data.get("name")
            if map_name:
                return map_name
            
            map_id = map_data.get("id")
            if map_id is not None:
                return f"MAP_{map_id}"
                
        except Exception as e:
            logger.warning(f"Error getting map name: {e}")
        
        return None
    
    def update_location_memory_from_llm(self, llm_response: str, map_name: str) -> None:
        """Extract and update location memory from LLM response."""
        try:
            for line in llm_response.split('\n'):
                if line.strip().upper().startswith('LOCATION_MEMORY:'):
                    memory_text = line.split(':', 1)[1].strip()
                    if memory_text:
                        self.memory_manager.set_location_memory(map_name, memory_text)
                        logger.info(f"Updated location memory for {map_name}: {memory_text[:50]}...")
                        self._auto_save_memory()
                        break
        except Exception as e:
            logger.warning(f"Error updating location memory from LLM: {e}")
    
    def _auto_save_memory(self) -> None:
        """Auto-save memory to cache file after updates"""
        try:
            self.memory_manager.save_to_file(self.memory_file)
            logger.debug(f"Auto-saved memory to {self.memory_file}")
        except Exception as e:
            logger.warning(f"Failed to auto-save memory: {e}")
    
    def get_location_memory_context(self, map_name: str) -> str:
        """Get location memory context for the current and nearby locations."""
        lines = []
        
        current_memory = self.memory_manager.get_location_memory(map_name)
        if current_memory:
            lines.append(f"📍 CURRENT LOCATION MEMORY ({map_name}):")
            lines.append(f"   {current_memory}")
        else:
            lines.append(f"📍 CURRENT LOCATION ({map_name}): 💪 New training ground! Look for trainers and wild Pokemon!")
        
        all_memories = self.memory_manager.get_all_location_memories()
        other_memories = {k: v for k, v in all_memories.items() if k != map_name}
        
        if other_memories:
            lines.append(f"\n🗺️ TRAINING LOCATIONS ({len(other_memories)} trained):")
            for loc_name, loc_info in list(other_memories.items())[:3]:
                preview = loc_info[:60] + "..." if len(loc_info) > 60 else loc_info
                lines.append(f"   - {loc_name}: {preview}")
            if len(other_memories) > 3:
                lines.append(f"   ... and {len(other_memories) - 3} more training spots")
        
        return "\n".join(lines) if lines else "No training history yet - time to start battling!"
    
    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility method for client that expects agent.step(game_state)"""
        frame = game_state.get('frame')
        if frame is None:
            logger.error("🚫 No frame in game_state for TrainerAgent.step")
            return {"action": "WAIT", "reasoning": "No frame available"}
        
        action = self.process_step(frame, game_state)
        return {"action": action, "reasoning": "Trainer agent decision"}
    
    def process_step(self, frame, game_state: Dict[str, Any]) -> str:
        """Main processing step for trainer mode."""
        if frame is None:
            logger.error("🚫 CRITICAL: TrainerAgent.process_step called with None frame")
            return "WAIT"
        
        try:
            self.state.step_counter += 1
            
            map_name = self.get_map_name(game_state)
            if map_name:
                self.state.current_map_name = map_name
            
            formatted_state = format_state_for_llm(game_state)
            
            location_memory_context = ""
            if map_name:
                location_memory_context = self.get_location_memory_context(map_name)
            
            recent_history = self._format_recent_history()
            recent_actions_str = ', '.join(list(self.state.recent_actions)[-10:]) if self.state.recent_actions else 'None'
            
            # Trainer-specific prompt
            prompt = f"""You are a POKEMON TRAINER playing Pokemon Emerald. Your mission: BUILD THE STRONGEST TEAM!

🎯 TRAINER OBJECTIVES:
- Seek out and engage in Pokemon battles (wild Pokemon and trainers)
- Train your team to become stronger
- Record battle outcomes and Pokemon encounters
- Learn which Pokemon and trainers are in each location
- Build a balanced and powerful team

RECENT ACTIONS (last 10): {recent_actions_str}

RECENT HISTORY:
{recent_history}

{location_memory_context}

CURRENT GAME STATE:
{formatted_state}

Available actions: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT

Instructions:
1. Look for battles - engage wild Pokemon and trainers
2. Document Pokemon types, levels, and battle results
3. Focus on leveling up your team

Response format:
ANALYSIS:
[What Pokemon are here? Any trainers? What's your team status?]

LOCATION_MEMORY: [Record Pokemon encounters, trainer locations, battle outcomes, good training spots]

PLAN:
[What's your training strategy? Which battles to seek?]

ACTION:
[Your chosen action - engage battles when possible]

Map: {map_name} | Battles: Check your team | Step: {self.state.step_counter}"""
            
            print("\n" + "="*100)
            print("💪 TRAINER AGENT PROMPT:")
            print("="*100)
            sys.stdout.write(prompt)
            sys.stdout.write("\n")
            sys.stdout.flush()
            print("="*100 + "\n")
            
            # Pass game_state to VLM for Langfuse metadata
            response = self.vlm.get_query(frame, prompt, "trainer_mode", game_state=game_state)
            print(f"🔍 VLM response: {response[:100]}..." if len(response) > 100 else f"🔍 VLM response: {response}")
            
            action = self._parse_action(response)
            
            if map_name:
                self.update_location_memory_from_llm(response, map_name)
            
            history_entry = f"Map: {map_name}, Action: {action}, Step: {self.state.step_counter}"
            self.state.history.append(history_entry)
            
            if isinstance(action, list):
                self.state.recent_actions.extend(action)
            else:
                self.state.recent_actions.append(action)
            
            return action
            
        except Exception as e:
            logger.error(f"Error in trainer agent processing: {e}")
            return "A"
    
    def _format_recent_history(self) -> str:
        """Format recent history for LLM"""
        if not self.state.history:
            return "No previous history."
        
        recent = list(self.state.history)[-10:]
        return "\n".join(f"{i+1}. {entry}" for i, entry in enumerate(recent))
    
    def _parse_action(self, response: str) -> str:
        """Parse action from LLM response."""
        valid_actions = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
        
        for line in response.split('\n'):
            if line.strip().upper().startswith('ACTION:'):
                action_text = line.split(':', 1)[1].strip().upper()
                for token in action_text.replace(',', ' ').split():
                    if token in valid_actions:
                        return token
        
        response_upper = response.upper()
        for action in valid_actions:
            if action in response_upper:
                return action
        
        return 'A'
    
    def save_memory(self, filepath: str) -> bool:
        """Save location memories to file."""
        return self.memory_manager.save_to_file(filepath)
    
    def load_memory(self, filepath: str) -> bool:
        """Load location memories from file."""
        return self.memory_manager.load_from_file(filepath)


_global_trainer_agent = None


def get_trainer_agent(vlm):
    """Get or create the global trainer agent instance"""
    global _global_trainer_agent
    if _global_trainer_agent is None:
        _global_trainer_agent = TrainerAgent(vlm)
    elif _global_trainer_agent.vlm != vlm:
        _global_trainer_agent = TrainerAgent(vlm)
    
    return _global_trainer_agent
