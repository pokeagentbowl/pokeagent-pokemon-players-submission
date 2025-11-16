"""
Overall Agent with File-Scaffolded Planning

This agent implements the high-level planning logic from agent_sphealsmart_v2.py:
1. Loads objectives from startup_objectives.json
2. Uses entity_database.json for entity-based pathfinding
3. Detects dialogue and player state
4. Delegates navigation to NavigationAgentNT
5. Handles menu/battle situations with VLM fallback

Main differences from navigation_agent_nt:
- Focuses on objective planning and state management
- Uses external JSON files for objectives and entity positions
- Handles dialogue interactions and state transitions
- Delegates low-level navigation to navigation_agent_nt
"""
from typing import List, Literal, Optional, Dict, Tuple
from pydantic import BaseModel, Field
import logging
import numpy as np
import os
import re
import json
from pathlib import Path
from PIL import Image

from custom_agent.base_agent import AgentRegistry
from custom_agent.base_langchain_agent import BaseLangChainAgent
from custom_agent.navigation_agent_nt import NavigationAgentNT
from custom_utils.detectors import detect_dialogue, detect_player_direction, detect_player_visible, detect_battle
from custom_utils.map_extractor import get_player_centered_grid
from custom_utils.label_traversable import compute_simple_tile_features
from custom_utils.log_to_active import (
    load_grass_cache, save_grass_cache, load_actions_log, 
    save_actions_log, log_action, update_grass_cache,
    log_interaction, mark_and_save_tile
)

logger = logging.getLogger(__name__)


def _clean_state_summary(state_summary: str) -> str:
    """
    Post-process format_state_summary output to remove 'State:' and 'Dialog:' prefixes.
    
    Removes patterns like:
    - "State: dialog |" -> ""
    - "Dialog: The clock is stopped... |" -> ""
    
    Args:
        state_summary: Raw output from format_state_summary
        
    Returns:
        Cleaned state summary with prefixes removed
    """
    import re
    # Remove "State: <value> |" pattern
    cleaned = re.sub(r'State:\s*[^|]*\|\s*', '', state_summary)
    # Remove "Dialog: <text> |" pattern completely
    cleaned = re.sub(r'Dialog:\s*[^|]*\|\s*', '', cleaned)
    return cleaned


def _compute_f1_overlap(text1: str, text2: str) -> float:
    """
    Compute F1 score based on word overlap between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    precision = len(intersection) / len(words1)
    recall = len(intersection) / len(words2)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# Constants
OBJECTIVES_FILE = "./startup_cache/startup_objectives.json"
ENTITY_DATABASE_FILE = "./startup_cache/entity_database.json"
DIALOGUE_LOG_FILE = "./startup_cache/dialogues.json"
GRASS_SELECTION_THRESHOLD = 0.5  # Threshold for visitation-weighted distance score
NAVIGATION_CACHE_DIR = "./navigation_caches"  # Directory for navigation cache (shared with navigation_agent_nt)


# VLM Structured Output Models
class ObjectiveReview(BaseModel):
    """VLM structured output for objective completion review."""
    analysis: str = Field(description="Analysis of objective completion status")
    is_completed: bool = Field(description="Whether the objective is completed")


class MenuBattleDecision(BaseModel):
    """VLM structured output for menu/battle navigation."""
    reasoning: str = Field(description="Reasoning for action choice")
    actions: List[str] = Field(description="List of button presses")
    current_objective_completed: bool = Field(description="Whether current objective is completed")


class DialogueSpeakerVerification(BaseModel):
    """VLM structured output for verifying dialogue speaker identity."""
    analysis: str = Field(description="Analysis of the dialogue content and speaker")
    is_mom: bool = Field(description="Whether the speaker is Mom (the player's mother)")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class DialogueTextExtraction(BaseModel):
    """VLM structured output for extracting dialogue text from game frame."""
    dialogue_text: str = Field(description="The extracted dialogue text, or 'NO_DIALOGUE' if none visible")


class MenuOption(BaseModel):
    """Single menu option with its selection sequence."""
    option: str = Field(description="The menu option text")
    sequence: List[str] = Field(description="List of button presses needed to select this option")


class ScreenTextExtraction(BaseModel):
    """VLM structured output for extracting all text from game screen."""
    dialogue_text: str = Field(description="The extracted dialogue text from bottom of screen, or 'NO_DIALOGUE' if none visible")
    menu_options: List[MenuOption] = Field(description="List of menu options with their selection sequences")
    current_selection: Optional[str] = Field(description="The currently selected option indicated by > arrow, or None if no selection visible")


class ObjectiveItem(BaseModel):
    """Single objective from startup_objectives.json"""
    name: str
    description: str
    type: str  # "menu_navigation", "world_navigation", "level_grinding"
    target_entity: Optional[str] = None
    target_level: Optional[int] = None
    status: str = "[ ]"  # "[ ]", "[x]", or "[~]"


class OverallAgentState(BaseModel):
    """Track high-level agent state across steps"""
    current_objective_index: int = 0
    objectives: List[ObjectiveItem] = []
    last_dialogue_state: bool = False
    dialogue_count: int = 0
    level_grinding_active: bool = False
    player_visible: bool = True
    player_facing: Optional[str] = None
    current_step_num: int = 0  # Global step counter for grass visitation tracking
    reasoning_history: List[str] = []  # Track last 15 reasoning messages
    
    # Dialogue loop detection
    dialogue_history: List[Dict[str, any]] = []  # Track last 10 dialogues with {step, text, actions}
    action_history: List[Dict[str, any]] = []  # Track last 10 actions with {step, actions, reasoning}
    
    # Fallback tracking
    last_fallback_step: Optional[int] = None
    last_repeated_dialogue: Optional[str] = None
    last_fallback_player_pos: Optional[Tuple[int, int]] = None
    
    # Local coordinate navigation tracking
    navigation_start_pos: Optional[Tuple[int, int]] = None  # Starting position for "L" coordinate tracking
    navigation_start_map: Optional[str] = None  # Starting map name
    navigation_target_delta: Optional[Tuple[int, int]] = None  # Target (dx, dy) for "L" coordinates
    total_moved: Tuple[int, int] = (0, 0)  # Total (dx, dy) moved since navigation started
    last_player_pos: Optional[Tuple[int, int]] = None  # Last known player position
    last_player_map: Optional[str] = None  # Last known player map
    
    # Action loop prevention
    skip_action_loop_check_steps: int = 0  # Skip action loop check for next N steps after fallback
    
    # Battle mode tracking
    battle_mode_active: bool = False  # Whether we're currently in battle fallback mode
    
    # Last extracted dialogue text tracking
    last_dialogue_text: Optional[str] = None  # Last dialogue text to prevent leaking between sessions
    last_screen_text: Optional[ScreenTextExtraction] = None  # Last screen text extraction for fallback use
    
    class Config:
        arbitrary_types_allowed = True


@AgentRegistry.register("overall-nt")
class OverallAgentNT(BaseLangChainAgent):
    """Overall agent with file-scaffolded planning"""
    
    def __init__(
        self,
        backend: str = "github_models",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        objectives_file: str = OBJECTIVES_FILE,
        entity_database_file: str = ENTITY_DATABASE_FILE,
        **kwargs
    ):
        """
        Initialize overall agent.
        
        Args:
            backend: LLM backend type
            model_name: Model name
            temperature: Generation temperature
            objectives_file: Path to objectives JSON file
            entity_database_file: Path to entity database JSON file
            **kwargs: Additional args passed to BaseLangChainAgent
        """
        super().__init__(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        
        # Store original model name for dynamic switching
        self.original_model_name = model_name
        
        # Initialize sub-agent for navigation
        self.nav_agent = NavigationAgentNT(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        
        # Load configuration files
        self.objectives_file = objectives_file
        self.entity_database_file = entity_database_file
        
        # Initialize state
        self.state = OverallAgentState()
        self._load_objectives()
        self._load_entity_database()
        
        # Load grass cache for level grinding
        self.grass_cache = load_grass_cache()
        
        # Load actions log
        self.actions_log = load_actions_log()
        
        logger.info(f"Initialized OverallAgentNT with {len(self.state.objectives)} objectives")
    
   
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> Dict:
        """
        Main decision loop with objective-based planning.
        
        Implementation of Steps 1, 2, 4, 8 from transfer_agent.py:
        - Step 1: Dialogue Detection
        - Step 2: Player Detection
        - Step 4: Handle Dialogue State Changes
        - Step 8: Get Next Objective
        
        Args:
            state_data: Game state data from emulator
            frame: Game frame as numpy array
            **kwargs: Additional arguments
            
        Returns:
            Dict containing 'action' (List[str]) and optionally other data
        """
        # Increment step counter
        self.state.current_step_num += 1
        
        # Dynamic model switching: use gpt-4o-mini until "Exit moving van" objective is completed
        if self.state.current_objective_index <= 1:
            self.model_name = "gpt-4o-mini"
            self.nav_agent.model_name = "gpt-4o-mini"
        else:
            self.model_name = self.original_model_name
            self.nav_agent.model_name = self.original_model_name
        
        # # Save debug frame
        # try:
        #     debug_dir = Path("./debugframes")
        #     debug_dir.mkdir(parents=True, exist_ok=True)
            
        #     # Convert frame to PIL Image if needed
        #     if isinstance(frame, np.ndarray):
        #         debug_frame = Image.fromarray(frame)
        #     else:
        #         debug_frame = frame
            
        #     debug_path = debug_dir / f"{self.state.current_step_num}.png"
        #     debug_frame.save(debug_path)
        #     logger.debug(f"Saved debug frame: {debug_path}")
        # except Exception as e:
        #     logger.warning(f"Failed to save debug frame: {e}")
        
        # Step 1: Screen Text Detection (replaces dialogue detection)
        current_dialogue = detect_dialogue(frame, threshold=0.3)
        # Only extract dialogue text if dialogue is detected to avoid spamming extractions
        if current_dialogue:
            screen_text = self._extract_dialogue_text(frame)
        else:
            screen_text = "NO_DIALOGUE"
        # current_dialogue = screen_text.dialogue_text != "NO_DIALOGUE"
        
        # Check if we just used fallback on the previous step
        if (self.state.last_fallback_step is not None and 
            self.state.last_fallback_step == self.state.current_step_num - 1):
            # If no current dialogue, break out of fallback mode
            if not current_dialogue:
                logger.info("Next turn after fallback: no current dialogue, breaking out of fallback mode")
                # Clear fallback tracking
                self.state.last_fallback_step = None
                self.state.last_repeated_dialogue = None
                self.state.last_fallback_player_pos = None
            else:
                # Dialogue is still active, continue with fallback mode
                logger.info("Next turn after fallback: dialogue still active, continuing with fallback mode")

        # Check if 5 turns have passed since fallback, clear tracking
        if (self.state.last_fallback_step is not None and 
            self.state.current_step_num - self.state.last_fallback_step >= 5):
            logger.info("5 turns after fallback: clearing fallback tracking")
            self.state.last_fallback_step = None
            self.state.last_repeated_dialogue = None
            self.state.last_fallback_player_pos = None

        # Check for dialogue loop BEFORE proceeding
        repeated_dialogue = self._check_dialogue_loop()
        if repeated_dialogue:
            # Get current objective
            current_objective = self._get_current_objective()
            if current_objective:
                # Use strong model fallback to break the loop
                result = self._fallback_vlm_strong_model(
                    state_data, frame, repeated_dialogue, current_objective
                )
                
                # Track fallback usage
                self.state.last_fallback_step = self.state.current_step_num
                self.state.last_repeated_dialogue = repeated_dialogue
                player_data = state_data.get('player', {})
                position = player_data.get('position', {})
                self.state.last_fallback_player_pos = (position.get('x', 0), position.get('y', 0))
                
                # After fallback, clear the dialogue history to prevent retriggering
                # This resets the cache so no previous dialogues can cause loop detection
                self.state.dialogue_history.clear()
                
                # Track action in history
                if len(self.state.action_history) >= 10:
                    self.state.action_history.pop(0)
                frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
                frame_resized = frame_pil.resize((32, 32), Image.NEAREST)
                self.state.action_history.append({
                    'step': self.state.current_step_num,
                    'actions': result.get('action', []),
                    'reasoning': result.get('reasoning', 'Fallback VLM - breaking dialogue loop'),
                    'frame_features': compute_simple_tile_features(frame_resized)
                })
                
                return result

        # Step 2: Player Detection (call detect_player_direction once with threshold 0.7)
        player_direction_result = detect_player_direction(frame, match_threshold=0.7)
        player_visible = player_direction_result is not None
        player_facing = player_direction_result[0] if player_direction_result else None
        
        # Battle Detection: Check if we're in battle
        battle_detected = detect_battle(frame)
        
        # Update battle mode state
        if battle_detected and not player_visible:
            # Enter battle mode if battle detected and player not visible
            if not self.state.battle_mode_active:
                logger.info("Battle detected - activating battle fallback mode")
                self.state.battle_mode_active = True
        elif self.state.battle_mode_active and player_visible:
            # Exit battle mode if player becomes visible again
            logger.info("Player visible - deactivating battle fallback mode")
            self.state.battle_mode_active = False
        
        # Update state
        self.state.player_visible = player_visible
        self.state.player_facing = player_facing
        
        # Update last known player position when visible
        if player_visible:
            player_data = state_data.get('player', {})
            position = player_data.get('position', {})
            player_x = position.get('x', 0)
            player_y = position.get('y', 0)
            player_map = player_data.get('location', 'unknown')
            self.state.last_player_pos = (player_x, player_y)
            self.state.last_player_map = player_map
        
        # Step 4: Handle Dialogue State Changes
        dialogue_state_changed = current_dialogue != self.state.last_dialogue_state
        
        if dialogue_state_changed:
            if current_dialogue:
                # Dialogue started
                logger.info("Dialogue started")
                dialogue_text = screen_text
                self._log_dialogue_session(state_data, is_start=True, dialogue_text=dialogue_text)
                
                # Extract and store screen text for potential fallback use
                self.state.last_screen_text = "NO_DIALOGUE" #self._extract_screen_text(frame)
                
                # Check for "Fix clock" objective completion
                current_objective = self._get_current_objective()
                if (current_objective and 
                    current_objective.name == "Fix clock" and 
                    dialogue_text and 
                    self._check_fix_clock_completion(dialogue_text, self.state.current_step_num)):
                    logger.info("Fix clock objective completed based on dialogue sequence")
                    self._mark_objective_complete(self.state.current_objective_index)
                
                # Mark tile as NPC if player is visible AND we have a valid facing direction from current detection
                # Use player_visible (not self.state.player_visible) and player_facing (not self.state.player_facing)
                # to ensure we only mark when detection succeeded on THIS step
                if player_visible and player_facing:
                    logger.debug(f"Marking NPC tile: player_visible={player_visible}, player_facing={player_facing}")
                    self._mark_npc_tile_on_dialogue_start(state_data, frame, player_facing)
                else:
                    logger.debug(f"Skipping NPC marking: player_visible={player_visible}, player_facing={player_facing}")
            else:
                # Dialogue ended
                logger.info("Dialogue ended")
                self._log_dialogue_session(state_data, is_start=False, frame=frame)
                
                # Add completed dialogue to history using last extracted text
                if self.state.last_dialogue_text and self.state.last_dialogue_text not in ["NO_DIALOGUE", "EXTRACTION_FAILED", "DIALOGUE_TEXT_PLACEHOLDER"]:
                    if len(self.state.dialogue_history) >= 10:
                        self.state.dialogue_history.pop(0)
                    self.state.dialogue_history.append({
                        'step': self.state.current_step_num,
                        'text': self.state.last_dialogue_text
                    })

        self.state.last_dialogue_state = current_dialogue
        
        # Step 8: Get Next Objective
        current_objective = self._get_current_objective()
        
        """
        # Check for action and dialogue loop
        if self.state.skip_action_loop_check_steps > 0:
            self.state.skip_action_loop_check_steps -= 1
            repeated_reasoning = None
        else:
            repeated_reasoning = self._check_action_loop()
        if repeated_reasoning:
            logger.warning(f"Action loop detected with reasoning: {repeated_reasoning[:50]}...")
            # Use strong model fallback to break the loop
            result = self._fallback_vlm_strong_model(
                state_data, frame, repeated_reasoning, current_objective or ObjectiveItem(name="Unknown", description="Unknown", type="unknown")
            )
            
            # Track action in history
            if len(self.state.action_history) >= 10:
                self.state.action_history.pop(0)
            frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
            frame_resized = frame_pil.resize((32, 32), Image.NEAREST)
            self.state.action_history.append({
                'step': self.state.current_step_num,
                'actions': result.get('action', []),
                'reasoning': 'Fallback VLM - breaking action loop',
                'frame_features': compute_simple_tile_features(frame_resized)
            })
            
            # Log to actions.json
            log_action(
                self.actions_log,
                self.state.current_step_num,
                'Fallback VLM - breaking action loop',
                result.get('action', [])
            )
            
            # Skip action loop check for next 2 steps to allow fallback to take effect
            self.state.skip_action_loop_check_steps = 2
            
            return result
        """
        
        if not current_objective:
            logger.info("All objectives complete!")
            return {'action': []}
        
        logger.info(f"Current objective: {current_objective.name} ({current_objective.type})")
        
        # Manual override: Check for "Choose name" objective completion
        if current_objective.name == "Choose name" and player_visible:
            logger.info("Manual override: 'Choose name' objective complete (player visible)")
            self._mark_objective_complete(self.state.current_objective_index)
            current_objective = self._get_current_objective()
            if not current_objective:
                logger.info("All objectives complete after manual override!")
                return {'action': []}
        
        # Manual override: Check for "Exit truck" objective completion
        # Complete when Mom dialogue has ended (dialogue_count >= 1 and dialogue is not active)
        if current_objective.name == "Exit truck" and self.state.dialogue_count >= 1 and not current_dialogue and player_map == "LITTLEROOT TOWN":
            logger.info("Manual override: 'Exit truck' objective complete (Mom dialogue ended)")
            self._mark_objective_complete(self.state.current_objective_index)
            current_objective = self._get_current_objective()
            if not current_objective:
                logger.info("All objectives complete after manual override!")
                return {'action': []}
        
        # Every 5 steps: Check objective completion with VLM review
        # Skip VLM review for objectives with manual overrides to prevent premature completion
        skip_vlm_objectives = ["Choose name", "Exit truck"]
        if (self.state.current_step_num % 5 == 0 and 
            self.state.current_step_num > 0 and 
            current_objective.name not in skip_vlm_objectives):
            logger.info(f"Step {self.state.current_step_num}: Performing 5-step objective review")
            is_completed = self._check_objective_completion_with_vlm(state_data, frame, current_objective)
            if is_completed:
                logger.info(f"5-step review indicates objective completed: {current_objective.name}")
                self._mark_objective_complete(self.state.current_objective_index)
                # Get next objective after marking complete
                current_objective = self._get_current_objective()
                if not current_objective:
                    logger.info("All objectives complete after 5-step review!")
                    return {'action': []}
        
        # Check level grinding completion
        if current_objective.type == "level_grinding" and self._check_level_up_completion(state_data):
            self._mark_objective_complete(self.state.current_objective_index)
            return {'action': ['B']}  # Exit battle/menu
        
        # Decision Point 3: Dialogue Active
        if current_dialogue:
            # Use dialogue text from screen extraction for loop detection (only when dialogue is active)
            dialogue_text = screen_text
            
            # Track dialogue in history only if there's actual content
            normalized_text = dialogue_text.strip().upper()
            if normalized_text and normalized_text not in ["NO_DIALOGUE", "NO DIALOGUE", ""]:
                if len(self.state.dialogue_history) >= 10:
                    self.state.dialogue_history.pop(0)
                self.state.dialogue_history.append({
                    'step': self.state.current_step_num,
                    'text': dialogue_text
                })
                logger.info(f"Tracked dialogue at step {self.state.current_step_num}: '{dialogue_text}'")
            
            logger.info("Dialogue active - pressing A to continue")
            result = {'action': ['A']}
            
            # Track action in history
            if len(self.state.action_history) >= 10:
                self.state.action_history.pop(0)
            frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
            frame_resized = frame_pil.resize((32, 32), Image.NEAREST)
            self.state.action_history.append({
                'step': self.state.current_step_num,
                'actions': result.get('action', []),
                'reasoning': f"Dialogue active - continuing dialogue for objective: {current_objective.name if current_objective else 'None'}",
                'frame_features': compute_simple_tile_features(frame_resized)
            })
            
            # Log to actions.json
            log_action(
                self.actions_log,
                self.state.current_step_num,
                f"Dialogue active - continuing dialogue for objective: {current_objective.name if current_objective else 'None'}",
                result.get('action', [])
            )
            
            return result
        
        # Decision Point 1: Player not visible (menu/battle)
        if not player_visible:
            if self.state.battle_mode_active:
                logger.info("Player not visible and battle mode active - using battle-specific VLM fallback")
                result = self._fallback_vlm_strong_model(
                    state_data, frame, "Battle state detected - player not visible", current_objective, loop_type="battle"
                )
            else:
                logger.info("Player not visible - using VLM for menu/battle navigation")
                result = self._handle_menu_battle_with_vlm(state_data, frame, current_objective)
            
            # Clear stored screen text after use
            self.state.last_screen_text = None
            
            # Track action in history
            if len(self.state.action_history) >= 10:
                self.state.action_history.pop(0)
            frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
            frame_resized = frame_pil.resize((32, 32), Image.NEAREST)
            self.state.action_history.append({
                'step': self.state.current_step_num,
                'actions': result.get('action', []),
                'reasoning': f"Menu/Battle VLM - objective: {current_objective.name if current_objective else 'None'}",
                'frame_features': compute_simple_tile_features(frame_resized)
            })
            
            # Log to actions.json
            log_action(
                self.actions_log,
                self.state.current_step_num,
                f"Menu/Battle VLM - objective: {current_objective.name if current_objective else 'None'}",
                result.get('action', [])
            )
            
            return result
        
        # Decision Point 4: Player visible
        logger.debug(f"Player visible facing {player_facing}")
        
        # Check if level grinding objective - use special grass targeting
        if current_objective.type == "level_grinding":
            logger.info("Level grinding active - targeting grass tiles")
            kwargs['player_direction_result'] = player_direction_result
            result = self._handle_level_grinding(state_data, frame, **kwargs)
            
            # Track action in history
            if len(self.state.action_history) >= 10:
                self.state.action_history.pop(0)
            frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
            frame_resized = frame_pil.resize((32, 32), Image.NEAREST)
            self.state.action_history.append({
                'step': self.state.current_step_num,
                'actions': result.get('action', []),
                'reasoning': f"Level grinding - objective: {current_objective.name if current_objective else 'None'}",
                'frame_features': compute_simple_tile_features(frame_resized)
            })
            
            # Log to actions.json
            log_action(
                self.actions_log,
                self.state.current_step_num,
                f"Level grinding - objective: {current_objective.name if current_objective else 'None'}",
                result.get('action', [])
            )
            
            return result
        
        # Check if world_navigation with target_entity - resolve entity coordinates
        if current_objective.type == "world_navigation" and current_objective.target_entity:
            # Get player position
            player_data = state_data.get('player', {})
            position = player_data.get('position', {})
            player_x = position.get('x', 0)
            player_y = position.get('y', 0)
            player_map = player_data.get('location', 'unknown')
            
            # Update local navigation tracking if active
            if self.state.navigation_target_delta is not None:
                self._update_local_navigation_tracking(player_x, player_y, player_map)
                
                # Check if local navigation is complete
                if self._check_local_navigation_complete():
                    logger.info(f"Local navigation objective '{current_objective.name}' complete!")
                    self._stop_local_navigation_tracking()
                    self._mark_objective_complete(self.state.current_objective_index)
                    return {'action': []}
            
            # Check if this is a local coordinate entity
            entity_name = current_objective.target_entity
            if entity_name in self.entity_database:
                entity_data = self.entity_database[entity_name]
                if entity_data and len(entity_data) > 0 and entity_data[0][0] == "L":
                    # Local coordinate navigation
                    target_dx = entity_data[0][1]
                    target_dy = entity_data[0][2]
                    
                    # Start tracking if not already started
                    if self.state.navigation_target_delta is None:
                        self._start_local_navigation_tracking(
                            (target_dx, target_dy),
                            player_x,
                            player_y,
                            player_map
                        )
                    
                    # Calculate current target position based on start + target delta
                    # This gives navigation agent a moving target as we track progress
                    start_x, start_y = self.state.navigation_start_pos
                    target_x = start_x + target_dx
                    target_y = start_y + target_dy
                    
                    logger.info(f"Local coordinate navigation: target ({target_x}, {target_y}), moved {self.state.total_moved}")
                    
                    # Pass target coordinates to navigation agent via kwargs
                    entity_kwargs = kwargs.copy()
                    entity_kwargs['target_x'] = target_x
                    entity_kwargs['target_y'] = target_y
                    if current_objective:
                        objective_text = f"{current_objective.name}: {current_objective.description}"
                        if current_objective.target_entity:
                            objective_text += f" (Target: {current_objective.target_entity})"
                        entity_kwargs['objective_context'] = objective_text
                    
                    result = self.nav_agent.choose_action(state_data, frame, **entity_kwargs)
                    return result
            
            # Map-based coordinate entity (format: ["MAP_NAME", x, y])
            entity_coords = self._get_entity_coordinates(
                current_objective.target_entity,
                player_x,
                player_y,
                player_map
            )
            
            if entity_coords is not None:
                target_x, target_y, target_map = entity_coords
                
                # Check if we're on the target map
                if player_map == target_map:
                    logger.info(f"On target map '{target_map}' - using entity '{current_objective.target_entity}' coordinates: ({target_x}, {target_y})")
                    
                    # Pass target coordinates and player detection to navigation agent via kwargs
                    entity_kwargs = kwargs.copy()
                    entity_kwargs['target_x'] = target_x
                    entity_kwargs['target_y'] = target_y
                    entity_kwargs['player_direction_result'] = player_direction_result
                    if current_objective:
                        objective_text = f"{current_objective.name}: {current_objective.description}"
                        if current_objective.target_entity:
                            objective_text += f" (Target: {current_objective.target_entity})"
                        entity_kwargs['objective_context'] = objective_text
                    
                    result = self.nav_agent.choose_action(state_data, frame, **entity_kwargs)
                    return result
                else:
                    # Not on target map yet - navigate normally to find the map
                    logger.info(f"Not on target map yet (current: {player_map}, target: {target_map}) - using normal navigation")
            else:
                logger.warning(f"Could not resolve entity '{current_objective.target_entity}' coordinates, using normal navigation")
        
        # Normal navigation - delegate to navigation agent
        logger.debug("Delegating to navigation agent for normal navigation")
        
        # Pass objective context and player detection to navigation agent for better target selection
        nav_kwargs = kwargs.copy()
        if current_objective:
            objective_text = f"{current_objective.name}: {current_objective.description}"
            if current_objective.target_entity:
                objective_text += f" (Target: {current_objective.target_entity})"
            nav_kwargs['objective_context'] = objective_text
        nav_kwargs['player_direction_result'] = player_direction_result
        
        result = self.nav_agent.choose_action(state_data, frame, **nav_kwargs)
        
        # Track action in history
        if len(self.state.action_history) >= 10:
            self.state.action_history.pop(0)
        frame_pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        frame_resized = frame_pil.resize((32, 32), Image.NEAREST)
        self.state.action_history.append({
            'step': self.state.current_step_num,
            'actions': result.get('action', []) if isinstance(result, dict) else result,
            'reasoning': f"Navigation agent - objective: {current_objective.name if current_objective else 'None'}",
            'frame_features': compute_simple_tile_features(frame_resized)
        })
        
        # Log to actions.json
        log_action(
            self.actions_log,
            self.state.current_step_num,
            f"Navigation agent - objective: {current_objective.name if current_objective else 'None'}",
            result.get('action', []) if isinstance(result, dict) else result
        )
        
        return result if isinstance(result, dict) else {'action': result}

    def _load_objectives(self) -> None:
        """Load objectives from JSON file"""
        objectives_path = Path(self.objectives_file)
        
        if not objectives_path.exists():
            logger.warning(f"Objectives file not found: {objectives_path}")
            self.state.objectives = []
            return
        
        try:
            with open(objectives_path, 'r') as f:
                objectives_data = json.load(f)
            
            self.state.objectives = [
                ObjectiveItem(**obj) for obj in objectives_data
            ]
            
            logger.info(f"Loaded {len(self.state.objectives)} objectives from {objectives_path}")
            
        except Exception as e:
            logger.error(f"Failed to load objectives: {e}")
            self.state.objectives = []
    
    def _save_objectives(self) -> None:
        """Save objectives back to JSON file"""
        try:
            objectives_data = [obj.dict() for obj in self.state.objectives]
            
            with open(self.objectives_file, 'w') as f:
                json.dump(objectives_data, f, indent=2)
            
            logger.debug(f"Saved objectives to {self.objectives_file}")
            
        except Exception as e:
            logger.error(f"Failed to save objectives: {e}")
    
    def _load_entity_database(self) -> None:
        """Load entity database from JSON file"""
        database_path = Path(self.entity_database_file)
        
        if not database_path.exists():
            logger.warning(f"Entity database not found: {database_path}")
            self.entity_database = {}
            return
        
        try:
            with open(database_path, 'r') as f:
                self.entity_database = json.load(f)
            
            logger.info(f"Loaded entity database with {len(self.entity_database)} entity types")
            
        except Exception as e:
            logger.error(f"Failed to load entity database: {e}")
            self.entity_database = {}
    
    def _get_entity_coordinates(
        self, 
        entity_name: str, 
        player_x: int, 
        player_y: int,
        player_map: str
    ) -> Optional[Tuple[int, int, str]]:
        """
        Get entity coordinates from entity database.
        
        Supports:
        - "L" (Local): coordinates relative to player (dx, dy) - handled separately in choose_action
        - Map name + position: ["MAP_NAME", x, y] for exact map coordinates
        
        Args:
            entity_name: Name of entity in database (e.g., "exit_truck", "mom_initial")
            player_x: Current player X position
            player_y: Current player Y position
            player_map: Current map name
            
        Returns:
            (target_x, target_y, target_map) tuple, or None if entity not found
        """
        if entity_name not in self.entity_database:
            logger.warning(f"Entity '{entity_name}' not found in database")
            return None
        
        entity_data = self.entity_database[entity_name]
        if not entity_data or len(entity_data) == 0:
            logger.warning(f"Entity '{entity_name}' has no coordinate data")
            return None
        
        # Format: [["L", dx, dy]] or [["MAP_NAME", x, y]]
        coord_type = entity_data[0][0]
        coord_1 = entity_data[0][1]
        coord_2 = entity_data[0][2]
        
        if coord_type == "L":
            # Local coordinates relative to player - should be handled in choose_action
            # Return None to indicate this needs special handling
            logger.debug(f"Entity '{entity_name}' uses local coordinates - handled separately")
            return None
        
        else:
            # Assume coord_type is a map name
            # Format: ["MAP_NAME", x, y]
            target_map = coord_type
            target_x = coord_1
            target_y = coord_2
            logger.info(f"Entity '{entity_name}' at map '{target_map}' coords ({target_x}, {target_y})")
            return (target_x, target_y, target_map)
    
    def _start_local_navigation_tracking(
        self,
        target_delta: Tuple[int, int],
        player_x: int,
        player_y: int,
        player_map: str
    ) -> None:
        """
        Start tracking navigation for local ("L") coordinates.
        
        Args:
            target_delta: Target (dx, dy) to move
            player_x: Starting player X position
            player_y: Starting player Y position
            player_map: Starting map name
        """
        self.state.navigation_start_pos = (player_x, player_y)
        self.state.navigation_start_map = player_map
        self.state.navigation_target_delta = target_delta
        self.state.total_moved = (0, 0)
        self.state.last_player_pos = (player_x, player_y)
        self.state.last_player_map = player_map
        
        logger.info(f"Started local navigation tracking: target delta {target_delta} from ({player_x}, {player_y}) on {player_map}")
    
    def _update_local_navigation_tracking(
        self,
        player_x: int,
        player_y: int,
        player_map: str
    ) -> None:
        """
        Update navigation tracking based on player movement.
        
        Handles:
        - Normal movement within same map
        - Map transitions (preserves relative coordinate offset)
        
        When map changes, the relative coordinate (target_delta - total_moved) is preserved
        by updating the navigation_start_pos to maintain the same relative offset.
        
        Example: routing to littleroot (3, -5), after reaching littleroot (3, 0),
        it jumps to route101 (3, 25), it should preserve the relative (0, -5) offset.
        
        Args:
            player_x: Current player X position
            player_y: Current player Y position
            player_map: Current map name
        """
        if self.state.navigation_target_delta is None:
            return  # Not tracking
        
        if self.state.last_player_pos is None or self.state.last_player_map is None:
            # Initialize tracking
            self.state.last_player_pos = (player_x, player_y)
            self.state.last_player_map = player_map
            return
        
        last_x, last_y = self.state.last_player_pos
        
        # Check if map changed
        if player_map != self.state.last_player_map:
            logger.info(f"Map transition detected: {self.state.last_player_map} -> {player_map}")
            
            # When map changes, preserve the relative coordinate offset
            # Current relative offset = target_delta - total_moved
            target_dx, target_dy = self.state.navigation_target_delta
            moved_x, moved_y = self.state.total_moved
            remaining_dx = target_dx - moved_x
            remaining_dy = target_dy - moved_y
            
            logger.info(f"Preserving relative offset: target_delta={self.state.navigation_target_delta}, "
                       f"total_moved={self.state.total_moved}, remaining=({remaining_dx}, {remaining_dy})")
            
            # Update navigation_start_pos to current position
            # This effectively resets the coordinate system to the new map
            self.state.navigation_start_pos = (player_x, player_y)
            self.state.navigation_start_map = player_map
            
            # Reset total_moved to 0 since we're starting from new position
            self.state.total_moved = (0, 0)
            
            # Keep the same target_delta (relative offset is preserved)
            # The target will now be: new_start_pos + target_delta
            
            logger.info(f"Updated navigation baseline to ({player_x}, {player_y}) on {player_map}")
            
            # Update last position
            self.state.last_player_pos = (player_x, player_y)
            self.state.last_player_map = player_map
        else:
            # Same map - calculate movement delta
            delta_x = player_x - last_x
            delta_y = player_y - last_y
            
            if delta_x != 0 or delta_y != 0:
                moved_x, moved_y = self.state.total_moved
                moved_x += delta_x
                moved_y += delta_y
                self.state.total_moved = (moved_x, moved_y)
                logger.debug(f"Moved ({delta_x}, {delta_y}), total moved: {self.state.total_moved}")
            
            # Update last position
            self.state.last_player_pos = (player_x, player_y)
    
    def _check_local_navigation_complete(self) -> bool:
        """
        Check if local navigation objective is complete.
        
        Returns:
            True if target delta has been reached
        """
        if self.state.navigation_target_delta is None:
            return False
        
        target_dx, target_dy = self.state.navigation_target_delta
        moved_x, moved_y = self.state.total_moved
        
        # Check if we've moved at least the target distance
        # For positive target: moved >= target
        # For negative target: moved <= target
        x_complete = (target_dx >= 0 and moved_x >= target_dx) or (target_dx < 0 and moved_x <= target_dx)
        y_complete = (target_dy >= 0 and moved_y >= target_dy) or (target_dy < 0 and moved_y <= target_dy)
        
        is_complete = x_complete and y_complete
        
        if is_complete:
            logger.info(f"Local navigation complete! Target: {self.state.navigation_target_delta}, Moved: {self.state.total_moved}")
        
        return is_complete
    
    def _stop_local_navigation_tracking(self) -> None:
        """Stop tracking local navigation and reset state."""
        self.state.navigation_start_pos = None
        self.state.navigation_start_map = None
        self.state.navigation_target_delta = None
        self.state.total_moved = (0, 0)
        self.state.last_player_pos = None
        self.state.last_player_map = None
        
        logger.debug("Stopped local navigation tracking")
    
    def _check_dialogue_loop(self) -> Optional[str]:
        """
        Check if there are repeated dialogues in the last 10 turns.
        
        Returns:
            Repeated dialogue text if loop detected, None otherwise
        """
        logger.debug("DIALOGUE LOOP CHECK FIRED ===================== Checking for dialogue loop...")
        
        # Skip loop detection for 5 turns after fallback to give it time to resolve
        if (self.state.last_fallback_step is not None and 
            self.state.current_step_num - self.state.last_fallback_step < 5):
            logger.debug("Skipping dialogue loop check: within 5 turns of fallback")
            return None
        
        if len(self.state.dialogue_history) < 2:
            return None
        
        # Get last 10 dialogues
        recent_dialogues = self.state.dialogue_history[-10:]
        
        logger.debug(f"Checking {len(recent_dialogues)} recent dialogues for loops")
        for entry in recent_dialogues[-5:]:  # Log last 5
            logger.debug(f"  Step {entry['step']}: '{entry.get('text', '')}'")
        
        # Count occurrences of each dialogue text (normalized)
        dialogue_counts = {}
        for entry in recent_dialogues:
            text = re.sub(r'[^\w\s]', '', entry.get('text', '').strip().lower())
            if text and text != 'no dialogue':
                dialogue_counts[text] = dialogue_counts.get(text, 0) + 1
        
        # Check if any dialogue appears 2+ times
        for text, count in dialogue_counts.items():
            if count >= 2:
                logger.warning(f"Dialogue loop detected: '{text[:50]}...' appeared {count} times in last 10 turns")
                return text
        
        return None
    
    def _check_action_loop(self) -> Optional[str]:
        """
        Check for repeated actions based on frame feature similarity.
        
        Returns:
            Description of loop if detected (cosine similarity > 0.95), None otherwise
        """
        if len(self.state.action_history) < 5:
            return None
        
        # Get last 5 frame features
        recent_features = [entry.get('frame_features') for entry in self.state.action_history[-5:] if 'frame_features' in entry]
        
        if len(recent_features) < 2:
            return None
        
        # Check pairwise cosine similarity
        for i in range(len(recent_features)):
            for j in range(i + 1, len(recent_features)):
                sim = self._cosine_similarity(recent_features[i], recent_features[j])
                if sim > 0.95:
                    logger.warning(f"Action loop detected: cosine similarity={sim:.3f} between frames {i} and {j}")
                    return f"Frame similarity {sim:.3f}"
        
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _fallback_vlm_strong_model(
        self,
        state_data: dict,
        frame: np.ndarray,
        repeated_content: str,
        current_objective: ObjectiveItem,
        loop_type: str = "dialogue"
    ) -> Dict:
        """
        Fallback VLM using stronger model (GPT-4o) when loop detected.
        
        Args:
            state_data: Game state data
            frame: Current game frame
            repeated_content: The repeated content (dialogue or reasoning) causing the loop
            current_objective: Current objective
            loop_type: Type of loop ("dialogue" or "action")
            
        Returns:
            Dict with actions to break the loop
        """
        from utils.state_formatter import format_state_summary
        
        # Extract comprehensive screen text information
        # Always extract fresh screen text for current state
        screen_text = False #self._extract_screen_text(frame)
        # logger.info(f"Extracted screen text for fallback VLM: Dialogue='{screen_text.dialogue_text[:50]}...', Menu Options={len(screen_text.menu_options) if screen_text.menu_options else 0}")

        state_summary = _clean_state_summary(format_state_summary(state_data))
        
        # Get last 5 actions and their reasoning
        recent_actions = self.state.action_history[-5:]
        actions_summary_lines = ["Actions", ]
        for i, entry in enumerate(recent_actions):
            actions_summary_lines.extend(
                entry['actions']
            )
            # full_reasoning = entry.get('reasoning', 'N/A')
            # if i >= len(recent_actions) - 2:  # Last 2 entries (most recent) show full reasoning
            #     reasoning_display = full_reasoning
            # else:  # Other 3 show truncated reasoning
            #     if len(full_reasoning) > 50:
            #         reasoning_display = full_reasoning[:50] + "..."
            #     else:
            #         reasoning_display = full_reasoning
            # actions_summary_lines.append(
            #     f"Step {entry['step']}: Actions={entry['actions']}, Reasoning={reasoning_display}"
            # )
        actions_summary = ", ".join(actions_summary_lines)
        
        # Determine header based on loop type
        actions_header = "Recent Actions:" if loop_type != "battle" else "Recent Actions:"
        
        if loop_type == "dialogue":
            loop_description = f"STUCK IN A DIALOGUE LOOP.\n\nREPEATED DIALOGUE (causing loop):\n\"{repeated_content}\""
        else:
            loop_description = f"STUCK IN AN ACTION LOOP.\n\nREPEATED REASONING (causing loop):\n\"{repeated_content}\""
        
        # Build current screen information
        menu_options_text = ""
        if screen_text and screen_text.menu_options:
            menu_options_text = "Option and sequence of buttons to press:\n" + "\n".join([f"  - {menu_option.option}: {', '.join(menu_option.sequence)}" for menu_option in screen_text.menu_options])
            screen_info = f"""
CURRENT SCREEN INFORMATION:
- Dialogue: {screen_text.dialogue_text}
- Menu Options:
{menu_options_text}
- Current Selection: {screen_text.current_selection if screen_text.current_selection else 'None'}"""
        else:
            menu_options_text = " None visible"
            screen_info = "Read screen information from the frame provided, including instructions, menu options, and curernt menu option which has a filled right arrow cursor >."

# 2. If you've been moving in one direction, try a different direction
# 3. If stuck in menu, try 'B' to back out
# 4. IMPORTANT: If you have been pressing the same sequence of buttons in recent actions, try changing up the order
# 5. Consider that the current approach is NOT working and needs to change

        prompt = f"""You are playing Pokemon Emerald and are {loop_description}.

Current Objective: {current_objective.name}
Description: {current_objective.description}
Type: {current_objective.type}
This has appeared multiple times in the last few steps, indicating you are stuck: {loop_description} 

This is the current screen and state info:
{screen_info}
{state_summary}

{actions_header}
{actions_summary}

You need to break out of this loop by trying a DIFFERENT approach:
1. If you've been pressing 'A', try moving in a different direction or "B" first
2. See the sequence of past actions and try to initiate a different order of sequence of button presses. Example: ["A", "UP", "A"] -> try ["UP", "A", "A"] or ["A", "A", "UP"]


Valid buttons: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, L, R

Respond with a JSON object:
{{
    "reasoning": "Explanation of why previous approach failed and what to try instead",
    "actions": ["BUTTON1", "BUTTON2", ...],
    "current_objective_completed": true/false
}}
"""
        
        logger.warning(f"Calling STRONG MODEL (GPT-4o) to break dialogue loop")
        logger.info(f"Prompt for strong model fallback:\n{prompt}")
        
        try:
            # Use GPT-4o (stronger model) for fallback
            from pydantic import BaseModel
            
            # Define structured output model for fallback
            class FallbackVLMResponse(BaseModel):
                reasoning: str
                actions: List[str]
                current_objective_completed: bool
            
            # Temporarily change model to GPT-4o for this call
            original_model = self.model_name
            self.model_name = "gpt-4o"
            
            # Use call_vlm_with_logging with structured output
            result: FallbackVLMResponse = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame,
                module_name="FALLBACK_VLM_STRONG_MODEL",
                structured_output_model=FallbackVLMResponse
            )
            
            # Restore original model
            self.model_name = original_model
            
            actions = result.actions
            reasoning = result.reasoning
            objective_completed = result.current_objective_completed
            
            logger.info(f"Strong model reasoning: {reasoning}")
            logger.info(f"Strong model actions: {actions}")
            
            # Track this reasoning
            if len(self.state.reasoning_history) >= 15:
                self.state.reasoning_history.pop(0)
            self.state.reasoning_history.append(f"[FALLBACK] {reasoning}")
            
            if objective_completed:
                obj_idx = next((i for i, obj in enumerate(self.state.objectives) if obj == current_objective), None)
                if obj_idx is not None:
                    self._mark_objective_complete(obj_idx)
            
            return {'action': actions, 'reasoning': reasoning}
                
        except Exception as e:
            logger.error(f"Strong model fallback failed: {e}")
            return {'action': ['B']}  # Fallback: press B to cancel
    
    def _extract_dialogue_text(self, frame: np.ndarray) -> str:
        """
        Extract dialogue text from the frame using VLM.
        
        Args:
            frame: Game frame as numpy array
        
        Returns:
            Extracted dialogue text or error/placeholder message
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(frame, np.ndarray):
                frame_pil = Image.fromarray(frame)
            else:
                frame_pil = frame
            
            # Ensure frame is upscaled to 480x320
            if frame_pil.size[1] == 160:
                frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
            
            # Crop to bottom 25% of the screen where dialogue appears
            bottom_quarter_height = frame_pil.height // 4
            dialogue_region = frame_pil.crop((0, frame_pil.height - bottom_quarter_height, frame_pil.width, frame_pil.height))
            
            # Convert cropped PIL to numpy array for VLM
            frame_np = np.array(dialogue_region)
            
            prompt = """You are analyzing a Pokemon game screenshot. 

Look at the bottom portion of the screen where dialogue text appears. Extract and return ONLY the dialogue text that is currently visible. 

Important: This is a completely new and independent screenshot. Do not reference, include, or remember any text from previous screenshots or extractions. Analyze only this current image.

Rules:
- Read and return only the actual dialogue text content from the screenshot
- If no dialogue is visible, return "NO_DIALOGUE"
- Keep the text exactly as it appears in the game

Respond with a JSON object:
{
    "dialogue_text": "the extracted text or NO_DIALOGUE"
}
"""
            
            logger.debug("Calling VLM for dialogue text extraction")
            
            # Temporarily ensure we use the weaker model for dialogue extraction
            original_model = self.model_name
            self.model_name = "gpt-4o-mini"
            
            result: DialogueTextExtraction = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame_np,
                module_name="DIALOGUE_TEXT_EXTRACTION",
                structured_output_model=DialogueTextExtraction
            )
            
            # Restore original model
            self.model_name = original_model
            
            dialogue_text = result.dialogue_text.strip()
            
            if dialogue_text.upper() in ["NO_DIALOGUE", "NO DIALOGUE", ""]:
                return "NO_DIALOGUE"
            
            return dialogue_text
            
        except Exception as e:
            logger.error(f"Error extracting dialogue text with VLM: {e}")
            return "DIALOGUE_EXTRACTION_ERROR"
    
    def _extract_screen_text(self, frame: np.ndarray) -> ScreenTextExtraction:
        """
        Extract comprehensive text information from the entire game screen.
        
        Args:
            frame: Game frame as numpy array
            
        Returns:
            ScreenTextExtraction with dialogue, menu options, and current selection
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(frame, np.ndarray):
                frame_pil = Image.fromarray(frame)
            else:
                frame_pil = frame
            
            # Ensure frame is upscaled to 480x320
            if frame_pil.size[1] == 160:
                frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
            
            prompt = """You are analyzing a Pokemon game screenshot. Extract all visible text information from the screen.

Look for and extract:

1. DIALOGUE TEXT: Any text in the dialogue box at the bottom of the screen
2. MENU OPTIONS WITH NAVIGATION: Any options shown in white rectangular boxes (like battle commands: Fight, Bag, Pokemon, Run). For each option, provide the button sequence needed to select it from the current cursor position.
3. CURRENT SELECTION: The option that has a right filled arrow/triangle ">" beside it indicating current cursor position

Rules:
- For dialogue_text: Extract the full dialogue text, or "NO_DIALOGUE" if none visible
- For menu_options: List of objects, each with "option" (string) and "sequence" (array of button presses) needed to select that option from current position
- For current_selection: The single option with ">" beside it, or null if no selection visible

Pokemon battle menus are typically arranged in a 2x2 grid:
Fight | Bag
Pokemon | Run

Pokemon selection menus are typically arranged vertically:
Yes
No
Pokemon
Bag

Navigation from current position:
- If "Fight" is selected (> Fight): Fight=["A"], Bag=["RIGHT","A"], Pokemon=["DOWN","A"], Run=["DOWN","RIGHT","A"]
- If "Bag" is selected (> Bag): Fight=["LEFT","A"], Bag=["A"], Pokemon=["DOWN","LEFT","A"], Run=["DOWN","A"]
- If "Pokemon" is selected (> Pokemon): Fight=["UP","A"], Bag=["UP","RIGHT","A"], Pokemon=["A"], Run=["RIGHT","A"]
- If "Run" is selected (> Run): Fight=["UP","LEFT","A"], Bag=["UP","A"], Pokemon=["LEFT","A"], Run=["A"]

Examples:
- If current selection is "Fight": menu_options = [{"option": "Fight", "sequence": ["A"]}, {"option": "Bag", "sequence": ["RIGHT", "A"]}, {"option": "Pokemon", "sequence": ["DOWN", "A"]}, {"option": "Run", "sequence": ["DOWN", "RIGHT", "A"]}]
- If current selection is "Bag": menu_options = [{"option": "Fight", "sequence": ["LEFT", "A"]}, {"option": "Bag", "sequence": ["A"]}, {"option": "Pokemon", "sequence": ["DOWN", "LEFT", "A"]}, {"option": "Run", "sequence": ["DOWN", "A"]}]

Respond with a JSON object:
{
    "dialogue_text": "the dialogue text or NO_DIALOGUE",
    "menu_options": [{"option": "option1", "sequence": ["BUTTON1", "BUTTON2"]}, {"option": "option2", "sequence": ["BUTTON1"]}],
    "current_selection": "selected_option" or null
}
"""
            
            logger.debug("Calling VLM for comprehensive screen text extraction")
            
            # Always use stronger model for comprehensive extraction
            original_model = self.model_name
            self.model_name = "gpt-5"
            
            result: ScreenTextExtraction = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame_pil,
                module_name="SCREEN_TEXT_EXTRACTION",
                structured_output_model=ScreenTextExtraction
            )
            
            # Restore original model
            self.model_name = original_model
            
            # Clean up the results
            result.dialogue_text = result.dialogue_text.strip()
            if result.dialogue_text.upper() in ["NO_DIALOGUE", "NO DIALOGUE", ""]:
                result.dialogue_text = "NO_DIALOGUE"
            
            # Clean menu options - ensure all button sequences are uppercase
            cleaned_options = []
            for menu_option in result.menu_options:
                option_name = menu_option.option.strip()
                # Skip "Save" option entirely
                if option_name.upper() == "SAVE":
                    continue
                if option_name and menu_option.sequence:
                    cleaned_buttons = [btn.upper().strip() for btn in menu_option.sequence if btn.strip()]
                    if cleaned_buttons:
                        cleaned_options.append(MenuOption(option=option_name, sequence=cleaned_buttons))
            result.menu_options = cleaned_options
            
            # Clean current selection
            if result.current_selection:
                result.current_selection = result.current_selection.strip()
                if not result.current_selection:
                    result.current_selection = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting screen text with VLM: {e}")
            return ScreenTextExtraction(
                dialogue_text="SCREEN_TEXT_EXTRACTION_ERROR",
                menu_options={},
                current_selection=None
            )
    
    def _compute_menu_navigation_sequences(self, current_selection: str, menu_options: List[str]) -> List[MenuOption]:
        """
        Compute button sequences needed to navigate from current selection to each menu option.
        
        Assumes standard Pokemon battle menu layout:
        Fight | Bag
        Pokemon | Run
        
        Args:
            current_selection: Currently selected option
            menu_options: List of all available options
            
        Returns:
            List of MenuOption objects with computed sequences
        """
        # Define positions in 2x2 grid
        positions = {
            "Fight": (0, 0),    # Top-left
            "Bag": (0, 1),      # Top-right  
            "Pokemon": (1, 0),  # Bottom-left
            "Run": (1, 1)       # Bottom-right
        }
        
        if current_selection not in positions:
            # Fallback: assume all options need just "A" if current position unknown
            return [MenuOption(option=option, sequence=["A"]) for option in menu_options]
        
        current_pos = positions[current_selection]
        result_options = []
        
        for option in menu_options:
            if option not in positions:
                result_options.append(MenuOption(option=option, sequence=["A"]))  # Unknown option
                continue
                
            if option == current_selection:
                result_options.append(MenuOption(option=option, sequence=["A"]))  # Already selected
                continue
            
            target_pos = positions[option]
            buttons = []
            
            # Calculate vertical movement
            row_diff = target_pos[0] - current_pos[0]
            if row_diff > 0:
                buttons.extend(["DOWN"] * row_diff)
            elif row_diff < 0:
                buttons.extend(["UP"] * abs(row_diff))
            
            # Calculate horizontal movement
            col_diff = target_pos[1] - current_pos[1]
            if col_diff > 0:
                buttons.extend(["RIGHT"] * col_diff)
            elif col_diff < 0:
                buttons.extend(["LEFT"] * abs(col_diff))
            
            # Add confirmation
            buttons.append("A")
            
            result_options.append(MenuOption(option=option, sequence=buttons))
        
        return result_options
    
    def _get_global_to_map_offset(self, map_name: str) -> Optional[Tuple[int, int]]:
        """
        Get the offset to convert global ("G") coordinates to map coordinates.
        
        This is calculated by finding Mom's dialogue in dialogues.json and comparing:
        - Mom's global coordinates from entity_database.json ("mom_initial": [["G", 138, 312]])
        - Mom's map coordinates from the dialogue NPC position
        
        The offset is: (global_x - map_x, global_y - map_y)
        
        Caches the result to navigation_caches/mom_initial_position.json to avoid repeated VLM calls.
        
        Args:
            map_name: Current map name (should be "LITTLEROOT TOWN" for initial calculation)
            
        Returns:
            (offset_x, offset_y) or None if not yet calculable
        """
        # Check cache first
        cache_path = Path(NAVIGATION_CACHE_DIR) / "mom_initial_position.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Verify cache has required fields
                if all(k in cache_data for k in ['global_x', 'global_y', 'map_x', 'map_y', 'map_name', 'offset_x', 'offset_y']):
                    offset_x = cache_data['offset_x']
                    offset_y = cache_data['offset_y']
                    logger.debug(f"Using cached global offset: ({offset_x}, {offset_y})")
                    return (offset_x, offset_y)
            except Exception as e:
                logger.warning(f"Failed to load offset cache: {e}")
        
        # Check if we have mom_initial in entity database
        if "mom_initial" not in self.entity_database:
            logger.debug("No mom_initial in entity database")
            return None
        
        mom_entity_data = self.entity_database["mom_initial"]
        if not mom_entity_data or len(mom_entity_data) == 0:
            return None
        
        # Verify it's a global coordinate
        if mom_entity_data[0][0] != "G":
            logger.warning("mom_initial is not in global coordinate format")
            return None
        
        mom_global_x = mom_entity_data[0][1]
        mom_global_y = mom_entity_data[0][2]
        
        # Load dialogues to find Mom's dialogue
        dialogue_path = Path(DIALOGUE_LOG_FILE)
        if not dialogue_path.exists():
            logger.debug("No dialogue log yet - cannot calculate global offset")
            return None
        
        try:
            with open(dialogue_path, 'r') as f:
                dialogues = json.load(f)
            
            # Find the earliest completed dialogue (has both start and end) on LITTLEROOT TOWN
            # Look for NPC class dialogues with dialogue text
            dialogue_sessions = {}
            
            # Group dialogues by dialogue_count to find complete sessions
            for dialogue in dialogues:
                count = dialogue.get('dialogue_count')
                if count is None:
                    continue
                
                if count not in dialogue_sessions:
                    dialogue_sessions[count] = {'start': None, 'end': None}
                
                if dialogue.get('event') == 'dialogue_start':
                    dialogue_sessions[count]['start'] = dialogue
                elif dialogue.get('event') == 'dialogue_end':
                    dialogue_sessions[count]['end'] = dialogue
            
            # Find complete sessions on LITTLEROOT TOWN with NPC class, sorted by step_number
            complete_sessions = []
            for count, session in dialogue_sessions.items():
                start_entry = session.get('start')
                end_entry = session.get('end')
                
                if (start_entry is not None and end_entry is not None and
                    start_entry.get('player_map', '').upper() == 'LITTLEROOT TOWN' and
                    start_entry.get('active_tile_class') == 'npc' and
                    start_entry.get('npc_tile_position') is not None and
                    start_entry.get('dialogue_text')):
                    
                    complete_sessions.append({
                        'dialogue_count': count,
                        'step_number': start_entry.get('step_number', 0),
                        'timestamp': start_entry.get('timestamp', 0),
                        'dialogue_text': start_entry.get('dialogue_text', ''),
                        'npc_pos': start_entry.get('npc_tile_position')
                    })
            
            # Sort by step_number (earliest first)
            complete_sessions.sort(key=lambda x: x['step_number'])
            
            # Check each session using VLM to verify it's Mom
            for session in complete_sessions:
                dialogue_text = session['dialogue_text']
                
                # Use VLM to verify this is Mom
                logger.info(f"Checking if dialogue is with Mom (step {session['step_number']}): {dialogue_text[:50]}...")
                
                prompt = f"""You are analyzing a dialogue from Pokemon Emerald to identify the speaker.

Dialogue text: "{dialogue_text}"

Context:
- This dialogue occurred on LITTLEROOT TOWN
- This is the player's starting town
- The player just started the game and is in their house
- Mom is the player's mother who greets them at the start of the game

Based on the dialogue content, determine if the speaker is Mom (the player's mother).

Common Mom dialogue includes:
- Welcoming the player to their new home
- Asking about unpacking or settling in
- Mentioning the moving truck
- Telling the player to set the clock upstairs
- General motherly/welcoming tone

Respond with a JSON object:
{{
    "analysis": "Brief analysis of the dialogue content and speaker identity",
    "is_mom": true/false,
    "confidence": "high/medium/low"
}}
"""
                
                try:
                    verification: DialogueSpeakerVerification = self.call_vlm_with_logging(
                        prompt=prompt,
                        image=None,  # Text-only analysis
                        module_name="MOM_DIALOGUE_VERIFICATION",
                        structured_output_model=DialogueSpeakerVerification
                    )
                    
                    logger.info(f"VLM verification: {verification.analysis}")
                    logger.info(f"Is Mom: {verification.is_mom}, Confidence: {verification.confidence}")
                    
                    if verification.is_mom and verification.confidence in ['high', 'medium']:
                        # This is Mom! Use this dialogue for offset calculation
                        npc_pos = session['npc_pos']
                        npc_map_x = npc_pos.get('x')
                        npc_map_y = npc_pos.get('y')
                        
                        if npc_map_x is None or npc_map_y is None:
                            continue
                        
                        # Calculate offset: global - map = offset
                        offset_x = mom_global_x - npc_map_x
                        offset_y = mom_global_y - npc_map_y
                        
                        logger.info(f"✓ Verified Mom dialogue at step {session['step_number']}")
                        logger.info(f"Calculated global offset from Mom's dialogue: ({offset_x}, {offset_y})")
                        logger.info(f"  Mom global: ({mom_global_x}, {mom_global_y}), Mom map: ({npc_map_x}, {npc_map_y})")
                        
                        # Cache the result
                        try:
                            cache_dir = Path(NAVIGATION_CACHE_DIR)
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            
                            cache_data = {
                                'global_x': mom_global_x,
                                'global_y': mom_global_y,
                                'map_x': npc_map_x,
                                'map_y': npc_map_y,
                                'map_name': 'LITTLEROOT TOWN',
                                'offset_x': offset_x,
                                'offset_y': offset_y,
                                'dialogue_step': session['step_number'],
                                'dialogue_text': dialogue_text
                            }
                            
                            with open(cache_path, 'w') as f:
                                json.dump(cache_data, f, indent=2)
                            
                            logger.info(f"Cached Mom's position to {cache_path}")
                        except Exception as cache_error:
                            logger.warning(f"Failed to cache Mom's position: {cache_error}")
                        
                        return (offset_x, offset_y)
                    
                except Exception as e:
                    logger.warning(f"VLM verification failed for dialogue: {e}")
                    continue
            
            logger.debug("No verified Mom dialogue found yet to calculate global offset")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load dialogues for offset calculation: {e}")
            return None
    
    def _score_grass_target(
        self, 
        grass_pos: Tuple[int, int], 
        player_pos: Tuple[int, int], 
        last_visited_step: Optional[int]
    ) -> float:
        """
        Score a grass target based on visitation time and distance.
        
        Score = (curr_step_num - last_visited_step) / curr_step_num * manhattan_distance
        
        Higher score = more desirable (less recently visited, farther away)
        
        Args:
            grass_pos: Grass tile position (x, y)
            player_pos: Player position (x, y)
            last_visited_step: Last step number when visited (None if never visited)
            
        Returns:
            Score value (higher is better)
        """
        # Calculate Manhattan distance
        manhattan_dist = abs(grass_pos[0] - player_pos[0]) + abs(grass_pos[1] - player_pos[1])
        
        # If never visited, give maximum time weight
        if last_visited_step is None or self.state.current_step_num == 0:
            time_weight = 1.0
        else:
            time_weight = (self.state.current_step_num - last_visited_step) / max(self.state.current_step_num, 1)
        
        score = time_weight * manhattan_dist
        return score
    
    def _get_current_objective(self) -> Optional[ObjectiveItem]:
        """Get the current active objective"""
        if not self.state.objectives:
            return None
        
        # Find first incomplete objective
        for i, obj in enumerate(self.state.objectives):
            if obj.status == "[ ]":
                self.state.current_objective_index = i
                return obj
        
        # All objectives complete
        return None
    
    def _mark_objective_complete(self, objective_index: int) -> None:
        """Mark an objective as complete"""
        if 0 <= objective_index < len(self.state.objectives):
            self.state.objectives[objective_index].status = "[x]"
            self._save_objectives()
            logger.info(f"Marked objective {objective_index} as complete: {self.state.objectives[objective_index].name}")

        self.state.current_objective_index = -1
        
        # Stop local navigation tracking when objective completes
        if self.state.navigation_target_delta is not None:
            self._stop_local_navigation_tracking()
    
    def _check_level_up_completion(self, state_data: dict) -> bool:
        """
        Check if level grinding objective is complete.
        
        Args:
            state_data: Game state data
            
        Returns:
            True if level grinding complete, False otherwise
        """
        current_obj = self._get_current_objective()
        if not current_obj or current_obj.type != "level_grinding":
            return False
        
        target_level = current_obj.target_level
        if target_level is None:
            return False
        
        # Check party Pokemon levels
        party = state_data.get('player', {}).get('party', [])
        if not party:
            return False
        
        # Check if lead Pokemon reached target level
        lead_pokemon = party[0]
        current_level = lead_pokemon.get('level', 0)
        
        if current_level >= target_level:
            logger.info(f"Level grinding complete: Level {current_level}/{target_level}")
            return True
        
        return False
    
    def _get_completed_dialogues(self) -> List[Dict]:
        """
        Get list of completed dialogues from dialogue log.
        
        Returns:
            List of completed dialogue info with coordinate, text, and step number
        """
        dialogue_path = Path(DIALOGUE_LOG_FILE)
        if not dialogue_path.exists():
            return []
        
        try:
            with open(dialogue_path, 'r') as f:
                dialogues = json.load(f)
            
            # Group dialogues by dialogue_count to find complete sessions
            dialogue_sessions = {}
            for dialogue in dialogues:
                count = dialogue.get('dialogue_count')
                if count is None:
                    continue
                
                if count not in dialogue_sessions:
                    dialogue_sessions[count] = {'start': None, 'end': None}
                
                if dialogue.get('event') == 'dialogue_start':
                    dialogue_sessions[count]['start'] = dialogue
                elif dialogue.get('event') == 'dialogue_end':
                    dialogue_sessions[count]['end'] = dialogue
            
            # Extract completed dialogues (have both start and end)
            completed = []
            for count, session in dialogue_sessions.items():
                start_entry = session.get('start')
                end_entry = session.get('end')
                
                if start_entry is not None and end_entry is not None:
                    npc_pos = start_entry.get('npc_tile_position', {})
                    dialogue_text = start_entry.get('dialogue_text', 'NO_TEXT')
                    step_num = start_entry.get('step_number', 0)
                    player_map = start_entry.get('player_map', 'unknown')
                    
                    if dialogue_text and dialogue_text not in ['EXTRACTION_FAILED', 'NO_DIALOGUE']:
                        completed.append({
                            'coordinate': (npc_pos.get('x'), npc_pos.get('y')),
                            'map': player_map,
                            'text': dialogue_text,
                            'step': step_num
                        })
            
            return completed
            
        except Exception as e:
            logger.error(f"Failed to load completed dialogues: {e}")
            return []
    
    def _check_fix_clock_completion(self, dialogue_text: str, current_step: int) -> bool:
        """
        Check if the "Fix clock" objective should be marked as completed.
        
        Looks for:
        1. A previous dialogue_start with "the clock is stopped"
        2. Current dialogue_text contains "mom" and "your new room"
        3. Current step > clock dialogue step
        
        Args:
            dialogue_text: Current dialogue text
            current_step: Current step number
            
        Returns:
            True if objective should be completed
        """
        try:
            import json
            from pathlib import Path
            
            log_path = Path(DIALOGUE_LOG_FILE)
            if not log_path.exists():
                return False
            
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            # Find the earliest dialogue_start with "the clock is stopped"
            clock_step = None
            for entry in log_data:
                if (entry.get('event') == 'dialogue_start' and 
                    'the clock is stopped' in entry.get('dialogue_text', '').lower()):
                    if clock_step is None or entry['step_number'] < clock_step:
                        clock_step = entry['step_number']
            
            if clock_step is None:
                return False
            
            # Check if current step is after clock step and text matches
            if (current_step > clock_step and 
                'mom' in dialogue_text.lower() and 
                'your new room' in dialogue_text.lower()):
                return True
                
        except Exception as e:
            logger.error(f"Failed to check fix clock completion: {e}")
        
        return False
    
    def _check_objective_completion_with_vlm(
        self,
        state_data: dict,
        frame: np.ndarray,
        objective: ObjectiveItem
    ) -> bool:
        """
        Use VLM to check if objective is completed based on recent reasoning history.
        
        Called every 5 steps to review progress.
        Also checks if previously completed objectives need to be rolled back.
        
        Args:
            state_data: Game state data
            frame: Current game frame
            objective: Current objective to check
            
        Returns:
            True if objective is completed, False otherwise
        """
        from utils.state_formatter import format_state_summary
        from pydantic import BaseModel
        
        state_summary = _clean_state_summary(format_state_summary(state_data))
        
        # Format reasoning history
        reasoning_text = "\n".join([
            f"{i+1}. {reason}" 
            for i, reason in enumerate(self.state.reasoning_history)
        ])
        
        if not reasoning_text:
            reasoning_text = "No reasoning history available yet."
        
        # Get list of previously completed objectives for rollback detection
        completed_objectives = []
        for i, obj in enumerate(self.state.objectives):
            if obj.status == "[x]":
                completed_objectives.append(f"{i}. {obj.name} - {obj.description}")
        
        completed_text = "\n".join(completed_objectives) if completed_objectives else "None"
        
        # Get completed dialogues
        completed_dialogues = self._get_completed_dialogues()
        dialogues_text = ""
        if completed_dialogues:
            dialogues_list = []
            for dlg in completed_dialogues:
                coord = dlg['coordinate']
                text_preview = dlg['text'][:60] + "..." if len(dlg['text']) > 60 else dlg['text']
                dialogues_list.append(
                    f"  - Map: {dlg['map']}, Pos: ({coord[0]}, {coord[1]}), Step: {dlg['step']}\n    Text: {text_preview}"
                )
            dialogues_text = "\n" + "\n".join(dialogues_list)
        else:
            dialogues_text = " None"
        
        prompt = f"""You are playing Pokemon Emerald and need to determine if the current objective is complete.

Current Objective: {objective.name}
Description: {objective.description}
Type: {objective.type}
{f"Target Entity: {objective.target_entity}" if objective.target_entity else ""}
{f"Target Level: {objective.target_level}" if objective.target_level else ""}

Previously Completed Objectives:
{completed_text}

Completed Dialogues:{dialogues_text}

Current State:
{state_summary}

Recent Reasoning History (last 15 steps):
{reasoning_text}

Based on the objective description, current game state, and recent actions/reasoning, determine:
1. If the current objective has been completed
2. If any previously completed objectives need to be rolled back (player hasn't actually progressed)

For example:
- If objective is "Exit moving van", check if player is NOW in a different location (not MOVING_VAN)
- If objective is "Talk to Mom", check if dialogue with Mom occurred
- If objective is "Navigate to Route 101", check if player is NOW in Route 101
- If objective is "Level up to 10", check if Pokemon level is NOW 10 or higher

IMPORTANT: If the player is still in the same location as an earlier objective (e.g., still in MOVING_VAN after "Exit moving van" was marked complete), that objective should be rolled back.

Respond with a JSON object:
{{
    "analysis": "Brief analysis of whether objective is complete and if rollback is needed",
    "is_completed": true/false,
    "objectives_to_rollback": [list of objective indices to mark incomplete again, or empty list]
}}
"""
        
        logger.info("Calling VLM for 5-step objective completion review")
        
        # use cheaper model for this frequent check
        original_model = self.model_name
        self.model_name = "gpt-4o-mini"

        try:
            # Extended model to include rollback info
            class ObjectiveReviewExtended(BaseModel):
                analysis: str = Field(description="Analysis of objective completion status")
                is_completed: bool = Field(description="Whether the objective is completed")
                objectives_to_rollback: List[int] = Field(description="Indices of objectives to rollback (mark incomplete)")
            
            review: ObjectiveReviewExtended = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame,
                module_name="OBJECTIVE_REVIEW_VLM",
                structured_output_model=ObjectiveReviewExtended
            )
            
            logger.info(f"VLM objective review: {review.analysis}")
            logger.info(f"VLM objective completed: {review.is_completed}")
            
            # Handle rollback if needed
            if review.objectives_to_rollback:
                logger.warning(f"VLM detected objectives to rollback: {review.objectives_to_rollback}")
                for obj_idx in review.objectives_to_rollback:
                    if 0 <= obj_idx < len(self.state.objectives):
                        self.state.objectives[obj_idx].status = "[ ]"
                        logger.warning(f"Rolled back objective {obj_idx}: {self.state.objectives[obj_idx].name}")
                self._save_objectives()
                # Reset current objective index to first incomplete
                self.state.current_objective_index = 0

            self.model_name = original_model

            return review.is_completed
            
        except Exception as e:
            logger.error(f"VLM objective review failed: {e}")
            return False
 
    
    def _handle_menu_battle_with_vlm(
        self,
        state_data: dict,
        frame: np.ndarray,
        objective: ObjectiveItem,
        screen_text: Optional[ScreenTextExtraction] = None
    ) -> Dict:
        """
        Handle menu/battle situations with VLM fallback.
        
        Args:
            state_data: Game state data
            frame: Game frame
            objective: Current objective
            
        Returns:
            Dict with actions
        """
        from utils.state_formatter import format_state_summary
        
        # Extract comprehensive screen text information (or use provided)
        # if screen_text is None:
        #     screen_text = self._extract_screen_text(frame)
        
        state_summary = _clean_state_summary(format_state_summary(state_data))
        
        # Build current screen information
        menu_options_text = ""
        # if screen_text.menu_options:
        #     menu_options_text = "Option and sequence of buttons to press:\n" + "\n".join([f"  - {menu_option.option}: {', '.join(menu_option.sequence)}" for menu_option in screen_text.menu_options])
        # else:
        #     menu_options_text = " None visible"
        
#         screen_info = f"""
# CURRENT SCREEN INFORMATION:
# - Dialogue: {screen_text.dialogue_text}
# - Menu Options:
# {menu_options_text}
# - Current Selection: {screen_text.current_selection if screen_text.current_selection else 'None'}
# """
        
        if self.model_name == "gpt-5" or self.model_name == "gpt-4o":
            menu_prompt = """Read the screen for any action options and current selection marked by a black cursor >. The player is not visible (likely in menu or battle).
If in a menu, navigate to the appropriate option to progress the objective (e.g.
   Yes 
 > No
   -> select Yes by pressing "Up", "A").

            """
        else:
            menu_prompt = "Read the screen for any action options and navigate accordingly. The player is not visible (likely in menu or battle).\n"

        # Build context-aware prompt
        prompt = f"""You are playing Pokemon Emerald.

Current Objective: {objective.name}
Description: {objective.description}
Type: {objective.type}

Current State:
{state_summary}

Task:
{menu_prompt}

Decide the next action to progress toward the objective.
If in battle, typically just press 'A' to attack and then navigate to choose a damaging move, or a healing move if health is low. If "Absorb" move is available, prefer that for HP gain.

If no clear action just press "A" to advance dialogue or confirm selection.

Valid buttons: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, L, R

Respond with a JSON object:
{{
    "reasoning": "Brief explanation of your decision",
    "actions": ["BUTTON1", "BUTTON2", ...],
    "current_objective_completed": true/false (Has the current objective been completed based on the current game state?)
}}
"""
        
        logger.info("Calling VLM for menu/battle navigation")
        
        try:
            decision: MenuBattleDecision = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame,
                module_name="MENU_BATTLE_VLM",
                structured_output_model=MenuBattleDecision
            )
            
            logger.info(f"VLM decision: {decision.reasoning}")
            logger.info(f"VLM actions: {decision.actions}")
            
            # Add reasoning to history (keep last 15)
            self.state.reasoning_history.append(decision.reasoning)
            if len(self.state.reasoning_history) > 15:
                self.state.reasoning_history.pop(0)
            
            # # Check if objective completed
            # if decision.current_objective_completed:
            #     logger.info(f"VLM indicates objective completed: {objective.name}")
            #     self._mark_objective_complete(self.state.current_objective_index)
            
            return {'action': decision.actions}
            
        except Exception as e:
            logger.error(f"VLM call failed: {e}, defaulting to 'A'")
            return {'action': ['A']}
    
    def _handle_level_grinding(self, state_data: dict, frame: np.ndarray, **kwargs) -> Dict:
        """
        Handle level grinding by targeting grass tiles with visitation-based scoring.
        
        Args:
            state_data: Game state data
            frame: Game frame
            **kwargs: Additional arguments
            
        Returns:
            Dict with actions
        """
        player_direction_result = kwargs.get('player_direction_result')
        import random
        
        # Get player position and map
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        player_x = position.get('x', 0)
        player_y = position.get('y', 0)
        player_pos = (player_x, player_y)
        player_map = player_data.get('location', 'unknown')
        
        # Update grass cache for current position if on grass
        # (This is handled by detecting grass in the current tile)
        
        # Get traversability map for "~" tiles
        map_data = state_data.get('map', {})
        traversability_map = get_player_centered_grid(
            map_data=map_data,
            fallback_grid=[['.' for _ in range(15)] for _ in range(15)],
            npc_detections=None
        )
        
        # Collect all potential grass positions from "~" in traversability map
        scored_targets = []
        
        for y in range(15):
            for x in range(15):
                if traversability_map[y][x] == "~":
                    tile_dx = x - 7
                    tile_dy = y - 7
                    map_tile_x = player_x + tile_dx
                    map_tile_y = player_y + tile_dy
                    grass_pos = (map_tile_x, map_tile_y)
                    
                    # Get last visited step from cache
                    last_visited = None
                    if player_map in self.grass_cache:
                        last_visited = self.grass_cache[player_map].get(grass_pos)
                    
                    # Score the target
                    score = self._score_grass_target(grass_pos, player_pos, last_visited)
                    
                    scored_targets.append({
                        'position': grass_pos,
                        'score': score,
                        'last_visited': last_visited,
                        'object': None  # No object for "~"
                    })
        
        if not scored_targets:
            logger.warning("No grass tiles detected or found in traversability map, using normal navigation")
            nav_kwargs = kwargs.copy()
            nav_kwargs['player_direction_result'] = player_direction_result
            return self.nav_agent.choose_action(state_data, frame, **nav_kwargs)
        
        # Sort by score (descending)
        scored_targets.sort(key=lambda x: x['score'], reverse=True)
        
        # Select target based on threshold
        high_score_targets = [t for t in scored_targets if t['score'] > GRASS_SELECTION_THRESHOLD]
        
        if high_score_targets:
            # Randomly select from high-score targets
            chosen_target = random.choice(high_score_targets)
            logger.info(f"Selected high-score grass target: {chosen_target['position']} (score={chosen_target['score']:.2f})")
        else:
            # All targets below threshold - randomly select from top 5
            top_candidates = scored_targets[:min(5, len(scored_targets))]
            chosen_target = random.choice(top_candidates)
            logger.info(f"Selected top-5 grass target: {chosen_target['position']} (score={chosen_target['score']:.2f})")
        
        # Update grass cache for chosen target
        update_grass_cache(self.grass_cache, player_map, chosen_target['position'], self.state.current_step_num)
        
        # Create a modified kwargs with specific target for navigation agent
        grass_filter_kwargs = kwargs.copy()
        grass_filter_kwargs['target_x'] = chosen_target['position'][0]
        grass_filter_kwargs['target_y'] = chosen_target['position'][1]
        grass_filter_kwargs['player_direction_result'] = player_direction_result
        
        # Delegate to navigation agent with specific target
        result = self.nav_agent.choose_action(state_data, frame, **grass_filter_kwargs)
        
        return result
    
    def _save_grass_tiles_to_cache_and_index(self, grass_objects: List[Dict], frame: np.ndarray, player_x: int, player_y: int, player_map: str) -> None:
        """
        Save all detected grass tiles to the navigation cache and update active_tile_index.
        
        Args:
            grass_objects: List of detected grass objects
            frame: Current game frame
            player_x: Player X position
            player_y: Player Y position
            player_map: Current map name
        """
        # First save tiles using navigation agent's method to update active_tile_index
        for grass_obj in grass_objects:
            try:
                # Get grass position from bounding box center
                bbox = grass_obj.bbox
                grass_screen_x = bbox['x'] + bbox['w'] / 2
                grass_screen_y = bbox['y'] + bbox['h'] / 2
                
                # Convert screen coordinates to map tile coordinates
                screen_tile_x = int(grass_screen_x / 32)
                screen_tile_y = int((grass_screen_y - 16) / 32)
                
                # Convert to map tile coordinates (player is at 7, 7 in screen coords)
                map_tile_x = player_x + (screen_tile_x - 7)
                map_tile_y = player_y + (screen_tile_y - 7)
                grass_pos = (map_tile_x, map_tile_y)
                
                # Mark tile as grass using navigation agent's method
                mark_and_save_tile(
                    grass_pos, frame, 'grass', self.nav_agent.active_tile_index, player_map,
                    allow_upgrade=False, player_map_tile_x=player_x, player_map_tile_y=player_y
                )
                
            except Exception as e:
                logger.warning(f"Failed to save grass tile to index: {e}")
                continue
        
        # Then save features for any remaining tiles (legacy compatibility)
        self._save_grass_tile_features_only(grass_objects, frame, player_x, player_y)
    
    def _log_dialogue_session(self, state_data: dict, is_start: bool, dialogue_text: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Log dialogue session to JSON file.
        
        Args:
            state_data: Game state data
            is_start: True if dialogue starting, False if ending
            dialogue_text: Dialogue text to log (for start events, optional)
            **kwargs: Additional arguments, including 'frame' for dialogue extraction
        """
        frame = kwargs.get('frame', None)
        
        try:
            log_path = Path(DIALOGUE_LOG_FILE)
            
            # Load existing log
            if log_path.exists():
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Get player position
            player_data = state_data.get('player', {})
            position = player_data.get('position', {})
            player_x = position.get('x', 0)
            player_y = position.get('y', 0)
            player_map = player_data.get('location', 'unknown')
            
            # Use last known position if current position is default (player not visible)
            if player_x == 0 and player_y == 0:
                if self.state.last_player_pos:
                    player_x, player_y = self.state.last_player_pos
                if self.state.last_player_map:
                    player_map = self.state.last_player_map
            
            # Add new entry
            import time
            entry = {
                'timestamp': time.time(),
                'step_number': self.state.current_step_num,
                'event': 'dialogue_start' if is_start else 'dialogue_end',
                'player_position': {'x': player_x, 'y': player_y},
                'player_map': player_map,
                'dialogue_count': self.state.dialogue_count
            }
            
            # For dialogue_start, add NPC tile information
            if is_start:
                player_facing = self.state.player_facing
                
                # Calculate NPC tile position based on facing direction
                npc_tile_x, npc_tile_y = player_x, player_y
                if player_facing:
                    facing_offset = {
                        'North': (0, -1),
                        'South': (0, 1),
                        'East': (1, 0),
                        'West': (-1, 0)
                    }
                    offset = facing_offset.get(player_facing, (0, 0))
                    npc_tile_x = player_x + offset[0]
                    npc_tile_y = player_y + offset[1]
                
                # Get active tile info from navigation cache
                active_tile_filename = f"{player_map}_{npc_tile_x}_{npc_tile_y}.png"
                active_tile_class = None
                
                # Load active tile index to get class
                nav_cache_dir = Path(NAVIGATION_CACHE_DIR)
                index_path = nav_cache_dir / "active_tile_index.json"
                if index_path.exists():
                    try:
                        with open(index_path, 'r') as f:
                            index_data = json.load(f)
                            # Handle both old list and new dict format
                            if isinstance(index_data, dict):
                                if active_tile_filename in index_data:
                                    active_tile_class = index_data[active_tile_filename].get('class')
                            elif isinstance(index_data, list):
                                for item in index_data:
                                    if item.get('filename') == active_tile_filename:
                                        active_tile_class = item.get('class')
                                        break
                    except Exception as e:
                        logger.debug(f"Could not load active tile class: {e}")
                
                # Add NPC info to entry
                entry['npc_tile_position'] = {'x': npc_tile_x, 'y': npc_tile_y}
                entry['active_tile_filename'] = active_tile_filename
                entry['active_tile_class'] = active_tile_class
            
            # Extract dialogue text if provided and dialogue is starting
            if is_start and dialogue_text is not None:
                logger.info(f"Logged dialogue text: {dialogue_text}")
            elif is_start:
                dialogue_text = self._extract_dialogue_text(frame)
                logger.info(f"Extracted and logged dialogue text: {dialogue_text}")
            
            # Add dialogue text to entry if available
            if dialogue_text:
                # Check if there's a previous entry with the same dialogue_count to append to
                previous_text = None
                for prev_entry in reversed(log_data):
                    if prev_entry.get('dialogue_count') == self.state.dialogue_count and 'dialogue_text' in prev_entry:
                        previous_text = prev_entry.get('dialogue_text', '')
                        break
                
                # Append to existing dialogue text if found, but avoid duplicating partial text
                if previous_text and previous_text not in ["EXTRACTION_FAILED", "DIALOGUE_TEXT_PLACEHOLDER", "NO_DIALOGUE"]:
                    # Check if the new text is just an extension of the previous text
                    # (e.g., "But" -> "But despite" -> "But despite our")
                    if not dialogue_text.startswith(previous_text) and previous_text not in dialogue_text:
                        # New independent text, append with separator
                        entry['dialogue_text'] = previous_text + " | " + dialogue_text
                    else:
                        # New text is an extension or contains the old text, just use the new (longer) text
                        entry['dialogue_text'] = dialogue_text
                else:
                    entry['dialogue_text'] = dialogue_text
            
            log_data.append(entry)
            
            # Update last dialogue text tracking
            if is_start and 'dialogue_text' in entry:
                self.state.last_dialogue_text = entry['dialogue_text']
            elif not is_start:
                self.state.last_dialogue_text = None
            
            # Save log
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log dialogue session: {e}")
        
        # Return dialogue text if starting dialogue
        return dialogue_text if is_start else None
    
    def _save_grass_tile_features_only(self, grass_objects: List[Dict], frame: np.ndarray, player_x: int, player_y: int) -> None:
        """
        Save grass tile features to active_tile_simplefeatures.json for legacy compatibility.
        This is called after mark_and_save_tile has already saved the tiles and updated active_tile_index.
        
        Args:
            grass_objects: List of detected grass objects
            frame: Current game frame
            player_x: Player X position
            player_y: Player Y position
        """
        from pathlib import Path
        from PIL import Image
        from custom_utils.label_traversable import compute_simple_tile_features
        
        cache_dir = Path("./navigation_caches")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create features file (flat key:value format)
        features_path = cache_dir / "active_tile_simplefeatures.json"
        if features_path.exists():
            try:
                with open(features_path, 'r') as f:
                    features_data = json.load(f)
                
                # Validate that features_data is a dict, not a list
                if not isinstance(features_data, dict):
                    logger.warning(f"active_tile_simplefeatures.json contains {type(features_data)} instead of dict, resetting to empty dict")
                    features_data = {}
            except Exception as e:
                logger.warning(f"Failed to load active cache: {e}")
                features_data = {}
        else:
            features_data = {}
        
        # Get existing filenames to avoid duplicates
        existing_filenames = set(features_data.keys())
        
        # Convert frame to PIL Image and ensure proper preprocessing
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame.astype('uint8'), 'RGB')
        else:
            frame_pil = frame
        
        # Preprocess frame to ensure it's 480x352 (needed for consistent tile cropping)
        width, height = frame_pil.size
        if height == 160:
            # Upscale 2x (240x160 -> 480x320)
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
            logger.debug("Upscaled frame from 240x160 to 480x320 for grass tile saving")
        
        # Add padding if needed (480x320 -> 480x352)
        if frame_pil.size[1] == 320:
            padded_frame = Image.new('RGB', (480, 352), (0, 0, 0))
            padded_frame.paste(frame_pil, (0, 16))
            frame_pil = padded_frame
            logger.debug("Padded frame from 480x320 to 480x352 for grass tile saving")
        
        # Verify final size
        if frame_pil.size != (480, 352):
            logger.warning(f"Unexpected frame size after preprocessing for grass tile saving: {frame_pil.size}, expected (480, 352)")
        
        saved_count = 0
        for grass_obj in grass_objects:
            try:
                # Get grass position from bounding box center
                bbox = grass_obj.bbox
                grass_screen_x = bbox['x'] + bbox['w'] / 2
                grass_screen_y = bbox['y'] + bbox['h'] / 2
                
                # Convert screen coordinates to map tile coordinates
                screen_tile_x = int(grass_screen_x / 32)
                screen_tile_y = int((grass_screen_y - 16) / 32)
                
                # Convert to map tile coordinates (player is at 7, 7 in screen coords)
                map_tile_x = player_x + (screen_tile_x - 7)
                map_tile_y = player_y + (screen_tile_y - 7)
                
                filename = f"{map_tile_x}_{map_tile_y}.png"
                
                # Skip if already in cache
                if filename in existing_filenames:
                    continue
                
                # Convert from map tile to frame tile
                frame_tile_x = screen_tile_x
                frame_tile_y = screen_tile_y
                
                # Check if tile is in visible frame
                if frame_tile_x < 0 or frame_tile_x >= 15 or frame_tile_y < 0 or frame_tile_y >= 11:
                    continue
                
                # Crop tile (32x32 pixels)
                # After preprocessing to 480x352, row 0 starts at pixel 0 (includes 16px black padding)
                # Row 0 = pixels 0-31, Row 1 = pixels 32-63, etc.
                left = frame_tile_x * 32
                top = frame_tile_y * 32
                right = left + 32
                bottom = top + 32
                
                tile_img = frame_pil.crop((left, top, right, bottom))
                
                # Compute features
                features = compute_simple_tile_features(tile_img)
                
                # Save tile image
                tile_path = cache_dir / filename
                tile_img.save(tile_path)
                
                # Add to features data (flat format: "filename": [features])
                features_data[filename] = features.tolist()
                
                existing_filenames.add(filename)
                saved_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to save grass tile features: {e}")
                continue
        
        # Save features file
        if saved_count > 0:
            try:
                with open(features_path, 'w') as f:
                    json.dump(features_data, f, indent=2)
                logger.info(f"Saved features for {saved_count} grass tiles to navigation cache")
            except Exception as e:
                logger.error(f"Failed to save grass tiles features cache: {e}")
    
    def _mark_npc_tile_on_dialogue_start(self, state_data: dict, frame: np.ndarray, player_facing: str) -> None:
        """
        Mark the tile in front of the player as NPC when dialogue starts.
        
        This is called when dialogue_state_changed == True and current_dialogue == True.
        Delegates to navigation_agent's _mark_tile_as_npc method and logs the interaction.
        
        CRITICAL: Only marks tiles DIRECTLY IN FRONT of the player based on facing direction.
        This prevents incorrect marking of distant tiles.
        
        Args:
            state_data: Game state data
            frame: Current game frame
            player_facing: Player facing direction ('North', 'South', 'East', 'West')
        """
        try:
            # Get player position
            player_data = state_data.get('player', {})
            position = player_data.get('position', {})
            player_x = position.get('x')
            player_y = position.get('y')
            player_map = player_data.get('location', 'unknown')
            
            # Validate player position
            if player_x is None or player_y is None:
                logger.warning(f"Cannot mark NPC tile: invalid player position ({player_x}, {player_y})")
                return
            
            # Validate player facing direction
            if not player_facing or player_facing not in ['North', 'South', 'East', 'West']:
                logger.warning(f"Cannot mark NPC tile: invalid player facing '{player_facing}'")
                return
            
            # Calculate NPC tile position DIRECTLY IN FRONT based on facing direction
            facing_offset = {
                'North': (0, -1),
                'South': (0, 1),
                'East': (1, 0),
                'West': (-1, 0)
            }
            
            offset = facing_offset[player_facing]
            npc_tile_x = player_x + offset[0]
            npc_tile_y = player_y + offset[1]
            npc_tile = (npc_tile_x, npc_tile_y)
            
            # Validate that the NPC tile is adjacent to player (Manhattan distance = 1)
            distance = abs(npc_tile_x - player_x) + abs(npc_tile_y - player_y)
            if distance != 1:
                logger.error(f"NPC tile {npc_tile} is not adjacent to player ({player_x}, {player_y}) - distance={distance}. Skipping.")
                return
            
            logger.info(f"Dialogue started - marking NPC tile at {npc_tile} (player at ({player_x}, {player_y}), facing {player_facing})")
            
            # Update navigation agent state before marking tile
            self.nav_agent._update_state(state_data)
            
            # Delegate to navigation agent's method
            mark_and_save_tile(
                npc_tile, frame, 'npc', self.nav_agent.active_tile_index, player_map,
                allow_upgrade=False, player_map_tile_x=self.nav_agent.player_map_tile_x,
                player_map_tile_y=self.nav_agent.player_map_tile_y
            )
            
            # Log the interaction
            log_interaction(
                action='A',
                tile_pos=npc_tile,
                result='dialogue_started',
                is_npc=True,
                map_location=player_map,
                navigation_target=None
            )
            
        except Exception as e:
            logger.error(f"Failed to mark NPC tile on dialogue start: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
