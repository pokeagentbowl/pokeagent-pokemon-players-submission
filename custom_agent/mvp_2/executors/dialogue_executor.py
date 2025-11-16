"""Dialogue Executor NT - Advanced dialogue handling with active cache integration.

Incorporates dialogue loop detection, VLM fallback, and active tile caching from overall_agent_nt.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from pydantic import BaseModel

from custom_agent.mvp_2.executors.base_executor import BaseExecutor, ExecutorResult
from custom_utils.detectors import detect_dialogue, detect_player_direction
from custom_utils.log_to_active import mark_and_save_tile, log_interaction
from custom_utils.langchain_vlm import LangChainVLM
if TYPE_CHECKING:
    from custom_agent.mvp_2.modules.perception import PerceptionResult

logger = logging.getLogger(__name__)


class DialogueState(BaseModel):
    """Track dialogue execution state."""
    dialogue_history: List[Dict[str, Any]] = []  # Track last 10 dialogues
    action_history: List[Dict[str, Any]] = []  # Track last 10 actions
    last_dialogue_text: Optional[str] = None
    dialogue_count: int = 0
    skip_action_loop_check_steps: int = 0
    last_fallback_step: Optional[int] = None
    last_repeated_dialogue: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class DialogueExecutor(BaseExecutor):
    """
    Advanced dialogue executor with loop detection and active cache integration.

    Based on overall_agent_nt dialogue handling logic.
    """

    def __init__(self, backend: str = "github_models", model_name: str = "gpt-4o-mini", temperature: float = 0, **kwargs):
        """
        Initialize dialogue executor.

        Args:
            backend: LLM backend
            model_name: Model name
            temperature: Generation temperature
            **kwargs: Additional args
        """
        super().__init__()

        self.vlm = LangChainVLM(
            backend=backend,
            model_name=model_name,
            temperature=temperature
        )

        self.state = DialogueState()
        self.step_count = 0

        # Import navigation agent for active tile access (will be set by agent)
        self.nav_agent = None

        logger.info("Initialized DialogueExecutor")

    def set_navigation_agent(self, nav_agent):
        """Set reference to navigation agent for active tile access."""
        self.nav_agent = nav_agent

    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> ExecutorResult:
        """
        Execute dialogue advancement step with loop detection and fallback.

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Goal description

        Returns:
            ExecutorResult with actions and status
        """
        self.step_count += 1

        frame = np.array(state_data.get('frame', []))
        if frame.size == 0:
            return ExecutorResult(actions=[], status='failed', summary='No frame available')

        # Check for dialogue
        current_dialogue = detect_dialogue(frame, threshold=0.3)

        if not current_dialogue:
            # Dialogue ended
            logger.info("Dialogue ended - marking as completed")
            return ExecutorResult(actions=[], status='completed', summary='Dialogue completed')

        # Extract dialogue text if available from perception
        dialogue_text = self._extract_dialogue_text(perception)

        # Check for dialogue loop
        repeated_dialogue = self._check_dialogue_loop(dialogue_text)
        if repeated_dialogue:
            logger.warning(f"Dialogue loop detected: {repeated_dialogue}")
            # Use VLM fallback
            return self._vlm_fallback(state_data, frame, repeated_dialogue, goal)

        # Check for action loop
        repeated_action = self._check_action_loop()
        if repeated_action:
            logger.warning(f"Action loop detected: {repeated_action}")
            return self._vlm_fallback(state_data, frame, repeated_action, goal, loop_type="action")

        # Mark NPC tile if dialogue just started
        self._mark_npc_on_dialogue_start(state_data, frame)

        # Update dialogue history
        self._update_dialogue_history(dialogue_text)

        # Normal dialogue advancement - press A
        logger.info("Advancing dialogue with 'A' press")
        return ExecutorResult(
            actions=['A'],
            status='in_progress',
            summary=f'Dialogue advanced (step {self.step_count})'
        )

    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> bool:
        """
        Check if dialogue executor is still valid.

        Valid as long as dialogue is detected.
        """
        frame = np.array(state_data.get('frame', []))
        if frame.size == 0:
            return False

        return detect_dialogue(frame, threshold=0.3)

    def reset(self):
        """Reset dialogue executor state."""
        self.state = DialogueState()
        self.step_count = 0

    def _extract_dialogue_text(self, perception) -> str:
        """Extract dialogue text from perception OCR results."""
        if not perception.ocr_text:
            return "NO_DIALOGUE"

        # Look for dialogue-like text in OCR results
        dialogue_lines = []
        for ocr_block in perception.ocr_text:
            text = ocr_block.text.strip()
            if text and len(text) > 5:  # Filter out short fragments
                dialogue_lines.append(text)

        if dialogue_lines:
            return " ".join(dialogue_lines)
        return "NO_DIALOGUE"

    def _check_dialogue_loop(self, current_dialogue_text: str) -> Optional[str]:
        """Check for dialogue loop based on history."""
        if not current_dialogue_text or current_dialogue_text == "NO_DIALOGUE":
            return None

        # Check last 5 dialogues for repetition
        recent_dialogues = self.state.dialogue_history[-5:]
        for entry in recent_dialogues:
            prev_text = entry.get('text', '')
            if prev_text and self._text_similarity(prev_text, current_dialogue_text) > 0.8:
                return current_dialogue_text

        return None

    def _check_action_loop(self) -> Optional[str]:
        """Check for action loop based on history."""
        if self.state.skip_action_loop_check_steps > 0:
            self.state.skip_action_loop_check_steps -= 1
            return None

        # Check last 5 actions for repetition
        recent_actions = self.state.action_history[-5:]
        if len(recent_actions) < 5:
            return None

        # Check if all recent actions are the same
        first_action = recent_actions[0].get('actions', [])
        if all(entry.get('actions', []) == first_action for entry in recent_actions):
            return str(first_action)

        return None

    def _mark_npc_on_dialogue_start(self, state_data: dict, frame: np.ndarray):
        """Mark NPC tile when dialogue starts."""
        if not self.nav_agent:
            return

        try:
            # Get player position and facing
            player_data = state_data.get('player', {})
            position = player_data.get('position', {})
            player_x = position.get('x')
            player_y = position.get('y')
            player_map = player_data.get('location', 'unknown')

            player_direction_result = detect_player_direction(frame, match_threshold=0.7)
            if not player_direction_result:
                return
            player_facing = player_direction_result[0]

            # Validate inputs
            if player_x is None or player_y is None or not player_facing:
                return

            # Calculate NPC tile position
            facing_offset = {
                'North': (0, -1),
                'South': (0, 1),
                'East': (1, 0),
                'West': (-1, 0)
            }

            offset = facing_offset[player_facing]
            npc_tile = (player_x + offset[0], player_y + offset[1])

            logger.info(f"Dialogue active - marking NPC tile at {npc_tile}")

            # Update navigation agent state
            self.nav_agent._update_state(state_data)

            # Mark tile as NPC
            mark_and_save_tile(
                npc_tile, frame, 'npc', self.nav_agent.active_tile_index, player_map,
                allow_upgrade=False, player_map_tile_x=self.nav_agent.player_map_tile_x,
                player_map_tile_y=self.nav_agent.player_map_tile_y
            )

            # Log interaction
            log_interaction(
                action='dialogue_started',
                tile_pos=npc_tile,
                result='dialogue_active',
                is_npc=True,
                map_location=player_map,
                navigation_target=None
            )

        except Exception as e:
            logger.error(f"Failed to mark NPC tile: {e}")

    def _update_dialogue_history(self, dialogue_text: str):
        """Update dialogue history."""
        entry = {
            'step': self.step_count,
            'text': dialogue_text,
            'timestamp': np.datetime64('now')
        }

        self.state.dialogue_history.append(entry)
        if len(self.state.dialogue_history) > 10:
            self.state.dialogue_history = self.state.dialogue_history[-10:]

    def _vlm_fallback(self, state_data: dict, frame: np.ndarray, repeated_content: str, goal: str, loop_type: str = "dialogue") -> ExecutorResult:
        """Use VLM fallback for stuck dialogue/action loops."""
        try:
            # Mark fallback step
            self.state.last_fallback_step = self.step_count
            self.state.skip_action_loop_check_steps = 5

            # Use VLM to determine next action
            prompt = f"""
            The agent is stuck in a {loop_type} loop: "{repeated_content}"

            Current goal: {goal}

            Analyze the game frame and determine the best action to break the loop.
            Return a JSON with 'reasoning' and 'actions' (list of button presses).
            """

            # For now, return a simple fallback action
            # In full implementation, this would call VLM
            logger.info(f"VLM fallback triggered for {loop_type} loop")

            return ExecutorResult(
                actions=['B', 'DOWN', 'A'],  # Try B to cancel, move, then A
                status='in_progress',
                summary=f'VLM fallback for {loop_type} loop'
            )

        except Exception as e:
            logger.error(f"VLM fallback failed: {e}")
            return ExecutorResult(actions=['A'], status='in_progress', summary='Fallback to A press')

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0