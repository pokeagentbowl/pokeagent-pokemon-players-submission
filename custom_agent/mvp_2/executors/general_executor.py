"""General executor NT - handles menus, dialogues, and general interactions with dialogue history.

Uses VLM to decide actions and determine task completion/failure.
Includes loading of past completed dialogues for context.
Covers everything not handled by battle or navigation executors.
"""
import numpy as np
from typing import TYPE_CHECKING, List, Literal, Optional, Dict
from pydantic import BaseModel, Field

from custom_agent.mvp_2.executors.base_executor import BaseExecutor, ExecutorResult
from custom_utils.log_to_active import load_completed_dialogues
from custom_utils.detectors import detect_dialogue

if TYPE_CHECKING:
    from custom_agent.mvp_2.modules.perception import PerceptionResult
    from custom_utils.langchain_vlm import LangChainVLM


GENERAL_EXECUTOR_PROMPT_NT = """
You are controlling Pokemon Emerald to achieve a specific goal.

## Info
Current goal: {goal}
Scene: {scene_description}
Location: {location}
Party Info: {party_info}
Items: {items}
Money: {money}

### Dialog History
{has_dialog}

### Action History
{history_context}

## Instructions
Analyze the current situation and decide:
1. Should you continue working on this goal, or has it been completed/failed/interrupted?
2. Is there new information from the scene or dialogue that suggest new goals?
3. If continuing, what button(s) should you press next?

Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R

Common patterns:
- A: Confirm, talk, interact, advance dialogue
- B: Cancel, go back
- START: Open menu
- SELECT: Use registered item
- Directional: Navigate menus, select options
- L/R: Cycle through options in some menus

PAST LEARNINGS:
For naming menus, pressing START and then A exits the menu.
When choosing names, always choose the default to save time.
After closing bag in the middle of a battle, and encountering a black screen, press LEFT.
You MUST always fight. Keep battling until pokemon faints.
If needing to look for a clock, it is K in the traversability map.
To talk to someone, you need to face them one cell away and press A

IMPORTANT:
- If the goal is completed, set status='completed' and provide summary
- If the goal cannot be completed or failed, set status='failed' and explain why
- If something interrupted (e.g., battle started, unexpected event), set status='failed' and explain
- Otherwise, set status='in_progress' and provide button(s) to press
- When giving up control (completed/failed), return empty buttons list
- Review your action history to avoid repeating ineffective actions
- Use the completed dialogues context to understand what has already happened in the game
"""


class GeneralExecutorDecision(BaseModel):
    """VLM decision for general executor."""
    reasoning: str = Field(description="Step by step analysis of the current situation and plan")
    status: Literal['in_progress', 'completed', 'failed'] = Field(
        description="Whether to continue (in_progress) or give up control (completed/failed)"
    )
    buttons: List[Literal['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']] = Field(
        description="List of buttons to press (empty if status is completed/failed)",
        default_factory=list
    )
    summary: Optional[str] = Field(
        description="Explanation when giving up control (required for completed/failed)",
        default=None
    )


class GeneralExecutor(BaseExecutor):
    """
    General executor NT for menus, dialogue, and interactions with dialogue history.

    Covers anything not handled by navigation or battle executors.
    Uses VLM to decide both actions AND task completion/failure.
    Includes loading of past completed dialogues for context.

    Future: Split into specialized executors (menu, dialogue, interaction).
    """

    def __init__(self, reasoner: 'LangChainVLM'):
        """
        Initialize general executor NT with VLM reasoner.

        Args:
            reasoner: LangChainVLM instance for making decisions
        """
        super().__init__()
        self.reasoner = reasoner
        self.internal_state = {
            'max_actions': 5,  # Safety limit to prevent infinite loops
            'turn_count': 0,  # Number of execute_step calls
            'max_turns': 5,  # Turn limit (can be blank actions, so separate from max_actions)
            'action_history': []  # List of {turn, reasoning, actions} for short-term memory
        }

    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> ExecutorResult:
        """
        Execute general action (menu/dialogue/interaction).

        Uses VLM to decide button press AND whether to give up control.
        VLM determines completion/failure, not rule-based heuristics.

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Goal from planner (e.g., "Talk to Professor Birch", "Open the menu and save")

        Returns:
            ExecutorResult with actions and status
        """
        # Increment turn count at the start
        self.internal_state['turn_count'] += 1

        # Check turn limit (prevents getting stuck even with blank actions)
        if self.internal_state['turn_count'] > self.internal_state['max_turns']:
            self.reset()
            return ExecutorResult(
                actions=[],
                status='failed',
                summary=f"Turn limit ({self.internal_state['max_turns']}) reached without completion"
            )

        # Get VLM decision
        decision = self._decide_action_with_vlm(perception, state_data, goal)

        # Track this turn in action history
        self.internal_state['action_history'].append({
            'turn': self.internal_state['turn_count'],
            'reasoning': decision.reasoning,
            'actions': decision.buttons
        })

        # Keep only last 10 turns in history to avoid prompt bloat
        if len(self.internal_state['action_history']) > 10:
            self.internal_state['action_history'] = self.internal_state['action_history'][-10:]

        # Handle based on VLM's status decision
        if decision.status == 'completed':
            self.reset()
            return ExecutorResult(
                actions=[],
                status='completed',
                summary=decision.summary or "Task completed",
                reasoning=decision.reasoning  # Include reasoning for long-term memory
            )
        elif decision.status == 'failed':
            self.reset()
            return ExecutorResult(
                actions=[],
                status='failed',
                summary=decision.summary or "Task failed",
                reasoning=decision.reasoning  # Include reasoning for long-term memory
            )
        else:  # in_progress
            return ExecutorResult(
                actions=decision.buttons,
                status='in_progress',
                reasoning=decision.reasoning  # Include reasoning for long-term memory
            )

    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> bool:
        """
        General executor validity is context-dependent.

        Invalid if dialogue detected.

        Args:
            state_data: Game state data
            perception: Current perception result

        Returns:
            bool: True if valid, False if dialogue detected
        """
        # Check for dialogue
        frame = np.array(state_data.get('frame'))
        if detect_dialogue(frame):
            logger.info("General executor is invalid - dialogue detected")
            return False
        
        # MVP: Always valid otherwise. VLM decides when to give up control via status field.
        return True

    def reset(self):
        """Reset internal state (counters and history) for new goal."""
        self.internal_state['turn_count'] = 0
        self.internal_state['action_history'] = []

    def _decide_action_with_vlm(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> GeneralExecutorDecision:
        """
        Decide next button press using VLM with structured output and dialogue history.

        The VLM decides both the action AND whether to give up control.

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Current goal string

        Returns:
            GeneralExecutorDecision with reasoning, status, buttons, and optional summary
        """
        frame = np.array(state_data.get('frame'))

        # dialog detector here
        from custom_utils.detectors import detect_dialogue
        is_dialog = detect_dialogue(frame, threshold=0.48)
        
        # Load and format dialogue history
        dialogue_history = load_completed_dialogues()
        has_dialog = self._format_dialogue_history(is_dialog, dialogue_history)

        # Format action history for prompt
        history_context = self._format_action_history()

        prompt = GENERAL_EXECUTOR_PROMPT_NT.format(
            goal=goal,
            scene_description=perception.scene_description,
            has_dialog=has_dialog,
            location=state_data.get('player', {}).get('location', 'unknown'),
            party_info=state_data.get('player', {}).get('party', {}),
            items=state_data.get('game', {}).get('items', []),
            money=state_data.get('game', {}).get('money', "No Info"),
            history_context=history_context
        )

        # Call VLM with structured output
        # Note: We don't pass frame here since perception already contains scene_description
        # For MVP, we rely on perception's description. Future: pass frame for richer context
        decision = self.reasoner.call_vlm(
            prompt=prompt,
            image=frame,
            module_name="GENERAL_EXECUTOR_NT",
            structured_output_model=GeneralExecutorDecision
        )

        return decision

    def _format_action_history(self) -> str:
        """Format action history for prompt context."""
        if not self.internal_state['action_history']:
            return "Action history: (empty - this is the first turn)"

        history_lines = ["Action history (recent turns):"]
        for entry in self.internal_state['action_history']:
            turn = entry['turn']
            reasoning = entry['reasoning']
            actions = entry['actions'] if entry['actions'] else ['(no action)']
            actions_str = ', '.join(actions)
            history_lines.append(f"  Turn {turn}: {reasoning} → Actions: {actions_str}")

        return '\n'.join(history_lines)

    def _format_dialogue_history(self, is_current_dialog: bool, completed_dialogues: List[Dict]) -> str:
        """
        Format dialogue status and history for the prompt.
        
        Args:
            is_current_dialog: Whether there's currently an active dialogue
            completed_dialogues: List of completed dialogue dicts
            
        Returns:
            Formatted string for the has_dialog field
        """
        parts = []
        
        # Current dialogue status
        if is_current_dialog:
            parts.append("There is an active dialog box at the bottom of the screen.")
        else:
            parts.append("There is no dialog box currently visible.")
        
        # Add completed dialogues if any
        if completed_dialogues:
            parts.append("\nRecent completed dialogues:")
            for dialogue in completed_dialogues:
                coord = dialogue['coordinate']
                text_preview = dialogue['text'][:100] + "..." if len(dialogue['text']) > 100 else dialogue['text']
                parts.append(
                    f"  - Map: {dialogue['map']}, Pos: ({coord[0]}, {coord[1]}), Step: {dialogue['step']}: {text_preview}"
                )
        
        return "\n".join(parts)