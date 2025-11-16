"""Planner module - high-level strategic planning with unified VLM call.

Uses single VLM call with discriminated union for different executor types.
Navigation target selection is integrated into the plan decision.
"""

import numpy as np
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal, TYPE_CHECKING

from custom_agent.mvp_2.modules.perception import PerceptionResult
from custom_utils.langchain_vlm import LangChainVLM
from custom_utils.navigation_targets import (
    NavigationTarget,
    format_targets_for_prompt,
    validate_and_select_target
)
from custom_utils.log_to_active import load_completed_dialogues

if TYPE_CHECKING:
    from custom_agent.mvp_2.agent import AgentLogEntry

logger = logging.getLogger(__name__)


def format_memories_for_context(memories: List) -> str:
    """
    Format memory entries into a comprehensive string for LLM context.

    Formats each memory with:
    - Timestep (step number)
    - Planner's reasoning and decision (if available)
    - Executor's thoughts (if available)
    - Actions taken

    Args:
        memories: List of MemoryEntry objects to format

    Returns:
        Formatted string with all memory details
    """
    if not memories:
        return "No relevant memories"

    formatted_lines = []
    for mem in memories:
        # Start with timestep
        memory_parts = [f"Step {mem.step_number}:"]

        # Add planner reasoning and decision if available
        llm_outputs = mem.llm_outputs or {}
        planner_reasoning = llm_outputs.get('planner_reasoning')
        planner_executor = llm_outputs.get('planner_executor_choice')
        planner_goal = llm_outputs.get('planner_goal')

        if planner_reasoning or planner_executor or planner_goal:
            planner_info = []
            if planner_reasoning:
                planner_info.append(f"Reasoning: {planner_reasoning}")
            if planner_executor:
                planner_info.append(f"Chose: {planner_executor}")
            if planner_goal:
                planner_info.append(f"Goal: {planner_goal}")
            memory_parts.append(f"  Planner - {', '.join(planner_info)}")

        # Add executor thoughts if available
        # General executor stores reasoning directly
        executor_reasoning = llm_outputs.get('reasoning')
        if executor_reasoning:
            memory_parts.append(f"  Executor thoughts: {executor_reasoning}")

        # Add actions taken
        if mem.actions:
            actions_str = ', '.join(mem.actions)
            memory_parts.append(f"  Actions: {actions_str}")
        else:
            memory_parts.append(f"  Actions: none")

        # Add detected objects for context
        detected_obj_names = [obj.name for obj in mem.perception.detected_objects]
        if detected_obj_names:
            memory_parts.append(f"  Objects: {', '.join(detected_obj_names)}")

        # Join all parts with newlines
        formatted_lines.append('\n'.join(memory_parts))

    return '\n\n'.join(formatted_lines)


def format_dialogue_history_for_planner(completed_dialogues: List[Dict]) -> str:
    """
    Format completed dialogues for the planner prompt.
    
    Args:
        completed_dialogues: List of completed dialogue dicts
        
    Returns:
        Formatted string for the dialogue_history field
    """
    if not completed_dialogues:
        return "No recent completed dialogues."
    
    parts = ["Recent completed dialogues:"]
    for dialogue in completed_dialogues:
        coord = dialogue['coordinate']
        text_preview = dialogue['text'][:150] + "..." if len(dialogue['text']) > 150 else dialogue['text']
        parts.append(
            f"  - Map: {dialogue['map']}, Pos: ({coord[0]}, {coord[1]}), Step: {dialogue['step']}: {text_preview}"
        )
    
    return "\n".join(parts)


# Prompt template
# TODO: navigation targets is a superset of objects
# TODO: FUTURE: social directedness as dedicated reasoning module
PLANNER_PROMPT = """
You are a high-level planner for a Pokemon Emerald agent.
Your overarching goal is to progress through the game
(beat gyms, advance story, explore).

## Info
Current map: {player_map}
You are the person at ({player_map_tile_x}, {player_map_tile_y}) of size 1x1.
The screen is of size 15x10 (width x height).

## MILESTONES
{milestones}

### DETECTED OBJECTS (POTENTIALLY NAVIGATION TARGETS)
{navigation_targets_section}

Note: Targets marked as [REACHABLE] have a clear path from your position.
Targets marked as [UNREACHABLE] are blocked by obstacles, but you can still select them
to attempt getting as close as possible (e.g., to approach objects behind temporary obstacles).

### RECENT ACTIONS (last 20)
{recent_logs}

### DIALOGUE HISTORY
{dialogue_history}

### WALKTHROUGH
Head North from Littleroot Town via Route 101 to Oldale Town. 
Head North from Oldale Town to Route 103 to fight rival at top.
After rival is fought, head south to Oldale Town.
In Oldale Town, go to Birch's lab to receive Pokedex and Pokeballs.
Then, head West from Oldale Town via Route 102 to Petalburg City.
Then, head West then North from Petalburg City via Route 104 to Rustboro City.
In Rustboro City, go to the Gym and challenge the Gym Leader, Roxanne.

If in buildings and want to exit, use the door / staircase (D / S)
If you keep losing battles, you should go level up your pokemon in wild battles.
When in a new town, head to the Pokemon Centre to heal your pokemon (find the nurse at the centre of the first level)
When choosing names, always choose the default to save time.

## Instructions
Based on the above, think step by step (reasoning):
- What is the current high level goal and steps to achieve it (plan)?
- Then for the most immediate next step, choose the next high-level action and which executor to use.
- If previous actions have failed, try a different action
- If a previous navigation target has failed, try a different one to see if it works

Available executors:
- navigation: Move to specific locations (if targets are available above, choose one by index)
    - NOTE: this only moves the player. It does not perform any interactions. Use the general executor for interactions.
- general: Handle Pokemon battles, Navigate menus, advance dialogue, press specific buttons, interact with objects

IMPORTANT:
- Always put your reasoning first
- Prioritise navigation unless in battle or advancing dialog (choose general)
- Navigate one square at a time in the Pokemon Centre and in gyms using general executor.
- If choosing navigation, specify the target_index from the available targets above
- If target you would want to navigate to is blocked by an NPC, use general executor to move around the NPC instead
- If choosing general, specify the goal (meant for the executor to follow) in natural language
- The goal should be as simple as possible, avoid chaining multiple instructions in one.
- If there are multiple goals, only do the first immediate one
- Prefer REACHABLE targets when possible, but UNREACHABLE targets can still be selected to get close
"""


# Structured output models (properly validated with discriminated union)
class NavigationPlan(BaseModel):
    """Plan to navigate to a specific target."""
    reasoning: str = Field(description="Reasoning for choosing navigation")
    executor_type: Literal['navigation'] = 'navigation'
    target_index: int = Field(description="Index of chosen navigation target (0-based)")


class OtherExecutorPlan(BaseModel):
    """Plan for battle or general executor."""
    reasoning: str = Field(description="Reasoning for the plan")
    executor_type: Literal['general'] = Field(description="Which executor to use")
    goal: str = Field(description="High-level goal in natural language")


# Wrapper model to avoid LangChain's Union issue
# LangChain doesn't support Union directly, but works with a model containing a Union field
# Note: Do NOT use discriminator else error!
class PlanDecisionResponse(BaseModel):
    """Response wrapper for plan decision.

    Wraps the Union in a field to make it compatible with LangChain's structured output.
    """
    decision: Union[NavigationPlan, OtherExecutorPlan] = Field(
        description="The plan decision (either navigation or other executor plan)"
    )


class PlanResult(BaseModel):
    """Result of planning."""
    executor_type: str
    goal: Any  # str for general/battle, dict for navigation (with 'target' and 'description')
    reasoning: str

    class Config:
        arbitrary_types_allowed = True


class Planner:
    """
    High-level planner that decides goals and delegates to executors.

    Uses single unified VLM call with discriminated union for different executor types.
    Memory retrieval uses image-based similarity.
    """

    def __init__(self, reasoner: LangChainVLM, memory):
        """
        Initialize planner.

        Args:
            reasoner: LangChainVLM instance for VLM calls
            memory: EpisodicMemory instance for retrieval
        """
        self.reasoner = reasoner
        self.memory = memory
        logger.info("Initialized Planner")

    def create_plan(
        self,
        perception: PerceptionResult,
        state_data: dict,
        recent_logs: List['AgentLogEntry']
    ) -> PlanResult:
        """
        Create a new plan using single VLM call.

        Args:
            perception: Current perception (includes navigation targets)
            state_data: Game state (includes frame)
            recent_logs: Recent structured log entries

        Returns:
            PlanResult with executor_type, goal, reasoning
        """
        from custom_agent.mvp_2.agent import format_logs_as_numbered_list

        frame = np.array(state_data.get('frame'))

        # Retrieve relevant memories (uses image-based similarity)
        # logger.info("Retrieving relevant memories")
        # relevant_memories = self.memory.retrieve(
        #     query_image=frame,
        #     top_k=3
        # )
        # logger.info(f"Retrieved {len(relevant_memories)} relevant memories")

        # Format inputs for prompt (use last 20 entries)
        recent_logs_str = format_logs_as_numbered_list(recent_logs, max_entries=20)

        # objects_str = ", ".join([
        #     obj.name for obj in perception.detected_objects
        # ]) if perception.detected_objects else "No objects detected"

        # Format memories using local function
        # memories_str = format_memories_for_context(relevant_memories)

        player_data: dict = state_data.get('player', {})
        position: dict = player_data.get('position')
        player_map = player_data.get('location')
        player_map_tile_x = position.get('x')
        player_map_tile_y = position.get('y')
        milestones = state_data.get('milestones', {})
        
        # Format navigation targets if available
        targets = perception.navigation_targets
        if targets:
            formatted_targets = format_targets_for_prompt(targets)
            navigation_targets_section = f"""AVAILABLE NAVIGATION TARGETS:
{formatted_targets}"""
            logger.info(f"Formatted {len(targets)} navigation targets for prompt")
        else:
            navigation_targets_section = "No navigation targets available."
            logger.info("No navigation targets available")

        # Load and format dialogue history
        completed_dialogues = load_completed_dialogues()
        dialogue_history = format_dialogue_history_for_planner(completed_dialogues)

        prompt = PLANNER_PROMPT.format(
            recent_logs=recent_logs_str,
            player_map=player_map,
            # objects=objects_str,
            navigation_targets_section=navigation_targets_section,
            # memories=memories_str,
            milestones=milestones,
            player_map_tile_x=player_map_tile_x,
            player_map_tile_y=player_map_tile_y,
            dialogue_history=dialogue_history
        )

        # Call VLM with structured output (wrapped Union)
        logger.info("Calling VLM for plan decision")
        response: PlanDecisionResponse = self.reasoner.call_vlm(
            prompt=prompt,
            image=frame,
            module_name="PLANNER",
            structured_output_model=PlanDecisionResponse
        )

        # Extract the actual decision from the wrapper
        decision = response.decision

        # Build result based on decision type
        if isinstance(decision, NavigationPlan):
            logger.info(f"VLM chose navigation executor, target_index={decision.target_index}")
            # Validate and select target
            if targets:
                try:
                    chosen_target = validate_and_select_target(decision.target_index, targets)
                    logger.info(f"Selected target: {chosen_target.description}")
                    goal = {
                        'target': chosen_target,
                        'description': f"Navigate to {chosen_target.description}"
                    }
                except ValueError as e:
                    logger.error(f"Target validation failed: {e}")
                    # Fallback: no target
                    goal = {
                        'target': None,
                        'description': "Navigation requested but target validation failed"
                    }
            else:
                # No targets available (shouldn't happen if VLM chose navigation, but handle gracefully)
                logger.warning("VLM chose navigation but no targets available")
                goal = {
                    'target': None,
                    'description': "Navigation requested but no targets available"
                }

            return PlanResult(
                executor_type='navigation',
                goal=goal,
                reasoning=decision.reasoning
            )
        else:
            # OtherExecutorPlan (battle or general)
            logger.info(f"VLM chose {decision.executor_type} executor")
            logger.info(f"Goal: {decision.goal}")
            return PlanResult(
                executor_type=decision.executor_type,
                goal=decision.goal,
                reasoning=decision.reasoning
            )
