"""Planner module - high-level strategic planning with unified VLM call.

Uses single VLM call with discriminated union for different executor types.
Navigation target selection is integrated into the plan decision.
"""

import numpy as np
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal, TYPE_CHECKING

from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult
from custom_utils.langchain_vlm import LangChainVLM
from custom_utils.navigation_targets import (
    NavigationTarget,
    format_targets_for_prompt,
    validate_and_select_target
)

if TYPE_CHECKING:
    from custom_agent.mvp_hierarchical.agent import AgentLogEntry

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

### WALKTHROUGH (follow all in order !!!IMPORTANT!!!)

### **Part 1: Starting Your Adventure**

1.  **Naming Your Character**: You will be asked for your name. Click on Start followed by A to choose a default name ASAP.

### **Part 2: Littleroot Town**

1.  **Arrival**: You will arrive in the back of a moving truck. Once it stops, exit the truck at the top right corner. Your Mom will greet you and lead you into your new home.
2.  **Set the Clock**: Once inside, your Mom will instruct you to go to your room. Go upstairs and walk over to the clock on the wall. Interact with it to set the time.
    *   **Instruction:** You must set the clock at (5, 1) to proceed with the game, set any time and confirm, ensure you are on the correct setting option before confirming.
3.  **Meet the Neighbors**: After setting the clock, go back downstairs. Your Mom will call you over to watch a TV program about your father, Norman, the Petalburg Gym Leader. After the program, she will tell you to go introduce yourself to Professor Birch's family next door.
4.  **Visit Professor Birch's House**: Exit your house and enter the house directly to your right. Speak with Professor Birch's wife. She will inform you that the Professor is out. Go upstairs to meet your rival, **May**. After a brief conversation, she will leave.

### **Part 3: Your First Pokémon and Battle**

1.  **Rescue Professor Birch**: Exit the house and walk north out of Littleroot Town onto **Route 101**. You will find Professor Birch being attacked by a wild Pokémon. He will ask for your help.
2.  **Choose Your Starter**: Walk up to the bag on the ground. You will be given a choice of three Pokémon. **You must select MUDKIP**, the Pokémon in the middle Poké Ball.

### **Part 4: The Journey to Oldale Town**

1.  **Receive Your Pokémon**: Professor Birch will thank you and take you to his Pokémon Lab. He will officially give you the **Mudkip** you used in the battle. When he asks if you want to give it a nickname, select **NO**.
2.  **Objective: Find May**: Professor Birch will ask you to find May, who is on Route 103, for a battle.
3.  **Level Up Your Mudkip**: Before finding May, you need to train your Mudkip. Leave the lab and walk north, through Oldale Town, and into the tall grass on **Route 103**.
    *   **Stuck Point/Help:** Your rival's Pokémon will be at Level 5. To ensure an easy victory, you should battle wild Pokémon until your Mudkip is at least **Level 6 or 7**.
    *   **Battle wild Pokémon**: You will encounter Pokémon like Wingull and Zigzagoon. Use **Tackle** to defeat them and gain experience. Do not run from these battles.
4.  **Heal Your Pokémon**: After battling a few wild Pokémon, your Mudkip's HP will be low. Return south to Oldale Town and enter the building with the red roof, the **Pokémon Center**. Talk to the nurse at the counter to restore your Mudkip to full health.

After healing, you are prepared to continue north on Route 103 to find and battle May.

### **Part 5: Rival Battle and Return to Littleroot**

1.  **Battle May**: Once your Mudkip is at least Level 7, walk north on Route 103 to find May. Talk to her to initiate your first rival battle.
    *   **Rival Battle**: May has a Level 5 **Treecko**. Use your **Tackle** attack to defeat it.
2.  **Return to the Lab**: After the battle, you and May will automatically return to Professor Birch's lab in Littleroot Town.
3.  **Receive the Pokédex**: As a reward for your victory, Professor Birch will give you a **Pokédex**, a device for recording data on Pokémon you've seen and caught. He will also give you **5 Poké Balls**.
4.  **Get the Running Shoes**: Exit the lab. As you walk out of town, your Mom will stop you and give you the **RUNNING SHOES**.

### **Part 6: Journey to Petalburg City**

1.  **Route 102 Trainers**: Head west from Oldale Town onto **Route 102**. You will encounter several trainers. You must battle them to proceed and gain experience.
    *   **Youngster Calvin**: He has a Level 5 **Zigzagoon**.
    *   **Bug Catcher Rick**: He has two Level 4 **Wurmples**. Your Mudkip will likely reach Level 8 after this battle.
    *   **Youngster Allen**: He has a Level 5 **Zigzagoon**.
    *   **Lass Tiana**: She has a Level 5 **Zigzagoon**.
2.  **Arrive in Petalburg City**: Continue heading west to enter **Petalburg City**.

### **Part 7: Petalburg City and Wally's Tutorial**

1.  **Visit the Petalburg Gym**: Heal your Pokémon at the Pokémon Center if needed, then enter the large building at the north end of the city, which is the **Petalburg Gym**.
2.  **Meet Your Father**: Inside, you'll find your father, **Norman**, the Gym Leader. During your conversation, a boy named **Wally** will enter.
3.  **Help Wally Catch a Pokémon**: Wally wants to catch his first Pokémon but doesn't know how. Norman will ask you to help him. You will automatically go with Wally to the tall grass on Route 102.
    *   **Catching Tutorial**: Wally will encounter a wild **Ralts** and use a Zigzagoon and a Poké Ball provided by your father to catch it. This is an automatic sequence.
4.  **Return to the Gym**: After the tutorial, you and Wally will return to the Gym. Norman will advise you that to become a strong trainer, you should travel to **Rustboro City** and challenge its Gym Leader, Roxanne.

### **Part 8: On to Rustboro City**

1.  **Route 104 and Petalburg Woods**: Exit the Gym and head west out of Petalburg City onto **Route 104**.
2.  **First Double Battle**: On the beach, you will find a house. Ignore it for now and continue north. You will soon enter **Petalburg Woods**.
3.  **Navigate the Woods**: Follow the path through the woods. You will encounter more trainers.
    *   **Bug Catcher Lyle**: He has two Level 3 **Wurmples**. Your Mudkip will likely reach Level 9 after this fight.
4.  **Team Aqua Encounter**: As you proceed, you will see a man in a blue uniform, a member of **Team Aqua**, confronting a researcher from the Devon Corporation. The researcher will ask for your help.
5.  **Battle Team Aqua Grunt**: Challenge the Team Aqua Grunt.
    *   **Trainer Battle**: He has a Level 9 **Poochyena**. Defeat it with your Mudkip. Your Mudkip should reach Level 10 and learn the move **WATER GUN**.
6.  **Receive the Great Ball**: After the battle, the researcher will thank you by giving you a **GREAT BALL**. He will then leave.
7.  **Exit the Woods**: Continue following the path north to exit Petalburg Woods and arrive back on Route 104.

### **Part 9: Rustboro City and the First Gym**

1.  **Arrival in Rustboro**: Follow the path north, past the **Pretty Petal flower shop**, and across a large bridge. Battle the trainers on the bridge for more experience. You will then arrive in **Rustboro City**.
2.  **Prepare for the Gym**: Heal your Pokémon at the Pokémon Center. The Rustboro Gym specializes in Rock-type Pokémon, which are weak against Water-type moves. Your Mudkip, now knowing **Water Gun**, is perfectly suited for this challenge. Ensure it is at least **Level 12** before proceeding.
3.  **Enter the Rustboro Gym**: The Gym is located northeast of the Pokémon Center. Inside, you will find a maze with two optional trainers. It is recommended to battle them for experience.
    *   **Youngster Josh**: Has a Level 10 **Geodude**.
    *   **Youngster Tommy**: Has two Level 8 **Geodudes**.
4.  **Challenge the Gym Leader**: After defeating the trainers, walk to the back of the Gym to challenge the leader, **Roxanne**.
    *   **Gym Battle vs. Roxanne**: Roxanne has two Pokémon: a Level 12 **Geodude** and a Level 15 **Nosepass**.
    *   **Strategy**: Use **Water Gun** with your Mudkip. It is a super-effective move that will likely defeat both of her Pokémon in a single hit.
5.  **Victory and Rewards**: After defeating Roxanne, you will receive the **Stone Badge** and **TM39**, which contains the move **Rock Tomb**.

## Tips
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
        from custom_agent.mvp_hierarchical.agent import format_logs_as_numbered_list

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

        prompt = PLANNER_PROMPT.format(
            recent_logs=recent_logs_str,
            player_map=player_map,
            # objects=objects_str,
            navigation_targets_section=navigation_targets_section,
            # memories=memories_str,
            milestones=milestones,
            player_map_tile_x=player_map_tile_x,
            player_map_tile_y=player_map_tile_y
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
