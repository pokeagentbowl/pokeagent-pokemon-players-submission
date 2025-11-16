"""Main MVP Hierarchical Agent - orchestrates perception, saliency, memory, planner, and executors."""

import logging
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from custom_agent.base_agent import AgentRegistry
from custom_agent.base_langchain_agent import BaseLangChainAgent
from custom_utils.langchain_vlm import LangChainVLM

from custom_agent.mvp_hierarchical.modules.perception import PerceptionModule, PerceptionResult
from custom_agent.mvp_hierarchical.modules.saliency import SaliencyDetector
from custom_agent.mvp_hierarchical.modules.memory import EpisodicMemory
from custom_agent.mvp_hierarchical.modules.planner import Planner
from custom_agent.mvp_hierarchical.executors.navigation_executor import NavigationExecutor
from custom_agent.mvp_hierarchical.executors.battle_executor import BattleExecutor
from custom_agent.mvp_hierarchical.executors.general_executor import GeneralExecutor
from custom_agent.mvp_hierarchical.executors.base_executor import ExecutorResult, BaseExecutor

logger = logging.getLogger(__name__)


class AgentLogEntry(BaseModel):
    """Structured log entry for agent actions."""
    planner_reasoning: Optional[str] = None  # Planner's reasoning (if planner ran)
    executor_type: str  # Which executor was chosen
    goal: str  # Goal description
    outcome: Optional[str] = None  # Final outcome (completed/failed) with summary

    def format_as_text(self) -> str:
        """Format log entry as human-readable text."""
        parts = []
        if self.planner_reasoning:
            parts.append(f"Plan: {self.planner_reasoning}")
        parts.append(f"Executor: {self.executor_type}")
        parts.append(f"Goal: {self.goal}")
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
        return " | ".join(parts)


class AgentState(BaseModel):
    """Tracks current agent state."""
    current_executor: Optional[BaseExecutor] = None  # Current executor instance
    current_executor_type: Optional[str] = None  # Executor type string
    current_goal: Optional[Any] = None  # Goal (str for general/battle, dict for navigation)
    suspended_goals: List[Dict[str, Any]] = []  # Stack of suspended goals
    recent_logs: List[AgentLogEntry] = []  # Recent structured logs (last 5 by default)
    current_log_entry: Optional[AgentLogEntry] = None  # Log entry being built for current execution

    class Config:
        arbitrary_types_allowed = True


def format_logs_as_numbered_list(logs: List[AgentLogEntry], max_entries: int = 5) -> str:
    """
    Format structured logs as numbered list.

    Args:
        logs: List of AgentLogEntry objects
        max_entries: Maximum number of entries to include (default 5)

    Returns:
        Formatted string with numbered list
    """
    if not logs:
        return "No recent actions"

    # Take last N entries
    recent_logs = logs[-max_entries:]

    formatted_lines = []
    for idx, log_entry in enumerate(recent_logs, start=1):
        formatted_lines.append(f"{idx}. {log_entry.format_as_text()}")

    return "\n".join(formatted_lines)


@AgentRegistry.register("mvp_hierarchical")
class MVPHierarchicalAgent(BaseLangChainAgent):
    """
    Hierarchical agent with perception, saliency, memory, planner, and executors.

    Main orchestrator that coordinates all modules and executors.
    """

    def __init__(
        self,
        backend: str = "github_models",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        action_batch_size: int = 100,
        max_log_entries: int = 5,
        **kwargs
    ):
        """
        Initialize agent with all modules.

        Args:
            backend: LLM backend type
            model_name: Name of the model
            temperature: Temperature for generation
            action_batch_size: Number of actions per step for executors
            max_log_entries: Maximum number of log entries to keep (default 5)
            **kwargs: Additional arguments passed to BaseLangChainAgent
        """
        # Initialize parent class (sets up vlm_helper for backwards compatibility)
        super().__init__(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )

        # 1. Initialize memory first
        self.memory = EpisodicMemory()

        # 2. Initialize shared reasoner (new schema, separate from vlm_helper)
        self.reasoner = LangChainVLM(
            backend=backend,
            model_name=model_name,
            temperature=temperature
        )

        # 3. Initialize modules
        self.perception = PerceptionModule(reasoner=self.reasoner)
        self.saliency = SaliencyDetector()
        self.planner = Planner(reasoner=self.reasoner, memory=self.memory)

        # 4. Initialize executors
        self.executors: Dict[str, BaseExecutor] = {
            'navigation': NavigationExecutor(action_batch_size=action_batch_size),
            'battle': BattleExecutor(),
            'general': GeneralExecutor(reasoner=self.reasoner)
        }

        # 5. Agent state
        self.agent_state = AgentState()
        self.step_count = 0
        self.max_log_entries = max_log_entries

        # 6. Track LLM outputs for current step (for memory storage)
        self.current_step_llm_outputs = {}

        logger.info("Initialized MVPHierarchicalAgent with all modules and executors")

    def step(self, game_state: dict) -> dict:
        """
        Main decision loop.

        Flow:
        1. Perception
        2. Saliency check
        3. Decision (planner if needed)
        4. Execute current plan
        5. Memory storage
        6. Return actions

        Args:
            game_state: Dict containing 'frame' and other game data

        Returns:
            dict: {'action': actions, 'reasoning': reasoning}
        """
        self.step_count += 1

        # Clear LLM outputs from previous step
        self.current_step_llm_outputs = {}

        # 1. Perception (always first)
        perception_result = self.perception.process(game_state)

        # Collect perception LLM outputs
        if perception_result.llm_outputs:
            self.current_step_llm_outputs.update(perception_result.llm_outputs)

        # 2. Saliency detection
        saliency_result = self.saliency.check_validity(
            current_executor=self.agent_state.current_executor,
            state_data=game_state,
            perception=perception_result
        )

        # 3. Decision logic
        if not saliency_result.executor_valid and self.agent_state.current_executor is not None:
            # Executor became invalid - suspend current goal
            self._suspend_current_goal()
            self._create_new_plan(perception_result, game_state)
        elif self.agent_state.current_executor is None:
            # No current executor - check for suspended goals or create new plan
            if self.agent_state.suspended_goals:
                self._resume_suspended_goal()
            else:
                self._create_new_plan(perception_result, game_state)

        # 4. Execute current plan (unified execution point)
        actions = self._execute_current_plan(perception_result, game_state)

        # 5. Memory storage
        self._store_to_memory(game_state, perception_result, actions)

        # Format reasoning for return (extract description from dict goals)
        reasoning = self.agent_state.current_goal
        if isinstance(reasoning, dict):
            reasoning = reasoning.get('description', '')

        return {'action': actions, 'reasoning': reasoning or ''}

    def _create_new_plan(self, perception_result, state_data):
        """Create new plan from planner."""
        plan = self.planner.create_plan(
            perception=perception_result,
            state_data=state_data,
            recent_logs=self.agent_state.recent_logs
        )

        # Update agent state
        self.agent_state.current_executor_type = plan.executor_type
        self.agent_state.current_executor = self.executors[plan.executor_type]
        self.agent_state.current_goal = plan.goal

        # Extract goal string
        # Reset executor to clean state for new goal
        self.agent_state.current_executor.reset()

        # Capture planner reasoning in logs and LLM outputs
        goal_str = plan.goal.get('description', plan.goal) if isinstance(plan.goal, dict) else plan.goal

        # Create structured log entry with planner info
        self.agent_state.current_log_entry = AgentLogEntry(
            planner_reasoning=plan.reasoning,
            executor_type=plan.executor_type,
            goal=goal_str,
            outcome=None  # Will be set when executor completes
        )

        # Collect planner LLM outputs
        self.current_step_llm_outputs['planner_reasoning'] = plan.reasoning
        self.current_step_llm_outputs['planner_executor_choice'] = plan.executor_type
        self.current_step_llm_outputs['planner_goal'] = goal_str

        logger.info(f"New plan: {goal_str} (executor: {plan.executor_type})")
        logger.info(f"Reasoning: {plan.reasoning}")

    def _execute_current_plan(self, perception_result, state_data) -> List[str]:
        """Execute current executor's step."""
        executor = self.agent_state.current_executor

        if executor is None:
            logger.warning("No current executor, returning empty actions")
            return []

        result = executor.execute_step(
            perception=perception_result,
            state_data=state_data,
            goal=self.agent_state.current_goal
        )

        # Capture executor reasoning for long-term memory (if available)
        if result.reasoning:
            executor_type = self.agent_state.current_executor_type
            self.current_step_llm_outputs[f'{executor_type}_executor_reasoning'] = result.reasoning

        # Check if executor completed/failed
        if result.status in ['completed', 'failed']:
            self._handle_executor_completion(result)

        return result.actions

    def _suspend_current_goal(self):
        """Suspend current goal and executor state (for interruption handling)."""
        if self.agent_state.current_executor is None:
            return

        # Save current executor state
        suspended_goal = {
            'executor_type': self.agent_state.current_executor_type,
            'goal': self.agent_state.current_goal,
            'executor_state': self.agent_state.current_executor.get_state()
        }

        self.agent_state.suspended_goals.append(suspended_goal)

        # Reset executor instance to clean state (suspended state is saved above)
        self.agent_state.current_executor.reset()

        # Log with appropriate format
        goal_str = self.agent_state.current_goal
        if isinstance(goal_str, dict):
            goal_str = goal_str.get('description', goal_str)
        logger.info(f"Suspended goal: {goal_str}")

    def _resume_suspended_goal(self):
        """Resume most recently suspended goal."""
        if not self.agent_state.suspended_goals:
            logger.warning("Attempted to resume suspended goal but stack is empty")
            return

        # Pop most recent suspended goal
        suspended = self.agent_state.suspended_goals.pop()

        # Restore executor and state
        self.agent_state.current_executor_type = suspended['executor_type']
        self.agent_state.current_executor = self.executors[suspended['executor_type']]
        self.agent_state.current_executor.restore_state(suspended['executor_state'])
        self.agent_state.current_goal = suspended['goal']

        # Extract goal string
        goal_str = self.agent_state.current_goal
        if isinstance(goal_str, dict):
            goal_str = goal_str.get('description', goal_str)

        # Create log entry for resumed goal (no planner reasoning since we're resuming)
        self.agent_state.current_log_entry = AgentLogEntry(
            planner_reasoning=None,  # Resumed goal, not from planner
            executor_type=suspended['executor_type'],
            goal=f"RESUMED: {goal_str}",
            outcome=None  # Will be set when executor completes
        )

        logger.info(f"Resumed goal: {goal_str}")

    def _handle_executor_completion(self, result: ExecutorResult):
        """Handle executor completion or failure."""
        # Extract goal string
        goal_str = self.agent_state.current_goal
        if isinstance(goal_str, dict):
            goal_str = goal_str.get('description', goal_str)

        # Create or update log entry with outcome
        if self.agent_state.current_log_entry is not None:
            # Update existing log entry (from planner) with outcome
            self.agent_state.current_log_entry.outcome = result.summary or f"{result.status}: {goal_str}"
        else:
            # Create new log entry (executor ran without planner)
            self.agent_state.current_log_entry = AgentLogEntry(
                planner_reasoning=None,  # No planner reasoning
                executor_type=self.agent_state.current_executor_type,
                goal=goal_str,
                outcome=result.summary or f"{result.status}: {goal_str}"
            )

        # Add completed log entry to recent logs
        self.agent_state.recent_logs.append(self.agent_state.current_log_entry)

        # Keep only recent logs (configurable, default 5)
        if len(self.agent_state.recent_logs) > self.max_log_entries:
            self.agent_state.recent_logs = self.agent_state.recent_logs[-self.max_log_entries:]

        # Clear current executor and log entry
        self.agent_state.current_executor = None
        self.agent_state.current_executor_type = None
        self.agent_state.current_goal = None
        self.agent_state.current_log_entry = None

        logger.info(f"Executor completed: {result.summary or result.status}")

    def _store_to_memory(
        self, 
        state_data: dict, 
        perception_result: PerceptionResult, 
        actions: List[str]
    ):
        """Store step to episodic memory."""
        # Convert frame to numpy array as memory module expects np.ndarray
        frame = state_data.get('frame')
        if frame is not None:
            frame = np.array(frame)

        # Use aggregated LLM outputs from all modules this step
        self.memory.store(
            step_number=self.step_count,
            raw_state=state_data,
            image=frame,
            perception=perception_result,
            actions=actions,
            llm_outputs=self.current_step_llm_outputs
        )
