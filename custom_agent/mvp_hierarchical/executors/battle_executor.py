"""Battle executor - handles Pokemon battles.

MVP: Simple battle handler that spams 'A' until battle ends.
Future: Proper battle AI with move selection, switching, item usage, etc.
"""
import logging
from typing import TYPE_CHECKING
from custom_agent.mvp_hierarchical.executors.base_executor import BaseExecutor, ExecutorResult

if TYPE_CHECKING:
    from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult

logger = logging.getLogger(__name__)


class BattleExecutor(BaseExecutor):
    """
    Simple battle executor (MVP: spam A until battle ends).

    Future: Implement proper battle AI with:
    - Move selection based on type effectiveness
    - Switching strategy
    - Item usage
    - HP management
    """

    def __init__(self):
        super().__init__()
        self.battle_started = False

    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> ExecutorResult:
        """
        Execute battle step (MVP: spam A).

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Goal from planner (e.g., "Complete the battle")

        Returns:
            ExecutorResult with action and status
        """
        # TODO: WARNING: CANNOT SPAM A SINCE IF A POKEMON IS KO / RAN OUT OF PP, STUCK IN INF LOOP
        # Check if battle ended
        if not state_data.get('game', {}).get('is_in_battle', False):
            logger.info("No longer in battle")
            return ExecutorResult(
                actions=[],
                status='completed',
                summary="Battle completed"
            )

        # Battle still ongoing - spam A
        return ExecutorResult(
            actions=['A'],
            status='in_progress'
        )

    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> bool:
        """
        Battle executor is valid only while in battle.

        Invalid if battle ended.

        Args:
            state_data: Game state data
            perception: Current perception result

        Returns:
            bool: True if still in battle, False otherwise
        """
        # validity = state_data.get('game', {}).get('is_in_battle', False)
        # if not validity:
        #     logger.info("Battle executor is no longer valid - not in battle")
        # return validity

        # we rely on is_in_battle from execute_step to determine when to stop
        # always return True since battles arent 'invalidated' but rather completed
        return True

    def reset(self):
        """Reset battle executor to initial state."""
        super().reset()
        self.battle_started = False

