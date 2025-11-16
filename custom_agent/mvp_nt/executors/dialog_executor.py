"""Dialog executor - handles Pokemon dialog

Simple dialog executor that just presses A
"""

from typing import TYPE_CHECKING
from custom_agent.mvp_hierarchical.executors.base_executor import BaseExecutor, ExecutorResult

if TYPE_CHECKING:
    from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult


class DialogExecutor(BaseExecutor):
    """
    Advances dialog by one
    """

    def __init__(self):
        super().__init__()
        self.dialog_started = False

    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> ExecutorResult:
        """
        Execute dialog advancement step (MVP: spam A).

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Goal from planner (e.g., "Complete the dialog")

        Returns:
            ExecutorResult with action and status
        """

        # Dialog still ongoing - spam A
        return ExecutorResult(
            actions=['A'],
            status='completed',
            summary='Dialog advanced by 1'
        )

    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> bool:
        """
        Dialog executor is valid only while in dialog.

        Invalid if dialog ended.

        Args:
            state_data: Game state data
            perception: Current perception result

        Returns:
            bool: True if still in dialog, False otherwise
        """
        return True

