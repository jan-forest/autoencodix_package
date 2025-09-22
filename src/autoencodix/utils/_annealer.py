from typing import Dict, Callable, Optional
from math import exp


class AnnealingScheduler:
    """VAE loss annealing scheduler with multiple annealing strategies."""

    def __init__(self) -> None:
        """Initialize the annealing scheduler with strategy mappings."""
        self._strategies: Dict[str, Callable[[int, int], float]] = {
            "5phase-constant": self._five_phase_constant,
            "3phase-linear": self._three_phase_linear,
            "3phase-log": self._three_phase_log,
            "logistic-mid": self._logistic_mid,
            "logistic-early": self._logistic_early,
            "logistic-late": self._logistic_late,
            "no-annealing": self._no_annealing,
        }

    @staticmethod
    def get_annealing_epoch(
        *, anneal_pretraining: bool, n_epochs_pretrain: int, current_epoch: int
    ) -> Optional[int]:
        """Check if annealing should be used for pretraining.
        Args:
            anneal_pretraining: Whether to apply annealing during pretraining phase.
            n_epochs_pretrain: Number of pretraining epochs.
            current_epoch: Current epoch number.
        Returns:
            int or None: Annealing epoch number, or None if no annealing.
        Raises:
            NotImplementedError: This is a deprecated method.
        """
        raise NotImplementedError(
            "Deprecated, for annealing the current epoch is passed, we split between training and \
                                  pretraining, so no extra calculation is needed"
        )

    def get_weight(
        self,
        *,
        epoch_current: Optional[int],
        total_epoch: int,
        func: str = "logistic-mid",
    ) -> float:
        """Calculate VAE loss annealing weight.

        Args:
            epoch_current: Current epoch in training, or None for full weight.
            total_epoch: Total epochs for training.
            func: Specification of annealing function. Default is 'logistic-mid'.

        Returns:
            Annealing weight between 0 (no VAE loss) and 1 (full VAE loss).

        Raises:
            NotImplementedError: If the specified annealing function is not implemented.
        """
        if epoch_current is None:
            return 1.0

        if func not in self._strategies:
            raise NotImplementedError("The annealer is not implemented yet")

        return self._strategies[func](epoch_current, total_epoch)

    def _five_phase_constant(self, epoch_current: int, total_epoch: int) -> float:
        """Five phase constant annealing strategy.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
            Annealing weight.
        """
        intervals = 5
        current_phase = int((epoch_current / total_epoch) * intervals)

        if current_phase == 0:
            return 0.0
        elif current_phase == 1:
            return 0.001
        elif current_phase == 2:
            return 0.01
        elif current_phase == 3:
            return 0.1
        else:
            return 1.0

    def _three_phase_linear(self, epoch_current: int, total_epoch: int) -> float:
        """Three phase linear annealing strategy.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
           Annealing weight.
        """
        first_phase_end = total_epoch / 3
        second_phase_end = 2 * (total_epoch / 3)

        if epoch_current < first_phase_end:
            return 0.0
        elif epoch_current < second_phase_end:
            return (epoch_current - first_phase_end) / (total_epoch / 3)
        else:
            return 1.0

    def _three_phase_log(self, epoch_current: int, total_epoch: int) -> float:
        """Three phase logarithmic annealing strategy.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
            Annealing weight.
        """
        first_phase_end = total_epoch / 3
        second_phase_end = 2 * (total_epoch / 3)

        if epoch_current < first_phase_end:
            return 0.0
        elif epoch_current < second_phase_end:
            return self._logistic_mid(epoch_current - first_phase_end, total_epoch / 3)
        else:
            return 1.0

    def _compute_logistic_weight(
        self, epoch_current: int, total_epoch: int, midpoint: float
    ) -> float:
        """Compute logistic annealing weight.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.
            midpoint: Midpoint ratio for the logistic function (0.0 to 1.0).

        Returns:
            Annealing weight.
        """
        b_param = (1 / total_epoch) * 20
        return 1 / (1 + exp(-b_param * (epoch_current - total_epoch * midpoint)))

    def _logistic_mid(self, epoch_current: int, total_epoch: int) -> float:
        """Logistic annealing with midpoint at half of total epochs.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
            Annealing weight.
        """
        return self._compute_logistic_weight(epoch_current, total_epoch, 0.5)

    def _logistic_early(self, epoch_current: int, total_epoch: int) -> float:
        """Logistic annealing with early midpoint at quarter of total epochs.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
            Annealing weight.
        """
        return self._compute_logistic_weight(epoch_current, total_epoch, 0.25)

    def _logistic_late(self, epoch_current: int, total_epoch: int) -> float:
        """Logistic annealing with late midpoint at three quarters of total epochs.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
            Annealing weight.
        """
        return self._compute_logistic_weight(epoch_current, total_epoch, 0.75)

    def _no_annealing(self, epoch_current: int, total_epoch: int) -> float:
        """No annealing strategy - constant full weight.

        Args:
            epoch_current: Current epoch number.
            total_epoch: Total number of epochs.

        Returns:
            Annealing weight (always 1.0).
        """
        return 1.0
