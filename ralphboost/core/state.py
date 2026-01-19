from dataclasses import dataclass, field
from typing import List, Any, Optional

@dataclass(frozen=True)
class RalphState:
    """
    Immutable state container for the boosting loop.
    Zero-copy where possible (in Python terms, references are copy, but we treat lists as immutable).
    """
    residual: Any  # Can be list, numpy array, etc.
    fitted: Optional[Any] = None
    components: List[Any] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    iteration: int = 0
    metrics: dict = field(default_factory=dict)

    def update(
        self,
        new_fitted: Any,
        new_residual: Any,
        component: Any,
        metric_update: dict | None = None,
        history_entry: dict | None = None,
    ) -> "RalphState":
        """
        Return a new state with the component applied.
        """
        new_components = self.components + [component]
        new_metrics = self.metrics.copy()
        if metric_update:
            new_metrics.update(metric_update)

        entry = {"iteration": self.iteration + 1, "metrics": new_metrics}
        if history_entry:
            entry.update(history_entry)
            
        return RalphState(
            residual=new_residual,
            fitted=new_fitted,
            components=new_components,
            history=self.history + [entry],
            iteration=self.iteration + 1,
            metrics=new_metrics
        )
