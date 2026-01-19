from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def _as_list(values: Sequence[float] | Iterable[float]) -> List[float]:
    if isinstance(values, list):
        return values
    return list(values)


class Objective:
    name: str

    def link(self, raw_pred: Sequence[float] | Iterable[float]) -> List[float]:
        raise NotImplementedError

    def loss(self, y_true: Sequence[float] | Iterable[float], raw_pred: Sequence[float] | Iterable[float]) -> float:
        raise NotImplementedError

    def remainder(
        self, y_true: Sequence[float] | Iterable[float], raw_pred: Sequence[float] | Iterable[float]
    ) -> List[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class MSEObjective(Objective):
    name: str = "mse"

    def link(self, raw_pred: Sequence[float] | Iterable[float]) -> List[float]:
        return _as_list(raw_pred)

    def loss(self, y_true: Sequence[float] | Iterable[float], raw_pred: Sequence[float] | Iterable[float]) -> float:
        y = _as_list(y_true)
        pred = self.link(raw_pred)
        n = len(y)
        if n == 0:
            return 0.0
        return sum((yt - yp) ** 2 for yt, yp in zip(y, pred)) / n

    def remainder(self, y_true: Sequence[float] | Iterable[float], raw_pred: Sequence[float] | Iterable[float]) -> List[float]:
        y = _as_list(y_true)
        pred = self.link(raw_pred)
        return [yt - yp for yt, yp in zip(y, pred)]


@dataclass(frozen=True)
class LogisticObjective(Objective):
    name: str = "logistic"
    eps: float = 1e-12

    def link(self, raw_pred: Sequence[float] | Iterable[float]) -> List[float]:
        z = _as_list(raw_pred)
        return [1.0 / (1.0 + math.exp(-v)) if v >= 0 else math.exp(v) / (1.0 + math.exp(v)) for v in z]

    def loss(self, y_true: Sequence[float] | Iterable[float], raw_pred: Sequence[float] | Iterable[float]) -> float:
        y = _as_list(y_true)
        p = self.link(raw_pred)
        n = len(y)
        if n == 0:
            return 0.0
        eps = self.eps
        total = 0.0
        for yt, pt in zip(y, p):
            pt = min(1.0 - eps, max(eps, pt))
            total += -(yt * math.log(pt) + (1.0 - yt) * math.log(1.0 - pt))
        return total / n

    def remainder(self, y_true: Sequence[float] | Iterable[float], raw_pred: Sequence[float] | Iterable[float]) -> List[float]:
        y = _as_list(y_true)
        p = self.link(raw_pred)
        return [yt - pt for yt, pt in zip(y, p)]

