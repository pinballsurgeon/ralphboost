from .base import Refiner

class OptimizationRefiner(Refiner):
    def refine_batch(self, candidates, state, domain=None):
        if not candidates:
            return []
        if domain is None:
            return list(candidates)

        refined = []
        for candidate in candidates:
            try:
                component = domain.refine(candidate, state)
            except Exception:
                component = candidate
            if component is not None:
                refined.append(component)
        return refined
