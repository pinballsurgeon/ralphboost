class Domain:
    """
    Abstract base class for RalphBoost domains.
    """
    def get_context(self):
        """Return domain-specific context/schema for the agent."""
        return {}

    def propose(self, state, k=1, agent=None, context=None):
        """
        Return up to k candidate component dicts.

        Domain packs may override this to provide deterministic proposals and optionally
        incorporate agent suggestions.
        """
        if agent is None:
            return []
        return agent.propose(state, k=k, context=context or self.get_context())[:k]

    def refine(self, candidate, state):
        """
        Deterministically fit/adjust candidate parameters to the current remainder signal.
        """
        return candidate

    def complexity(self, component, state=None) -> float:
        """
        Return a non-negative proxy for regularization (e.g., #params, span, events).
        """
        return 0.0

    def fingerprint(self, component, state=None):
        """
        Return a hashable fingerprint used for dedupe.
        """
        return None

    def initialize_fitted(self, target):
        """Create a zero-initialized fitted signal/model matching target structure."""
        raise NotImplementedError

    def compute_residual(self, target, fitted):
        raise NotImplementedError

    def energy(self, signal):
        raise NotImplementedError

    def apply(self, component, current_fit, context=None):
        raise NotImplementedError

    def select_best(self, candidates):
        if not candidates:
            return None
        return candidates[0]
