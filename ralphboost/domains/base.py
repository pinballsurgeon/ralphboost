class Domain:
    """
    Abstract base class for RalphBoost domains.
    """
    def get_context(self):
        """Return domain-specific context/schema for the agent."""
        return {}

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
