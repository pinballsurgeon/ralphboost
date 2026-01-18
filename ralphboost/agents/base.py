class Agent:
    """
    Abstract base class for RalphBoost agents.
    """
    def propose(self, state, k=1, context=None):
        """
        Propose k candidate components based on the current state (residual) and domain context.
        """
        raise NotImplementedError
