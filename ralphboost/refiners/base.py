class Refiner:
    """
    Abstract base class for component refiners.
    """
    def refine_batch(self, candidates, state):
        """
        Refine a batch of candidate components to better fit the residual.
        """
        raise NotImplementedError
