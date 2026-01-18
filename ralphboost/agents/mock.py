from .base import Agent

class MockSignalAgent(Agent):
    def propose(self, state, k=1, context=None):
        return [
            {"frequency": 1.0, "amplitude": 0.5, "phase": 0.0, "confidence": 0.9}
        ]
