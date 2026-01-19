try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:
    class BaseEstimator:
        def get_params(self, deep=True):
            return self.__dict__
        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self
            
    class RegressorMixin:
        def score(self, X, y, sample_weight=None):
            pass

class RalphBooster(BaseEstimator, RegressorMixin):
    """
    RalphBoost: Agentic Gradient Boosting Framework.
    """
    def __init__(
        self,
        max_iterations=100,
        learning_rate=0.1,
        k_candidates=10,
        objective="mse",
        min_improvement=0.0,
        min_residual_reduction=0.01,
        early_stopping_rounds=None,
        complexity_weight=0.0,
        duplicate_penalty=0.0,
        duplicate_min_improvement=None,
        thinking_budget=10.0,
        agent_backend='gemini',
        n_jobs=1,
        random_state=None,
        verbose=0,
        domain=None
    ):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.k_candidates = k_candidates
        self.objective = objective
        self.min_improvement = min_improvement
        self.min_residual_reduction = min_residual_reduction
        self.early_stopping_rounds = early_stopping_rounds
        self.complexity_weight = complexity_weight
        self.duplicate_penalty = duplicate_penalty
        self.duplicate_min_improvement = duplicate_min_improvement
        self.thinking_budget = thinking_budget
        self.agent_backend = agent_backend
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.domain = domain
        
        # State
        self.components_ = []
        self.history_ = []
        self.final_residual_ = None
        self.final_fitted_ = None

    def fit(self, X, y=None, eval_set=None, callbacks=None, sample_rate=None):
        """
        Fit the model to the data.
        """
        if self.max_iterations is None or int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if self.learning_rate is None or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if self.k_candidates is None or int(self.k_candidates) <= 0:
            raise ValueError("k_candidates must be a positive integer.")
        if self.min_residual_reduction is not None:
            mrr = float(self.min_residual_reduction)
            if mrr < 0.0 or mrr > 1.0:
                raise ValueError("min_residual_reduction must be in [0, 1] or None.")
        if self.min_improvement is None or float(self.min_improvement) < 0.0:
            raise ValueError("min_improvement must be >= 0.")
        if self.early_stopping_rounds is not None and int(self.early_stopping_rounds) < 0:
            raise ValueError("early_stopping_rounds must be >= 0 or None.")

        # 1. Setup Domain
        # If domain was passed in __init__, use it. 
        # If not, try to infer or default to SignalDomain (if sample_rate provided)
        if not hasattr(self, 'domain') or self.domain is None:
             from ..domains.signal import SignalDomain
             self.domain = SignalDomain(sample_rate=sample_rate if sample_rate else 100.0)

        # 2. Setup Agent
        # Import inside method to avoid circular imports or heavy load
        if self.agent_backend == 'mock':
            from ..agents.mock import MockSignalAgent
            agent = MockSignalAgent()
        elif self.agent_backend == 'fft':
            from ..agents.fft_agent import FFTSignalAgent
            sr = getattr(self.domain, "sample_rate", None)
            if sr is None:
                sr = sample_rate if sample_rate else 1.0
            agent = FFTSignalAgent(sample_rate=sr)
        elif self.agent_backend == 'gemini':
            from ..agents.gemini import GeminiAgent
            # agent = GeminiAgent()
            # If import fails (google-genai missing), fallback or raise?
            # GeminiAgent raises ImportError if missing.
            try:
                agent = GeminiAgent()
            except ImportError:
                print("Warning: google-genai not installed. Falling back to MockAgent.")
                from ..agents.mock import MockSignalAgent
                agent = MockSignalAgent()
        else:
            from ..agents.mock import MockSignalAgent
            agent = MockSignalAgent()

        # 3. Setup Refiner
        from ..refiners.optimization import OptimizationRefiner
        refiner = OptimizationRefiner()

        # 4. Run Engine
        from .engine import RalphEngine
        from .objectives import LogisticObjective, MSEObjective

        if isinstance(self.objective, str):
            objective_name = self.objective.strip().lower()
            if objective_name in {"mse", "l2"}:
                objective = MSEObjective()
            elif objective_name in {"logistic", "logloss", "cross_entropy"}:
                objective = LogisticObjective()
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
        else:
            objective = self.objective

        engine = RalphEngine(
            self.domain,
            agent,
            refiner,
            objective=objective,
            learning_rate=self.learning_rate,
            k_candidates=self.k_candidates,
            min_improvement=self.min_improvement,
            early_stopping_rounds=self.early_stopping_rounds,
            complexity_weight=self.complexity_weight,
            duplicate_penalty=self.duplicate_penalty,
            duplicate_min_improvement=self.duplicate_min_improvement,
            verbose=self.verbose,
        )
        
        # Target handling: if y is None, assume X is the target signal
        target = y if y is not None else X
        
        result = engine.fit(target, max_iterations=self.max_iterations, min_residual_reduction=self.min_residual_reduction)
        
        self.components_ = result.components
        self.history_ = result.history
        self.metrics_ = result.metrics
        self.final_residual_ = result.final_residual
        self.final_fitted_ = result.final_fitted
        
        return self

    @property
    def components(self):
        return self.components_
        
    @property
    def metrics(self):
        return self.metrics_

    @property
    def final_residual(self):
        return self.final_residual_

    @property
    def final_fitted(self):
        return self.final_fitted_

    @property
    def history(self):
        return self.history_

    def predict(self, X):
        """
        Predict using the model.
        """
        if self.domain is None:
            raise ValueError("No domain set; call fit() first or pass domain=... in __init__.")

        if X is None:
            raise ValueError("X cannot be None.")

        fitted = self.domain.initialize_fitted(X)
        for component in self.components_:
            fitted = self.domain.apply(component, fitted)

        if isinstance(self.objective, str) and self.objective.strip().lower() in {"logistic", "logloss", "cross_entropy"}:
            from .objectives import LogisticObjective

            fitted = LogisticObjective().link(fitted)

        return fitted
