from typing import Sequence
import numpy as np
import os
from recsim.environments.interest_exploration import choice_model as _orig

LAMBDA_REP = float(os.environ.get("IE_REP_PENALTY", "0.0"))


class MultinomialLogitChoiceModel(_orig.MultinomialLogitChoiceModel):
    """Shadow upstream class; add a repetition penalty within last K steps."""
    def __init__(self, temperature: float = 1.0, repeat_window: int = 3, penalty: float = 0.2):
        super().__init__(temperature)
        self._repeat_window = repeat_window
        # Use env-configured lambda, fallback to the constructor value if you ever want that.
        self._penalty = LAMBDA_REP if LAMBDA_REP is not None else penalty
        self._recent_topics: Sequence[int] = []
        
    def score_documents(self, docs):
        scores = super().score_documents(docs)

        def _topic(d):
            return getattr(d, "topic", getattr(d, "topics", [None])[0])

        if self._recent_topics:
            penal = np.array(
                [self._penalty if _topic(d) in self._recent_topics[-self._repeat_window:] else 0.0
                 for d in docs], dtype=np.float32)
            scores = scores - penal
        return scores

    def update_state(self, slate_doc, user_response, _unused=None):
        topic = getattr(slate_doc, "topic", getattr(slate_doc, "topics", [None])[0])
        if topic is not None:
            self._recent_topics.append(topic)
            self._recent_topics = self._recent_topics[-32:]
        return super().update_state(slate_doc, user_response, _unused)

