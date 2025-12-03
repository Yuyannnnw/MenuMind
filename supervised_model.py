from typing import List
from dataclasses import dataclass
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor


SEED = 42

@dataclass
class Learner:
    vectorizer: HashingVectorizer
    model: SGDRegressor
    fitted: bool = False

    def features(self, menu_text: str, llm_score: int) -> np.ndarray:
        X_text = self.vectorizer.transform([menu_text])        # shape (1, n_features)
        # append llm_score as a numeric feature
        X = np.hstack([X_text.toarray(), np.array([[llm_score]], dtype=float)])
        return X

    def initial_fit(self, menus: List[str], llm_scores: List[int], user_scores: List[float]) -> None:
        X_text = self.vectorizer.transform(menus).toarray()
        X = np.hstack([X_text, np.array(llm_scores, dtype=float).reshape(-1, 1)])
        y = np.array(user_scores, dtype=float)
        self.model.partial_fit(X, y)
        self.fitted = True

    def update(self, menu_text: str, llm_score: int, user_score: float) -> None:
        X = self.features(menu_text, llm_score)
        y = np.array([user_score], dtype=float)
        self.model.partial_fit(X, y)

    def predict(self, menu_text: str, llm_score: int) -> float:
        X = self.features(menu_text, llm_score)
        pred = float(self.model.predict(X)[0])
        return max(1.0, min(5.0, pred))

def build_learner() -> Learner:
    vect = HashingVectorizer(n_features=1024, alternate_sign=False, norm=None)
    # modest learning rate; tweak if needed
    sgd = SGDRegressor(loss="squared_error", learning_rate="constant", eta0=0.02, random_state=SEED)
    return Learner(vectorizer=vect, model=sgd)
