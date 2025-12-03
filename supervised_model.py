# supervised_model.py
from typing import List
from dataclasses import dataclass
import numpy as np
import math

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor

SEED = 42


# -------------------------------------
# Metrics tracking: MAE / RMSE / count
# -------------------------------------
class Metrics:
    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0

        self.history_n = []
        self.history_mae = []
        self.history_rmse = []

    def update(self, pred, y):
        error = pred - y
        self.sum_abs += abs(error)
        self.sum_sq += error * error
        self.n += 1

        mae_now = self.mae
        rmse_now = self.rmse

        self.history_n.append(self.n)
        self.history_mae.append(mae_now)
        self.history_rmse.append(rmse_now)

    @property
    def mae(self):
        return self.sum_abs / self.n if self.n > 0 else None

    @property
    def rmse(self):
        return math.sqrt(self.sum_sq / self.n) if self.n > 0 else None

    def __repr__(self):
        if self.n == 0:
            return "MAE=None, RMSE=None, n=0"
        return f"MAE={self.mae:.3f}, RMSE={self.rmse:.3f}, n={self.n}"


# -------------------------------------
# The unified learner
# -------------------------------------
@dataclass
class Learner:
    vectorizer: HashingVectorizer
    model: SGDRegressor
    fitted: bool = False
    num_samples: int = 0
    metrics: Metrics = Metrics()

    TEXT_SCALE: float = 3.0   # boosts text signal
    NORM_LLM: float = 100.0   # llm score scaled to [0,1]

    # -------------
    # Feature builder
    # -------------
    def features(self, menu_text: str, llm_score: int) -> np.ndarray:
        X_text = self.vectorizer.transform([menu_text]).toarray()
        X_text *= self.TEXT_SCALE

        llm_scaled = llm_score / self.NORM_LLM
        X = np.hstack([X_text, np.array([[llm_scaled]], dtype=float)])
        return X

    # -------------------------------
    # Initial batch training
    # -------------------------------
    def initial_fit(self, menus: List[str], llm_scores: List[int], user_scores: List[float]) -> None:
        X_text = self.vectorizer.transform(menus).toarray()
        X_text *= self.TEXT_SCALE

        llm_scaled = np.array(llm_scores, dtype=float) / self.NORM_LLM
        X = np.hstack([X_text, llm_scaled.reshape(-1, 1)])

        y = np.array(user_scores, dtype=float)
        self.model.partial_fit(X, y)
        self.fitted = True
        self.num_samples += len(y)

    # -------------------------------
    # Online update
    # -------------------------------
    def update(self, menu_text: str, llm_score: int, user_score: float) -> None:
        X = self.features(menu_text, llm_score)
        y = np.array([user_score], dtype=float)
        self.model.partial_fit(X, y)
        self.fitted = True

        self.num_samples += 1

        # Hybrid prediction after update
        pred = self.hybrid_predict(menu_text, llm_score)
        self.metrics.update(pred, user_score)

    # -------------------------------
    # Raw regression prediction
    # -------------------------------
    def predict_raw(self, menu_text: str, llm_score: int) -> float:
        if not self.fitted:
            return None
        X = self.features(menu_text, llm_score)
        return float(self.model.predict(X)[0])

    # -------------------------------
    # Hybrid α-weighted scoring
    # -------------------------------
    def hybrid_predict(self, menu_text: str, llm_score: int) -> float:
        raw_pred = self.predict_raw(menu_text, llm_score)
        if raw_pred is None:
            return float(llm_score)  # cold start

        # α grows with number of training samples
        alpha = min(self.num_samples / 50, 1.0)

        final = alpha * raw_pred + (1 - alpha) * float(llm_score)
        return max(1.0, min(5.0, final))  # clamp to [1,5]

    # -------------------------------
    # Public predict method
    # -------------------------------
    def predict(self, menu_text: str, llm_score: int) -> float:
        return self.hybrid_predict(menu_text, llm_score)


# -------------------------------------
# Build learner with tuned HashingVectorizer / SGD
# -------------------------------------
def build_learner() -> Learner:
    vect = HashingVectorizer(
        n_features=1024,
        alternate_sign=False,
        norm=None
    )
    sgd = SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        eta0=0.02,
        random_state=SEED
    )
    return Learner(vectorizer=vect, model=sgd)
