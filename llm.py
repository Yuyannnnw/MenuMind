import ollama
import json
from typing import Tuple

MODEL_NAME = "llama3.1:8b"

def llama_score(menu_text: str, user_profile: dict, model: str = MODEL_NAME) -> Tuple[int, str]:
    """
    Ask Llama (via Ollama) for a base 1–5 score + short rationale.
    We instruct it to ignore protected attributes for scoring.
    """
    print("\nOllama model:", MODEL_NAME)
    sys_rules = (
        "You are a careful dining advisor. "
        "Your job is to evaluate whether a user should visit the dining hall "
        "Consider how well the overall menu fits the user's dietary needs, "
        "allergies, and food preferences. Ignore protected attributes such as race. "
        "Return ONLY valid JSON with integer 'score' (1–5) and string 'rationale' (≤140 chars)."
    )

    prompt = f"""User profile: {json.dumps(user_profile, ensure_ascii=False)}
    Today's menu: {menu_text}
    Rate this MENU AS A WHOLE from 1 (do not go) to 5 (strongly recommend).
    Output strict JSON only, like this:
    {{"score": 4, "rationale": "Good variety of vegetarian options and safe for allergies."}}
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys_rules},
                      {"role": "user", "content": prompt}]
        )
        reply = response["message"]["content"].strip()
        obj = json.loads(reply)
        score = int(obj.get("score", 3))
        why = obj.get("rationale", "LLM rationale unavailable.")
    except Exception as e:
        print(f"[WARN] LLM error: {e}")
        score, why = 3, "Fallback (parse or connection error)."

    score = max(1, min(5, score))
    return score, why