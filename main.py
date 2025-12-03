from datetime import date
import json
import time
import re
from typing import List
import os
import joblib

from simulate_menu import generate_fake_menu
from menu_scraper import menu_scraper
from supervised_model import build_learner
from llm import llama_score
from plot import plot_metrics

MIN_ROWS_TO_TRAIN = 50
ROUNDS = 80
CSV_PATH = "data/dining_data.csv"
MODEL_PATH = "learner.pkl"


# -----------------------------
# Saving / loading model
# -----------------------------
def load_or_build_learner():
    """Load saved model if exists, otherwise create a new learner."""
    if os.path.exists(MODEL_PATH):
        print("Loading existing learner from disk...")
        learner = joblib.load(MODEL_PATH)
        print("Model loaded successfully.\n")
        return learner
    else:
        print("No saved model found. Creating a new learner...\n")
        return build_learner()


def save_learner(learner):
    """Save model to disk."""
    joblib.dump(learner, MODEL_PATH)
    print("[Model saved]")


# -----------------------------
# CSV appending helper
# -----------------------------
def append_csv(path: str, row: dict) -> None:
    import csv
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["menu", "llm_score", "user_score"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------
# Menu processing
# -----------------------------
def process_menu_dict(menu_dict):
    stop_words = {'and', 'with', 'a', 'the', 'of', 'in', 'for', 'is', 'on', 'or', 'just'}
    all_words = []

    for _, dishes_list in menu_dict.items():
        for text_item in dishes_list:
            text_item = text_item.lower()
            words = re.findall(r'\b\w+\b', text_item)
            filtered_words = [word for word in words 
                              if word not in stop_words and len(word) >= 3]
            all_words.extend(filtered_words)

    return " ".join(all_words)


# -----------------------------
# Main
# -----------------------------
def main():

    print("\n=== Adaptive Dining Advisor (LLM + Online Regression) ===\n")

    with open('profile.json', 'r') as file:
        profile = json.load(file)

    learner = load_or_build_learner()

    menus: List[str] = []
    llm_scores: List[int] = []
    user_scores: List[float] = []

    for t in range(1, ROUNDS + 1):
        print("-" * 72)
        print(f"Round {t}")

        # ==========================================================
        #  FIRST 20 ROUNDS: USE FAKE MENU
        # ==========================================================
        if t <= 30:
            print("=== Using simulated menu for warm-up training ===")
            menu = generate_fake_menu()
            print("🍽️ Fake Menu:")
            print(menu)

        # ==========================================================
        #  AFTER 20 ROUNDS: SCRAPE REAL MENU
        # ==========================================================
        else:
            print("=== Find Menu ===")
            while True:
                try:
                    year = int(input("Enter year (e.g., 2025): "))
                    month = int(input("Enter month (1-12): "))
                    day = int(input("Enter day (1-31): "))
                except ValueError:
                    print("Invalid input: must enter integers.")
                    return

                target = date(year, month, day)
                meal = input("Enter meal (Lunch/Dinner): ").strip()

                print(f"Starting scraper for {meal} on {target.isoformat()}...")
                raw_menu_dict = menu_scraper(target_date=target, meal=meal)

                if raw_menu_dict:
                    print("🍽️ Raw Menu:")
                    print(raw_menu_dict)
                    menu = process_menu_dict(raw_menu_dict)
                    print("🍽️ Processed Menu:")
                    print(menu)
                    break

        # ==========================================================
        # LLM score (same for fake and real menus)
        # ==========================================================
        s_llm, why = llama_score(menu, profile)
        print(f"LLM score: {s_llm} | Rationale: {why}")

        # ==========================================================
        # Prediction logic
        # ==========================================================
        if learner.fitted:
            pred = learner.predict(menu, s_llm)
            print(f"Hybrid prediction (model + LLM): {pred:.2f}")

        elif len(user_scores) >= MIN_ROWS_TO_TRAIN:
            learner.initial_fit(menus, llm_scores, user_scores)
            save_learner(learner)
            pred = learner.predict(menu, s_llm)
            print(f"[Model initialized] Prediction: {pred:.2f}")

        else:
            pred = float(s_llm)
            print(f"[Cold start] Using LLM score only: {pred:.2f}")

        # ==========================================================
        # Ask for user rating (same for fake or real menus)
        # ==========================================================
        while True:
            rating = input("Your rating for this advice (1–5, or 'q' to quit): ").strip()

            if rating.lower() == "q" and learner.fitted:
                print("Exiting. Saving model...")
                save_learner(learner)
                print("Model is saved.")
                return
            elif rating.lower() == "q":
                print("Exiting directly. No model saved.")
                return

            try:
                r_int = int(rating)
                if 1 <= r_int <= 5:
                    break
            except ValueError:
                pass
            print("Enter integer 1–5.")

        # ==========================================================
        # Log data
        # ==========================================================
        menus.append(menu)
        llm_scores.append(s_llm)
        user_scores.append(float(r_int))

        append_csv(CSV_PATH, {
            "menu": menu, 
            "llm_score": s_llm, 
            "user_score": r_int
        })

        # ==========================================================
        # Online update
        # ==========================================================
        if len(user_scores) >= MIN_ROWS_TO_TRAIN:
            learner.update(menu, s_llm, float(r_int))
            save_learner(learner)
            print("Updated model metrics:", learner.metrics)

        time.sleep(0.2)

    print(f"Done with {ROUNDS} rounds")
    print("Model is saved.")

    plot_metrics(learner)


if __name__ == "__main__":
    main()
