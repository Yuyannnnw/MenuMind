# from datetime import date
# import json
# import time
# import re
# from typing import List

# from menu_scraper import menu_scraper
# from supervised_model import build_learner
# from llm import llama_score

# MIN_ROWS_TO_TRAIN = 1
# ROUNDS = 30
# CSV_PATH = "data/dining_data.csv"

# def append_csv(path: str, row: dict) -> None:
#     import os, csv
#     exists = os.path.exists(path)
#     with open(path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=["menu", "llm_score", "user_score"])
#         if not exists:
#             writer.writeheader()
#         writer.writerow(row)

# def process_menu_dict(menu_dict):
#     """
#     Converts a nested menu dictionary into a flat list of cleaned, individual words.
#     """
#     stop_words = {'and', 'with', 'a', 'the', 'of', 'in', 'for', 'is', 'on', 'or'}
#     all_words = []

#     for _, dishes_list in menu_dict.items():    

#         for text_item in dishes_list:
#             text_item = text_item.lower()
#             words = re.findall(r'\b\w+\b', text_item)
#             filtered_words = [word for word in words if word not in stop_words]
#             all_words.extend(filtered_words)
    
#     return ", ".join(all_words)

# def main():
    
#     print("\n=== Adaptive Dining Advisor (LLM + Online Regression) ===\n")
#     print("Profile being used (edit in the script if you wish):")
#     with open('profile.json', 'r') as file:
#         profile = json.load(file)

#     #print(f"We'll simulate {rounds} menus. You rate each 1–5.")
#     print(f"After {MIN_ROWS_TO_TRAIN} ratings, the supervised model starts predicting and learning online.\n")

#     learner = build_learner()
#     menus: List[str] = []
#     llm_scores: List[int] = []
#     user_scores: List[float] = []

#     for t in range(1, ROUNDS + 1):
#         print("-" * 72)
#         print(f"Round {t}")
#         print("=== Find Menu ===")
#         while True:
#             try:
#                 year = int(input("Enter year (e.g., 2025): "))
#                 month = int(input("Enter month (1-12): "))
#                 day = int(input("Enter day (1-31): "))
#             except ValueError:
#                 print("Invalid input: please enter integers only.")
#                 return

#             target = date(year, month, day)

#             meal = input("Enter meal (Lunch/Dinner): ").strip()

#             print(f"Starting scraper for {meal} on {target.isoformat()}...")
#             menu = menu_scraper(target_date=target, meal=meal)

#             if menu: # Success
#                 print("🍽️Menu: ")
#                 print(menu)
#                 menu = process_menu_dict(menu)
#                 break

#         s_llm, why = llama_score(menu, profile)
#         print(f"LLM score: {s_llm} | Rationale: {why}")

#         if len(user_scores) >= MIN_ROWS_TO_TRAIN and learner.fitted:
#             # Model already online: predict before asking user
#             pred = learner.predict(menu, s_llm)
#             print(f"Model predicted user_score: {pred:.2f} (will be used as advice)")
#         elif len(user_scores) >= MIN_ROWS_TO_TRAIN and not learner.fitted:
#             # First time reaching threshold: warm-start on all accumulated data
#             learner.initial_fit(menus, llm_scores, user_scores)
#             pred = learner.predict(menu, s_llm)
#             print(f"Model initialized. Predicted user_score: {pred:.2f}")
#         else:
#             pred = float(s_llm)
#             print(f"[Cold start] Using LLM score as advice: {pred:.2f}")

#         # Ask for ground-truth user rating
#         while True:
#             try:
#                 rating = input("Your rating for this advice (1–5, or 'q' to quit): ").strip()
#                 if rating.lower() == "q":
#                     raise KeyboardInterrupt
#                 r_int = int(rating)
#                 if 1 <= r_int <= 5:
#                     break
#                 else:
#                     print("Please enter an integer from 1 to 5.")
#             except ValueError:
#                 print("Please enter an integer from 1 to 5, or 'q' to quit.")

#         # Log and update
#         menus.append(menu)
#         llm_scores.append(s_llm)
#         user_scores.append(float(r_int))
#         append_csv(CSV_PATH, {"menu": menu, "llm_score": s_llm, "user_score": r_int})

#         # Once model is active, update online
#         if len(user_scores) >= MIN_ROWS_TO_TRAIN:
#             if not learner.fitted:
#                 # Initial fit already handled above next round
#                 pass
#             else:
#                 learner.update(menu, s_llm, float(r_int))

#         # Small delay just for readability
#         time.sleep(0.2)

#     print("\nDone. Logged data to:", CSV_PATH)
#     if learner.fitted:
#         print("Model trained and updated online during the session.")
#     else:
#         print(f"Fewer than {MIN_ROWS_TO_TRAIN} ratings; only LLM-based advice used.")


# if __name__ == "__main__":
#     main()

from datetime import date
import json
import time
import re
from typing import List
import os
import joblib

from menu_scraper import menu_scraper
from supervised_model import build_learner
from llm import llama_score

MIN_ROWS_TO_TRAIN = 2
ROUNDS = 30
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

    print(f"After {MIN_ROWS_TO_TRAIN} ratings, the supervised model starts predicting and learning online.\n")

    # >>> Load OR build model here <<<
    learner = load_or_build_learner()

    menus: List[str] = []
    llm_scores: List[int] = []
    user_scores: List[float] = []

    for t in range(1, ROUNDS + 1):
        print("-" * 72)
        print(f"Round {t}")

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
            menu = menu_scraper(target_date=target, meal=meal)

            if menu:
                print("🍽️ Menu:")
                print(menu)
                menu = process_menu_dict(menu)
                print("🍽️ Process Menu:")
                print(menu)
                break

        # LLM scoring
        s_llm, why = llama_score(menu, profile)
        print(f"LLM score: {s_llm} | Rationale: {why}")

        # Prediction logic
        if learner.fitted:
            # Model has already been trained (this run or a previous one)
            pred = learner.predict(menu, s_llm)
            print(f"Model predicted user_score: {pred:.2f}")
        elif len(user_scores) >= MIN_ROWS_TO_TRAIN:
            # First time we have enough data to train from scratch
            learner.initial_fit(menus, llm_scores, user_scores)
            save_learner(learner)
            pred = learner.predict(menu, s_llm)
            print(f"Model initialized. Predicted user_score: {pred:.2f}")
        else:
            # Completely cold: no trained model yet and not enough data
            pred = float(s_llm)
            print(f"[Cold start] Using LLM score: {pred:.2f}")

        # Ask for user rating
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

        # Log data
        menus.append(menu)
        llm_scores.append(s_llm)
        user_scores.append(float(r_int))
        append_csv(CSV_PATH, {"menu": menu, "llm_score": s_llm, "user_score": r_int})

        # Online update
        if len(user_scores) >= MIN_ROWS_TO_TRAIN and learner.fitted:
            learner.update(menu, s_llm, float(r_int))
            save_learner(learner)  # <<< save after each update

        time.sleep(0.2)

    print("Done with f{ROUNDS} rounds")
    print("Model is saved.")


if __name__ == "__main__":
    main()
