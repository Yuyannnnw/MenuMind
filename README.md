# 🍽️ MenuMind

**An Adaptive Dining Recommendation System Using LLMs and Online Learning**

MenuMind is a personalized dining recommendation system designed to help students decide whether a daily dining hall menu is worth visiting. The system combines a local large language model (LLM) with an online supervised learning model to generate menu recommendations that adapt to an individual user’s preferences over time.

The project is developed using the University of Virginia’s Observatory Hill Dining Room (O-Hill) as a real-world case study.

## 📄 Project Report: [MenuMind Final Report (PDF)](docs/MenuMind_Report.pdf)

## ⚙️ Setup and Usage

### User Setup
The user has to update the [profile.json](profile.json) file for this project.

### Running the System
```
python main.py
```

The program will:

1. Load or initialize the learner

2. Obtain a menu ([synthetic](simulate_menu.py) or [scraped](menu_scraper.py))

3. Generate an LLM score ([llm.py](llm.py))

4. Predict a final rating ([supervised_model.py](supervised_model.py))

5. Prompt the user for feedback

6. Update the model and metrics ([supervised_model.py](supervised_model.py))
