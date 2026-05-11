# 🏏 IPL Win Probability & Score Prediction System

This project is a machine learning-based IPL analytics system that predicts:

- First Innings final score
- Second innings win probability (live match simulation)
- Pre-match insights (team-level analysis)

It uses historical IPL ball-by-ball data to build predictive models and visualize match progression.

---

## 📁 Project Structure
Prediction_League/
│
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned datasets used for modeling
│
├── models/
│
├── notebooks/
│ ├── first_innings.ipynb
│ ├── second_innings.ipynb
│ └── prematch_prediction.ipynb
│
├── app.py 
├── .gitignore
└── README.md


---

## ⚙️ Features

### 🏏 First Innings Model
- Predicts final score using:
  - Current runs
  - Wickets fallen
  - Balls remaining
  - Run rate
- Model: Random Forest Regressor

### 📊 Win Probability Engine
- Maps predicted score → historical win probability
- Generates live match win probability curve

### 🔍 Data Processing
- Ball-by-ball aggregation
- Feature engineering:
  - Run rate
  - Balls left
  - Wickets
  - Cumulative score

---

## 🧠 Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Joblib
