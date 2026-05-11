import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(page_title="IPL Predictor", layout="wide")
st.title("🏏 IPL Prediction System")

# ==================================
# SESSION STATE
# ==================================
if "match_ready" not in st.session_state:
    st.session_state.match_ready = False

if "innings" not in st.session_state:
    st.session_state.innings = None

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================
# LOAD DATA
# ==================================
matches = pd.read_csv("data/processed/matches_new.csv")

matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
matches = matches.dropna(subset=["date"])
matches = matches.sort_values("date")

# ==================================
# LOAD MODELS
# ==================================
prematch_model = joblib.load("models/prematch_model.pkl")
features = joblib.load("models/features.pkl")

first_innings = joblib.load("models/first_innings.pkl")

second_innings = joblib.load("models/second_innings.pkl")
second_features = joblib.load("models/second_features.pkl")

# ==================================
# FEATURE ENGINEERING
# ==================================
def generate_features(team1, team2, venue):

    matches_played = pd.concat([matches["team1"], matches["team2"]]).value_counts()
    wins = matches["winner"].value_counts()

    win_rate = (wins / matches_played).fillna(0.5)

    win_rate_diff = win_rate.get(team1, 0.5) - win_rate.get(team2, 0.5)

    h2h = matches[
        ((matches["team1"] == team1) & (matches["team2"] == team2)) |
        ((matches["team1"] == team2) & (matches["team2"] == team1))
    ]

    h2h_rate = (
        (h2h["winner"] == team1).sum() -
        (h2h["winner"] == team2).sum()
    )

    form1 = matches[(matches["team1"] == team1) | (matches["team2"] == team1)].tail(5)
    form2 = matches[(matches["team1"] == team2) | (matches["team2"] == team2)].tail(5)

    form_rate = form1["winner"].eq(team1).sum() - form2["winner"].eq(team2).sum()

    venue_matches = matches[matches["venue"] == venue]

    def venue_wr(team):
        total = venue_matches[
            (venue_matches["team1"] == team) |
            (venue_matches["team2"] == team)
        ].shape[0]

        wins = venue_matches[venue_matches["winner"] == team].shape[0]

        return wins / total if total > 0 else 0.5

    venue_rate = venue_wr(team1) - venue_wr(team2)

    return win_rate_diff, form_rate, h2h_rate, venue_rate


# ==================================
# LIVE GRAPH
# ==================================
def plot_win_probability():

    if len(st.session_state.history) < 2:
        return

    df = pd.DataFrame(st.session_state.history)

    fig, ax = plt.subplots()

    df["step"] = range(len(df))

    ax.plot(df["step"], df["win_prob"])
    ax.fill_between(df["step"], df["win_prob"], alpha=0.2)

    ax.set_xlabel("Ball Progression")
    ax.set_ylabel("Win Probability (%)")
    ax.set_title("Live Match Momentum")

    st.pyplot(fig)


# ==================================
# USER INPUT
# ==================================
teams = sorted(matches["team1"].dropna().unique())
venues = sorted(matches["venue"].dropna().unique())

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.radio("Toss Decision", ["bat", "field"])
venue = st.selectbox("Venue", venues)

# ==================================
# MATCH ORDER
# ==================================
if toss_decision == "bat":
    first_batting = toss_winner
    first_bowling = team2 if toss_winner == team1 else team1
else:
    first_bowling = toss_winner
    first_batting = team2 if toss_winner == team1 else team1

st.info(f"🏏 {first_batting} will bat first")

# ==================================
# PREMATCH PREDICTION
# ==================================
if st.button("Predict Match"):

    st.session_state.history = []  # reset history

    wr, fr, h2h, vr = generate_features(team1, team2, venue)

    input_df = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "venue": venue,
        "win_rate_diff": wr,
        "form_diff": fr,
        "h2h_diff": h2h,
        "venue_diff": vr,
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)

    prob = prematch_model.predict_proba(input_df)[0]

    st.success(f"{team1} Win Probability: {prob[1]*100:.2f}%")
    st.success(f"{team2} Win Probability: {prob[0]*100:.2f}%")

    st.session_state.match_ready = True


# ==================================
# INNINGS SELECTION
# ==================================
if st.session_state.match_ready:

    st.header("Select Innings")

    c1, c2 = st.columns(2)

    if c1.button("1st Innings"):
        st.session_state.innings = 1

    if c2.button("2nd Innings"):
        st.session_state.innings = 2


# ==================================
# FIRST INNINGS
# ==================================
if st.session_state.innings == 1:

    st.subheader("🏏 First Innings Predictor")

    score = st.number_input("Current Score", 0, 300)
    wickets = st.number_input("Wickets Lost", 0, 10)
    balls = st.number_input("Balls Bowled", 1, 120)

    if st.button("Predict Final Score"):

        ball_left = 120 - balls

        crr = score / (balls / 6 + 1e-6)

        X = pd.DataFrame([{
            "current_score": score,
            "total_wicket": wickets,
            "ball_left": ball_left,
            "current_rr": crr,
        }])

        pred = first_innings.predict(X)[0]

        st.success(f"Predicted Score Range: {int(pred-5)} - {int(pred+5)}")


# ==================================
# SECOND INNINGS
# ==================================
if st.session_state.innings == 2:

    st.subheader("🔥 Second Innings Win Predictor")

    batting_team = first_bowling
    bowling_team = first_batting

    target = st.number_input("Target", 1, 300)
    runs = st.number_input("Current Runs", 0, 300)
    wickets = st.number_input("Wickets Lost", 0, 10)
    balls = st.number_input("Balls Bowled", 1, 120)

    if st.button("Predict Win Probability"):

        balls_left = 120 - balls
        runs_left = target - runs

        crr = runs / (balls / 6 + 1e-6)
        rrr = runs_left / (balls_left / 6 + 1e-6)

        X = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": 10 - wickets,
            "target": target,
            "crr": crr,
            "rrr": rrr,
        }])

        X = pd.get_dummies(X)
        X = X.reindex(columns=second_features, fill_value=0)

        prob = second_innings.predict_proba(X)[0]

        bat_prob = prob[1] * 100

        st.success(f"{batting_team} Win %: {bat_prob:.2f}")
        st.success(f"{bowling_team} Win %: {prob[0]*100:.2f}")

        # save history
        st.session_state.history.append({
            "win_prob": bat_prob
        })

        plot_win_probability()