from flask import Flask, request, jsonify, make_response
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import textwrap

CSV_PATH = "matches_full.csv"
N_ESTIMATORS = 400
RANDOM_STATE = 42

app = Flask(__name__)

rf_model = None
target_le = None
feature_cols = None
home_lookup = None
away_lookup = None
available_seasons = None
available_teams = None
mean_vals = None


def load_and_train():
    global rf_model, target_le, feature_cols, home_lookup, away_lookup
    global available_seasons, available_teams, mean_vals

    df = pd.read_csv(CSV_PATH)
    if 'comp' in df.columns:
        df = df[df['comp'].astype(str).str.contains('La Liga', case=False, na=False)]

    required_cols = {'season', 'team', 'opponent', 'venue', 'gf', 'ga', 'result'}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[FATAL] Missing required columns: {missing}")

    df = df[df['result'].isin(['W', 'L', 'D'])].copy()
    df['gd'] = df['gf'] - df['ga']
    df['match_result'] = df['result'].map({'W': 'HomeWin', 'L': 'AwayWin', 'D': 'Draw'})

    home_df = df[df['venue'].str.lower() == 'home']
    away_df = df[df['venue'].str.lower() == 'away']

    home_stats = (
        home_df
        .groupby(['season', 'team'])
        .agg(
            home_gf_avg=('gf', 'mean'),
            home_ga_avg=('ga', 'mean'),
            home_gd_avg=('gd', 'mean'),
            home_win_rate=('result', lambda s: (s == 'W').mean()),
            home_draw_rate=('result', lambda s: (s == 'D').mean())
        )
        .reset_index()
    )

    away_stats = (
        away_df
        .groupby(['season', 'team'])
        .agg(
            away_gf_avg=('gf', 'mean'),
            away_ga_avg=('ga', 'mean'),
            away_gd_avg=('gd', 'mean'),
            away_win_rate=('result', lambda s: (s == 'W').mean()),
            away_draw_rate=('result', lambda s: (s == 'D').mean())
        )
        .reset_index()
    )

    feat_df = (
        home_df
        .merge(home_stats, on=['season', 'team'], how='left')
        .merge(
            away_stats.add_prefix('opp_').rename(columns={'opp_season': 'season',
                                                          'opp_team': 'opponent'}),
            on=['season', 'opponent'],
            how='left'
        )
    )

    feature_cols_local = [
        'home_gf_avg', 'home_ga_avg', 'home_gd_avg', 'home_win_rate', 'home_draw_rate',
        'opp_away_gf_avg', 'opp_away_ga_avg', 'opp_away_gd_avg',
        'opp_away_win_rate', 'opp_away_draw_rate'
    ]

    feat_df[feature_cols_local] = feat_df[feature_cols_local].fillna(
        feat_df[feature_cols_local].mean()
    )

    X = feat_df[feature_cols_local].values
    y = feat_df['match_result'].values

    target_le_local = LabelEncoder()
    y_enc = target_le_local.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    )
    model.fit(X, y_enc)

    home_lookup_local = home_stats.set_index(['season', 'team']).to_dict('index')
    away_lookup_local = away_stats.set_index(['season', 'team']).to_dict('index')
    available_seasons_local = sorted(df['season'].dropna().unique().tolist())
    available_teams_local = sorted(df['team'].dropna().unique().tolist())
    mean_vals_local = {col: feat_df[col].mean() for col in feature_cols_local}

    rf_model = model
    target_le = target_le_local
    feature_cols = feature_cols_local
    home_lookup = home_lookup_local
    away_lookup = away_lookup_local
    available_seasons = available_seasons_local
    available_teams = available_teams_local
    mean_vals = mean_vals_local


def build_feature_vector(season, home_team, away_team):
    def pick(row, key, default):
        return row.get(key, default) if row and key in row and pd.notna(row[key]) else default

    hrow = home_lookup.get((season, home_team))
    arow = away_lookup.get((season, away_team))

    fv = [
        pick(hrow, 'home_gf_avg', mean_vals['home_gf_avg']),
        pick(hrow, 'home_ga_avg', mean_vals['home_ga_avg']),
        pick(hrow, 'home_gd_avg', mean_vals['home_gd_avg']),
        pick(hrow, 'home_win_rate', mean_vals['home_win_rate']),
        pick(hrow, 'home_draw_rate', mean_vals['home_draw_rate']),
        pick(arow, 'away_gf_avg', mean_vals['opp_away_gf_avg']),
        pick(arow, 'away_ga_avg', mean_vals['opp_away_ga_avg']),
        pick(arow, 'away_gd_avg', mean_vals['opp_away_gd_avg']),
        pick(arow, 'away_win_rate', mean_vals['opp_away_win_rate']),
        pick(arow, 'away_draw_rate', mean_vals['opp_away_draw_rate'])
    ]
    return np.array(fv).reshape(1, -1)


@app.route("/")
def index():
    seasons_json = json.dumps(available_seasons)
    teams_json = json.dumps(available_teams)

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>La Liga Match Outcome Predictor</title>
<style>
body {
  background: linear-gradient(135deg, #2c3e50, #3498db);
  font-family: Arial, sans-serif;
  color: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  margin: 0;
}
.container {
  background: rgba(0,0,0,0.5);
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 0 10px rgba(0,0,0,0.5);
  text-align: center;
  width: 300px;
}
h1 {
  margin-bottom: 20px;
  font-size: 1.5em;
  color: #f1c40f;
}
label {
  display: block;
  margin-top: 10px;
  font-weight: bold;
}
select, button {
  width: 100%;
  padding: 8px;
  margin-top: 5px;
  border: none;
  border-radius: 5px;
  font-size: 1em;
}
button {
  background-color: #f39c12;
  color: white;
  cursor: pointer;
  transition: background 0.3s;
}
button:hover {
  background-color: #e67e22;
}
#result {
  margin-top: 20px;
  background: #34495e;
  padding: 10px;
  border-radius: 5px;
  font-size: 0.9em;
  white-space: pre-wrap;
}
</style>
<script>
const seasons = """ + seasons_json + """;
const teams = """ + teams_json + """;
</script>
</head>
<body>
<div class="container">
  <h1>La Liga Predictor</h1>
  <form onsubmit="event.preventDefault(); predict();">
    <label>Season: <select id="season"></select></label>
    <label>Home Team: <select id="homeTeam"></select></label>
    <label>Away Team: <select id="awayTeam"></select></label>
    <button type="submit">Predict</button>
  </form>
  <div id="result"></div>
</div>
<script>
function populate() {
  seasons.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    document.getElementById('season').appendChild(opt);
  });
  teams.forEach(t => {
    const o1 = document.createElement('option');
    o1.value = t;
    o1.textContent = t;
    document.getElementById('homeTeam').appendChild(o1);
    const o2 = document.createElement('option');
    o2.value = t;
    o2.textContent = t;
    document.getElementById('awayTeam').appendChild(o2);
  });
}
populate();

function predict() {
  const season = document.getElementById('season').value;
  const home = document.getElementById('homeTeam').value;
  const away = document.getElementById('awayTeam').value;
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ season: parseInt(season), home_team: home, away_team: away })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('result').innerHTML =
      'Prediction: ' + data.prediction + '\\n\\nProbabilities:\\n' + JSON.stringify(data.probabilities, null, 2);
  });
}
</script>
</body>
</html>
"""
    resp = make_response(textwrap.dedent(html))
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    season = data.get("season", max(available_seasons))
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if home_team == away_team:
        return jsonify({"error": "Teams must be different"}), 400
    if home_team not in available_teams or away_team not in available_teams:
        return jsonify({"error": "Invalid team name(s)"}), 400

    X_new = build_feature_vector(season, home_team, away_team)
    proba = rf_model.predict_proba(X_new)[0]
    pred_idx = np.argmax(proba)
    pred_label = target_le.inverse_transform([pred_idx])[0]

    return jsonify({
        "season": season,
        "home_team": home_team,
        "away_team": away_team,
        "prediction": pred_label,
        "probabilities": {cls: float(p) for cls, p in zip(target_le.classes_, proba)}
    })


if __name__ == "__main__":
    print("[INFO] Training model...")
    load_and_train()
    app.run(host="0.0.0.0", port=5000, debug=True)
