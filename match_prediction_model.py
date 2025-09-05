
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

def calculate_team_stats(data, n_matches=10):
    stats = {}
    for team in pd.concat([data['home_team'], data['away_team']]).unique():
        team_matches = data[(data['home_team'] == team) | (data['away_team'] == team)]
        results = []
        goals_scored = []
        goals_conceded = []
        for i, row in team_matches.iterrows():
            if row['home_team'] == team:
                scored = row['home_score']
                conceded = row['away_score']
                res = 'Win' if row['home_score'] > row['away_score'] else ('Draw' if row['home_score'] == row['away_score'] else 'Loss')
            else:
                scored = row['away_score']
                conceded = row['home_score']
                res = 'Win' if row['away_score'] > row['home_score'] else ('Draw' if row['away_score'] == row['home_score'] else 'Loss')
            results.append(res)
            goals_scored.append(scored)
            goals_conceded.append(conceded)
            last_n = results[-n_matches:]
            win_rate = last_n.count('Win') / len(last_n)
            avg_scored = np.mean(goals_scored[-n_matches:])
            avg_conceded = np.mean(goals_conceded[-n_matches:])
            stats[(i, team)] = (win_rate, avg_scored, avg_conceded)
    return stats

if __name__ == "__main__":
    df = pd.read_csv("results.csv")
    df['result'] = df.apply(match_result, axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    team_stats = calculate_team_stats(df, n_matches=10)

    features = []
    for i, row in df.iterrows():
        if (i, row['home_team']) in team_stats and (i, row['away_team']) in team_stats:
            home_stats = team_stats[(i, row['home_team'])]
            away_stats = team_stats[(i, row['away_team'])]

            features.append({
                'home_win_rate': home_stats[0],
                'home_avg_scored': home_stats[1],
                'home_avg_conceded': home_stats[2],
                'away_win_rate': away_stats[0],
                'away_avg_scored': away_stats[1],
                'away_avg_conceded': away_stats[2],
                'result': row['result']
            })

    features_df = pd.DataFrame(features)

    X = features_df.drop('result', axis=1)
    y = features_df['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    import joblib
    joblib.dump(model, 'match_prediction_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_features.pkl')



