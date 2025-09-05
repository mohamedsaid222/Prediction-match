from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import cross_origin

prediction_bp = Blueprint('prediction', __name__)

# Load the model and features
model = joblib.load('src/match_prediction_model.pkl')
features = joblib.load('src/model_features.pkl')

def calculate_team_stats(data, team, n_matches=10):
    """Calculate team statistics from historical data"""
    team_matches = data[(data['home_team'] == team) | (data['away_team'] == team)]
    
    if len(team_matches) == 0:
        return 0.0, 0.0, 0.0
    
    results = []
    goals_scored = []
    goals_conceded = []
    
    for _, row in team_matches.tail(n_matches).iterrows():
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
    
    win_rate = results.count('Win') / len(results) if results else 0.0
    avg_scored = np.mean(goals_scored) if goals_scored else 0.0
    avg_conceded = np.mean(goals_conceded) if goals_conceded else 0.0
    
    return win_rate, avg_scored, avg_conceded

@prediction_bp.route('/predict', methods=['POST'])
@cross_origin()
def predict_match():
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Both home_team and away_team are required'}), 400
        
        # Load historical data
        df = pd.read_csv('src/results.csv')
        
        # Calculate team statistics
        home_stats = calculate_team_stats(df, home_team)
        away_stats = calculate_team_stats(df, away_team)
        
        # Prepare features for prediction
        prediction_features = {
            'home_win_rate': home_stats[0],
            'home_avg_scored': home_stats[1],
            'home_avg_conceded': home_stats[2],
            'away_win_rate': away_stats[0],
            'away_avg_scored': away_stats[1],
            'away_avg_conceded': away_stats[2],
            'win_rate_diff': home_stats[0] - away_stats[0],
            'goal_scored_diff': home_stats[1] - away_stats[1],
            'goal_conceded_diff': home_stats[2] - away_stats[2]
        }
        
        # Create DataFrame for prediction
        X = pd.DataFrame([prediction_features])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get class names
        classes = model.classes_
        
        # Create probability dictionary
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        return jsonify({
            'prediction': prediction,
            'probabilities': prob_dict,
            'home_team_stats': {
                'win_rate': home_stats[0],
                'avg_scored': home_stats[1],
                'avg_conceded': home_stats[2]
            },
            'away_team_stats': {
                'win_rate': away_stats[0],
                'avg_scored': away_stats[1],
                'avg_conceded': away_stats[2]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/teams', methods=['GET'])
@cross_origin()
def get_teams():
    try:
        # Load historical data
        df = pd.read_csv('src/results.csv')
        
        # Get unique teams
        teams = sorted(list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist())))
        
        return jsonify({'teams': teams})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

