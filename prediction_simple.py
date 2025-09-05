from flask import Blueprint, request, jsonify
import json
import random
from flask_cors import cross_origin

prediction_bp = Blueprint('prediction', __name__)

# Sample teams data
TEAMS = [
    "Brazil", "Argentina", "Germany", "France", "Spain", "Italy", "England", 
    "Netherlands", "Portugal", "Belgium", "Croatia", "Uruguay", "Colombia", 
    "Mexico", "Chile", "Peru", "Ecuador", "Paraguay", "Venezuela", "Bolivia",
    "Egypt", "Morocco", "Nigeria", "Ghana", "Senegal", "Algeria", "Tunisia",
    "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia", "Qatar",
    "United States", "Canada", "Costa Rica", "Jamaica", "Panama", "Honduras"
]

# Simple team stats (mock data)
TEAM_STATS = {
    "Brazil": {"win_rate": 0.75, "avg_scored": 2.1, "avg_conceded": 0.8},
    "Argentina": {"win_rate": 0.70, "avg_scored": 2.0, "avg_conceded": 0.7},
    "Germany": {"win_rate": 0.68, "avg_scored": 1.9, "avg_conceded": 0.9},
    "France": {"win_rate": 0.72, "avg_scored": 2.2, "avg_conceded": 0.8},
    "Spain": {"win_rate": 0.65, "avg_scored": 1.8, "avg_conceded": 0.7},
    "Italy": {"win_rate": 0.63, "avg_scored": 1.7, "avg_conceded": 0.8},
    "England": {"win_rate": 0.60, "avg_scored": 1.9, "avg_conceded": 1.0},
    "Netherlands": {"win_rate": 0.62, "avg_scored": 1.8, "avg_conceded": 0.9},
    "Portugal": {"win_rate": 0.58, "avg_scored": 1.6, "avg_conceded": 0.8},
    "Belgium": {"win_rate": 0.55, "avg_scored": 1.7, "avg_conceded": 1.1}
}

def get_team_stats(team):
    """Get team statistics or return default values"""
    if team in TEAM_STATS:
        return TEAM_STATS[team]
    else:
        # Return random stats for unknown teams
        return {
            "win_rate": round(random.uniform(0.3, 0.7), 2),
            "avg_scored": round(random.uniform(0.8, 2.5), 1),
            "avg_conceded": round(random.uniform(0.5, 2.0), 1)
        }

def predict_match_result(home_team, away_team):
    """Simple prediction logic based on team stats"""
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    
    # Calculate strength difference
    home_strength = home_stats["win_rate"] + (home_stats["avg_scored"] - home_stats["avg_conceded"]) * 0.1
    away_strength = away_stats["win_rate"] + (away_stats["avg_scored"] - away_stats["avg_conceded"]) * 0.1
    
    # Add home advantage
    home_strength += 0.1
    
    # Calculate probabilities
    total_strength = home_strength + away_strength
    if total_strength > 0:
        home_prob = home_strength / total_strength
        away_prob = away_strength / total_strength
    else:
        home_prob = away_prob = 0.4
    
    # Draw probability (inverse of strength difference)
    strength_diff = abs(home_strength - away_strength)
    draw_prob = max(0.15, 0.4 - strength_diff)
    
    # Normalize probabilities
    total_prob = home_prob + away_prob + draw_prob
    home_prob = home_prob / total_prob
    away_prob = away_prob / total_prob
    draw_prob = draw_prob / total_prob
    
    # Determine prediction
    if home_prob > away_prob and home_prob > draw_prob:
        prediction = "Home Win"
    elif away_prob > home_prob and away_prob > draw_prob:
        prediction = "Away Win"
    else:
        prediction = "Draw"
    
    return {
        "prediction": prediction,
        "probabilities": {
            "Home Win": round(home_prob, 3),
            "Away Win": round(away_prob, 3),
            "Draw": round(draw_prob, 3)
        },
        "home_team_stats": home_stats,
        "away_team_stats": away_stats
    }

@prediction_bp.route('/predict', methods=['POST'])
@cross_origin()
def predict_match():
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Both home_team and away_team are required'}), 400
        
        result = predict_match_result(home_team, away_team)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/teams', methods=['GET'])
@cross_origin()
def get_teams():
    try:
        return jsonify({'teams': sorted(TEAMS)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

