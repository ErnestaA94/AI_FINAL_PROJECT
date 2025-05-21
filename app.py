from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

model = None
X_scaler = None
y_scaler = None
player_features = None
players_data = None

FEATURE_COLUMNS = [
    'PTS','REB','AST','STL','BLK','MIN',
    'FG_PCT','FT_PCT','FG3_PCT',
    'PTS_AVG_5','REB_AVG_5','AST_AVG_5',
    'TS_PCT','USAGE_RATE','PER',
    'DAYS_REST','B2B','GAMES_LAST_7D',
    'SEASON_GAME_NUM','IS_HOME'
]

def load_resources():
    """Load all resources"""
    global model, X_scaler, y_scaler, player_features, players_data
    
    print("Loading resources...")
    
    try:
        # Load player features
        player_features = pd.read_csv('../results/player_features.csv', low_memory=False)
        print(f"✓ Player features loaded: {player_features.shape}")
        
        # Create players data
        players_data = player_features.groupby('PLAYER_ID').agg({
            'PLAYER_NAME': 'first',
            'GAME_ID': 'count'
        }).reset_index()
        players_data.columns = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_COUNT']

        from tensorflow.keras.models import load_model # type: ignore
        from tensorflow.keras.losses import MeanSquaredError # type: ignore
        # Load model
        model = load_model("../results/best_model.h5", 
                   custom_objects={"mse": MeanSquaredError()})
        print(f"✓ Model loaded successfully")
        
        # Load scalers
        X_scaler = joblib.load('../results/X_scaler.pkl')
        y_scaler = joblib.load('../results/y_scaler.pkl')
        print(f"✓ Scalers loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"Error loading resources: {e}")
        return False

def prepare_player_sequence(player_id, sequence_length=10):
    """Prepare input sequence for a player"""
    player_data = player_features[player_features['PLAYER_ID'] == player_id].copy()
    
    if len(player_data) < sequence_length:
        return None, "Not enough games for prediction"
    
    # Sort by date
    if 'GAME_DATE_EST' in player_data.columns:
        player_data['GAME_DATE_EST'] = pd.to_datetime(player_data['GAME_DATE_EST'])
        player_data = player_data.sort_values('GAME_DATE_EST')
    else:
        player_data = player_data.sort_values('GAME_ID')
    
    # Get last sequence_length games
    recent_games = player_data.tail(sequence_length)
    
    # Create sequence
    X = recent_games[FEATURE_COLUMNS].values
    X = np.nan_to_num(X, 0)
    
    return X.reshape(1, sequence_length, len(FEATURE_COLUMNS)), None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/players')
def get_players():
    if players_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    players_list = players_data.to_dict('records')
    players_list.sort(key=lambda x: (-x['GAME_COUNT'], x['PLAYER_NAME']))
    
    return jsonify(players_list)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    player_id = int(data.get('player_id'))
    
    # Get player info
    player_info = players_data[players_data['PLAYER_ID'] == player_id]
    if player_info.empty:
        return jsonify({'error': 'Player not found'}), 404
    
    player_name = player_info.iloc[0]['PLAYER_NAME']
    
    # Get recent stats
    player_games = player_features[player_features['PLAYER_ID'] == player_id]
    recent_avg = player_games['PTS'].tail(5).mean()
    
    try:
        # Prepare sequence
        X, error = prepare_player_sequence(player_id)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Scale and predict
        X_reshaped = X.reshape(-1, len(FEATURE_COLUMNS))
        X_scaled = X_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(1, 10, len(FEATURE_COLUMNS))
        
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        predicted = float(y_pred[0][0])
        
        confidence = 'High' if abs(predicted - recent_avg) < 5 else 'Medium'
        
        return jsonify({
            'player_id': player_id,
            'player_name': player_name,
            'predicted_points': predicted,
            'recent_average': recent_avg,
            'difference': predicted - recent_avg,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/player/<int:player_id>/history')
def player_history(player_id):
    player_data = player_features[player_features['PLAYER_ID'] == player_id]
    
    if player_data.empty:
        return jsonify({'error': 'Player not found'}), 404
    
    # Sort by date
    if 'GAME_DATE_EST' in player_data.columns:
        player_data = player_data.sort_values('GAME_DATE_EST')
    
    recent_games = player_data.tail(20)
    
    history = []
    for idx, (_, row) in enumerate(recent_games.iterrows()):
        history.append({
            'game': idx + 1,
            'points': float(row.get('PTS', 0)),
            'rebounds': float(row.get('REB', 0)),
            'assists': float(row.get('AST', 0)),
            'minutes': float(row.get('MIN', 0))
        })
    
    return jsonify(history)

if __name__ == '__main__':
    if load_resources():
        print("\n✓ All resources loaded!")
        print("Starting server...")
        print("Visit: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load resources")