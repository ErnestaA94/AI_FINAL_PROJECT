from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
model = None
X_scaler = None
y_scaler = None
player_features = None
players_data = None

# Features for the model
FEATURE_COLUMNS = [
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN',
    'FG_PCT', 'FT_PCT', 'FG3_PCT',
    'PTS_AVG_3', 'REB_AVG_3', 'AST_AVG_3',
    'STL_AVG_3', 'BLK_AVG_3', 'MIN_AVG_3',
    'FG_PCT_AVG_3', 'FT_PCT_AVG_3',
    'PTS_AVG_5', 'REB_AVG_5', 'AST_AVG_5'
]

def load_resources():
    """Load all resources with validation"""
    global model, X_scaler, y_scaler, player_features, players_data
    
    print("Loading resources...")
    
    try:
        # Load player features
        player_features = pd.read_csv('results/player_features.csv', low_memory=False)
        print(f"✓ Player features loaded: {player_features.shape}")
        
        # Validate data
        print(f"  - PTS mean: {player_features['PTS'].mean():.2f}")
        print(f"  - PTS std: {player_features['PTS'].std():.2f}")
        
        # Create players data
        players_data = player_features.groupby('PLAYER_ID').agg({
            'PLAYER_NAME': 'first',
            'GAME_ID': 'count'
        }).reset_index()
        players_data.columns = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_COUNT']
        
        # Load model
        model = tf.keras.models.load_model(
            'results/best_model.h5',
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        print("✓ Model loaded successfully")
        print(f"  - Input shape: {model.input_shape}")

        # Load scalers
        X_scaler = joblib.load('results/X_scaler.pkl')
        y_scaler = joblib.load('results/y_scaler.pkl')
        print(f"✓ Scalers loaded successfully")
        print(f"  - Y scaler mean: {y_scaler.mean_[0]:.2f}")
        print(f"  - Y scaler scale: {y_scaler.scale_[0]:.2f}")
        
        # WARNING CHECK
        if y_scaler.mean_[0] < 15:  # If mean is suspiciously low
            print("\n⚠️  WARNING: Y scaler appears to be trained on different data!")
            print("    Expected PTS mean ~20+, but scaler shows ~10")
            print("    Predictions may be incorrect!")
        
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

def adjust_prediction(predicted, recent_avg, player_name):
    """Adjust predictions that seem unrealistic due to scaling issues"""
    
    # If prediction is unrealistically low compared to recent average
    if predicted < recent_avg * 0.3:  # Less than 30% of recent average
        print(f"⚠️  Adjusting unrealistic prediction for {player_name}")
        print(f"   Raw prediction: {predicted:.2f}, Recent avg: {recent_avg:.2f}")
        
        # Use a weighted average heavily favoring recent performance
        adjusted = recent_avg * 0.8 + predicted * 0.2
        return adjusted, "Low confidence (adjusted)"
    
    # If prediction is too high
    elif predicted > recent_avg * 2.5:  # More than 250% of recent average
        adjusted = recent_avg * 1.2
        return adjusted, "Low confidence (adjusted)"
    
    # Normal range
    confidence = 'High' if abs(predicted - recent_avg) < 5 else 'Medium'
    return predicted, confidence

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
    debug = data.get('debug', False)
    
    # Get player info
    player_info = players_data[players_data['PLAYER_ID'] == player_id]
    if player_info.empty:
        return jsonify({'error': 'Player not found'}), 404
    
    player_name = player_info.iloc[0]['PLAYER_NAME']
    
    # Get recent stats
    player_games = player_features[player_features['PLAYER_ID'] == player_id]
    recent_avg = player_games['PTS'].tail(10).mean()
    season_avg = player_games['PTS'].mean()
    
    try:
        # Prepare sequence
        X, error = prepare_player_sequence(player_id)
        
        if error:
            return jsonify({'error': error}), 400
        
        if debug:
            print(f"\nDebug - Player: {player_name} (ID: {player_id})")
            print(f"Debug - Recent avg: {recent_avg:.2f}, Season avg: {season_avg:.2f}")
            print(f"Debug - Sequence shape: {X.shape}")
            print(f"Debug - First game features: {X[0][0][:5]}")
        
        # Scale and predict
        X_reshaped = X.reshape(-1, len(FEATURE_COLUMNS))
        X_scaled = X_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(1, 10, len(FEATURE_COLUMNS))
        
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        
        if debug:
            print(f"Debug - Scaled prediction: {y_pred_scaled[0][0]:.6f}")
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        raw_predicted = float(y_pred[0][0])
        
        if debug:
            print(f"Debug - Raw prediction: {raw_predicted:.2f}")
        
        # Adjust prediction if needed
        predicted, confidence = adjust_prediction(raw_predicted, recent_avg, player_name)
        
        if debug and predicted != raw_predicted:
            print(f"Debug - Adjusted prediction: {predicted:.2f}")
        
        response = {
            'player_id': player_id,
            'player_name': player_name,
            'predicted_points': round(predicted, 1),
            'recent_average': round(recent_avg, 1),
            'season_average': round(season_avg, 1),
            'difference': round(predicted - recent_avg, 1),
            'confidence': confidence
        }
        
        if debug:
            response['debug_info'] = {
                'raw_prediction': round(raw_predicted, 2),
                'scaled_prediction': float(y_pred_scaled[0][0]),
                'adjustment_applied': predicted != raw_predicted
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
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
            'date': str(row.get('GAME_DATE_EST', '')),
            'points': float(row.get('PTS', 0)),
            'rebounds': float(row.get('REB', 0)),
            'assists': float(row.get('AST', 0)),
            'minutes': float(row.get('MIN', 0))
        })
    
    # Add summary stats
    summary = {
        'last_10_avg': float(player_data['PTS'].tail(10).mean()),
        'season_avg': float(player_data['PTS'].mean()),
        'max_points': float(player_data['PTS'].max()),
        'min_points': float(player_data['PTS'].min())
    }
    
    return jsonify({
        'history': history,
        'summary': summary
    })

@app.route('/model/info')
def model_info():
    """Get model and data information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model': {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params()
        },
        'scalers': {
            'X_mean': X_scaler.mean_[:5].tolist(),
            'X_scale': X_scaler.scale_[:5].tolist(),
            'y_mean': float(y_scaler.mean_[0]),
            'y_scale': float(y_scaler.scale_[0])
        },
        'data': {
            'total_games': len(player_features),
            'total_players': player_features['PLAYER_ID'].nunique(),
            'pts_mean': float(player_features['PTS'].mean()),
            'pts_std': float(player_features['PTS'].std())
        },
        'warnings': []
    }
    
    # Add warnings
    if y_scaler.mean_[0] < 15:
        info['warnings'].append('Y scaler mean is suspiciously low - model may be trained on different data')
    
    return jsonify(info)

if __name__ == '__main__':
    if load_resources():
        print("\n✓ All resources loaded!")
        print("\n⚠️  Note: If you're seeing unrealistic predictions,")
        print("   the model may have been trained on filtered data.")
        print("   Consider retraining with full dataset.\n")
        print("Starting server...")
        print("Visit: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load resources")