from flask import Flask, render_template, request, jsonify
from models.model import get_matchups, predict_points

app = Flask(__name__)

@app.route('/')
def index():
    matchups = get_matchups()  # Get matchups for the dropdown
    return render_template('index.html', matchups=matchups)

@app.route('/predict', methods=['POST'])
def predict():
    matchup = request.json.get('matchup')
    prediction = predict_points(matchup)  # Call predict_points with selected matchup
    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
