from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features).reshape(1, -1)
    prediction = xgb_model.predict(input_data)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
