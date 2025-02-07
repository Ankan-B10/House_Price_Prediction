from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel pkl", 'rb'))  # Fix filename if needed


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(f"Received inputs - Location: {location}, BHK: {bhk}, Bath: {bath}, Sqft: {sqft}")

    # Validate and convert inputs
    if not location or not bhk or not bath or not sqft:
        return "Error: Please provide all input values."

    try:
        bhk = float(bhk)
        bath = float(bath)
        sqft = float(sqft)
    except ValueError:
        return "Error: Please enter valid numeric values."

    # Prepare input for model
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    try:
        prediction = pipe.predict(input_data)[0] * 1e5
        return str(np.round(prediction, 2))
    except Exception as e:
        return f"Prediction error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True, port=5005)
