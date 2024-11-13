from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from num2words import num2words  # Install this package with `pip install num2words`

app = Flask(__name__)

# Load the trained model
pipeline = joblib.load('car_price_model.pkl')

# Load the dataset to get unique values for dropdowns
df = pd.read_csv('car_price.csv').drop(columns=['Unnamed: 0'])

@app.route('/')
def home():
    # Render the HTML template for the home page
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def get_data():
    # Pass unique values for dropdowns to the frontend
    data = {
        'car_names': sorted(df['car_name'].unique()),
        'fuel_types': sorted(df['fuel_type'].unique()),
        'transmissions': sorted(df['transmission'].unique()),
        'ownerships': [opt for opt in df['ownership'].unique() if opt != "0th owner"]
    }
    return jsonify(data)

def convert_to_words(price):
    # Separate rupees and paise (decimal places)
    rupees = int(price)
    paise = int((price - rupees) * 100)
    
    # Convert rupees and paise to words
    rupees_in_words = num2words(rupees, lang='en_IN').capitalize()
    if paise > 0:
        paise_in_words = num2words(paise, lang='en_IN') + " paise"
        return f"{rupees_in_words} rupees and {paise_in_words}"
    else:
        return f"{rupees_in_words} rupees"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = request.form
    car_name = data['car_name']
    manufacture = int(data['manufacture'])
    fuel_type = data['fuel_type']
    seats = int(data['seats'])
    kms_driven = int(data['kms_driven'])
    transmission = data['transmission']
    ownership = data['ownership']
    engine = int(data['engine'])

    # Create a DataFrame for the input
    user_input = {
        'car_name': car_name,
        'kms_driven': kms_driven,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'ownership': ownership,
        'manufacture': manufacture,
        'engine': engine,
        'Seats': seats
    }
    user_df = pd.DataFrame([user_input])

    # Predict price
    predicted_price = pipeline.predict(user_df)[0]
    
    # Format predicted price as Indian Rupee and convert to words
    formatted_price = f"₹{predicted_price:,.2f}"
    price_in_words = convert_to_words(predicted_price)
    
    return jsonify({
        'predicted_price': formatted_price,
        'predicted_price_in_words': price_in_words
    })

if __name__ == '__main__':
    app.run(debug=True)
