from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-fitted model, transformers, and label encoders
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
numeric_transformer = pickle.load(open('numeric_transformer.pkl', 'rb'))
oh_transformer = pickle.load(open('oh_transformer.pkl', 'rb'))
le_brand = pickle.load(open('le_brand.pkl', 'rb'))
le_model = pickle.load(open('le_model.pkl', 'rb'))

# Define column names for transformers
num_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
cat_features = ['seller_type', 'fuel_type', 'transmission_type', 'brand', 'model']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data
    brand = request.form.get('brand')
    model = request.form.get('model')
    vehicle_age = int(request.form.get('vehicle_age'))
    km_driven = int(request.form.get('km_driven'))
    seller_type = request.form.get('seller_type')
    fuel_type = request.form.get('fuel_type')
    transmission_type = request.form.get('transmission_type')
    mileage = float(request.form.get('mileage'))
    engine = int(request.form.get('engine'))
    max_power = float(request.form.get('max_power'))
    seats = int(request.form.get('seats'))

    # Prepare data for prediction
    data = pd.DataFrame([[vehicle_age, km_driven, mileage, engine, max_power, seats, seller_type,
                          fuel_type, transmission_type, brand, model]],
                        columns=['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 
                                 'seats', 'seller_type', 'fuel_type', 'transmission_type', 'brand', 'model'])

    # Check if brand and model are known; if not, handle appropriately
    if brand not in le_brand.classes_:
        return render_template('index.html', predicted_price="Error: Unseen brand. Please select a known brand.")
    if model not in le_model.classes_:
        return render_template('index.html', predicted_price="Error: Unseen model. Please select a known model.")

    # Transform brand and model using the label encoders
    data['brand'] = le_brand.transform([brand])[0]
    data['model'] = le_model.transform([model])[0]

    # Transform data
    numeric_data = numeric_transformer.transform(data[num_features])
    categorical_data = oh_transformer.transform(data[cat_features]).toarray()
    final_data = np.hstack([numeric_data, categorical_data])

    # Make prediction
    prediction = rf_model.predict(final_data)

    # Render the result at the bottom of the form
    return render_template('index.html', predicted_price=f"Estimated Price: ₹{round(prediction[0], 2)}")

if __name__ == "__main__":
    app.run(debug=True)
