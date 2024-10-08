from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('final_iforest_model.pkl')
div_name_encoder = joblib.load('div_name_encoder.pkl')
merchant_encoder = joblib.load('merchant_encoder.pkl')
cat_desc_encoder = joblib.load('cat_desc_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data with validation
        fiscal_yr = int(request.form['FISCAL_YR'])
        fiscal_mth = int(request.form['FISCAL_MTH'])
        div_name = request.form['DIV_NAME']
        merchant = request.form['MERCHANT']
        cat_desc = request.form['CAT_DESC']
        amt = float(request.form['AMT'])
        year = int(request.form['Year'])
        month = int(request.form['Month'])
        dayofweek = int(request.form['DayOfWeek'])
        fiscalquarter = int(request.form['FiscalQuarter'])

        # Encode categorical inputs with handling for unseen categories
        div_name_encoded = encode_input(div_name, div_name_encoder, "Division name")
        merchant_encoded = encode_input(merchant, merchant_encoder, "Merchant")
        cat_desc_encoded = encode_input(cat_desc, cat_desc_encoder, "Category description")

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[fiscal_yr, fiscal_mth, div_name_encoded, merchant_encoded, cat_desc_encoded, amt, year, month, dayofweek, fiscalquarter]], 
                                  columns=['FISCAL_YR', 'FISCAL_MTH', 'DIV_NAME', 'MERCHANT', 'CAT_DESC', 'AMT', 'Year', 'Month', 'DayOfWeek', 'FiscalQuarter'])

        # Make a prediction
        prediction = model.predict(input_data)

        # Convert prediction to human-readable format
        if prediction[0] == 1:
            result = "This is a normal transaction."
        else:
            result = "This transaction is anomalous!"

        return render_template('index.html', prediction_text=result)

    except ValueError as e:
        # Handle any other type conversion errors or unexpected input
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

def encode_input(input_value, encoder, input_name):
    try:
        return encoder.transform([input_value])[0]
    except ValueError:
        return -1  # Assign an "Unknown" category label or a default encoding

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)