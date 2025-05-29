import pickle
import os
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Define features with their input configurations
features = {
    'Age': {'type': 'number', 'min': 18, 'max': 70, 'description': 'Employee age in years'},
    'DailyRate': {
        'type': 'number', 
        'min': 0,
        'description': 'Daily pay rate in USD ($)',
        'unit': 'USD'
    },
    'DistanceFromHome': {
        'type': 'number', 
        'min': 0,
        'description': 'Distance in kilometers',
        'unit': 'KM'
    },
    'EnvironmentSatisfaction': {
        'type': 'select',
        'options': [1, 2, 3, 4, 5],
        'description': 'Rate between least (1) - most (5) satisfied'
    },
    'Gender': {
        'type': 'select',
        'options': ['Male', 'Female'],
        'description': 'Male or Female'
    },
    'JobInvolvement': {
        'type': 'select',
        'options': [1, 2, 3, 4, 5],
        'description': 'Rate between least (1) - most (5) involved'
    },
    'JobLevel': {
        'type': 'select',
        'options': [1, 2, 3, 4, 5],
        'description': 'Between least (1) - most (5) senior'
    },
    'JobSatisfaction': {
        'type': 'select',
        'options': [1, 2, 3, 4, 5],
        'description': 'Rate between least (1) - most (5) satisfied'
    },
    'OverTime': {
        'type': 'radio',
        'options': ['Yes', 'No'],
        'description': 'Works overtime? (Yes or No)'
    },
    'RelationshipSatisfaction': {
        'type': 'select',
        'options': [1, 2, 3, 4, 5],
        'description': 'Rate between least (1) - most (5) satisfied'
    },
    'StockOptionLevel': {'type': 'number', 'min': 0, 'max': 3, 'description': '0-3 (highest)'},
    'TotalWorkingYears': {'type': 'number', 'min': 0, 'description': 'Total years worked'},
    'WorkLifeBalance': {
        'type': 'select',
        'options': [1, 2, 3, 4],
        'description': 'Rate between worst (1) - best (4) balance'
    },
    'YearsAtCompany': {'type': 'number', 'min': 0, 'description': 'Years at current company'},
    'YearsInCurrentRole': {'type': 'number', 'min': 0, 'description': 'Years in current role'}
}

# Define the expected feature order for the model
model_feature_order = [
    'Age', 'DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction', 
    'Gender', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'OverTime',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole'
]

# Load model
model_path = os.path.join('model', 'xgb_attrition_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        missing_fields = []
        invalid_fields = []
        
        for feature in model_feature_order:
            value = request.form.get(feature)
            
            # Check for missing required fields
            if value is None or value == '':
                missing_fields.append(feature)
                continue
                
            # Handle special cases
            if feature == 'Gender':
                input_data.append(1 if value == 'Male' else 0)
            elif feature == 'OverTime':
                input_data.append(1 if value == 'Yes' else 0)
            else:
                try:
                    # Convert to float (works for both string numbers and actual numbers)
                    num_value = float(value)
                    # Validate min/max if specified
                    config = features.get(feature, {})
                    if 'min' in config and num_value < config['min']:
                        invalid_fields.append(f"{feature} (min {config['min']})")
                    if 'max' in config and num_value > config['max']:
                        invalid_fields.append(f"{feature} (max {config['max']})")
                    input_data.append(num_value)
                except ValueError:
                    invalid_fields.append(f"{feature} (invalid number)")
        
        # Check for any missing or invalid fields
        if missing_fields or invalid_fields:
            error_messages = []
            if missing_fields:
                error_messages.append(f"Missing fields: {', '.join(missing_fields)}")
            if invalid_fields:
                error_messages.append(f"Invalid values: {', '.join(invalid_fields)}")
            return render_template('index.html',
                                features=features,
                                prediction_text=" | ".join(error_messages),
                                result_class="error")

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0][1] * 100  # Probability of attrition
        
        # Prepare result
        if prediction[0] == 1:
            result_text = f"High risk of attrition ({probability:.1f}% probability)"
            result_class = "high-risk"
        else:
            result_text = f"Low risk of attrition ({probability:.1f}% probability)"
            result_class = "low-risk"
            
        return render_template('index.html', 
                            features=features,
                            prediction_text=result_text,
                            probability=f"{probability:.1f}",
                            result_class=result_class)
        
    except Exception as e:
        return render_template('index.html',
                            features=features,
                            prediction_text=f"System error: {str(e)}",
                            result_class="error")

if __name__ == '__main__':
    app.run(debug=True)