# Importing necessary libraries
from flask import Flask, render_template, request                     # type: ignore
import pandas as pd                                                   # type: ignore
from sklearn.model_selection import train_test_split                  # type: ignore
from sklearn.ensemble import RandomForestClassifier                   # type: ignore
from sklearn.metrics import accuracy_score                            # type: ignore

# Creates an instance of the Flask class
app = Flask(__name__)

# Initializing variables
# Placeholder for the trained model
model = None 
# Placeholder for predictions
preds = [] 
# Placeholder for accuracy score 
acc = 0 
# Flag indicating whether the model is trained
trained = False 
# Flag indicating whether the training process is completed
train = False 
# Placeholder for min-max normalization ranges
minmax = []  
# Placeholders for train-test split data
X_train, X_test, y_train, y_test = [], [], [], []

# Route for the home page
@app.route('/')
def home():
    # Renders the 'frontend.html'
    return render_template('frontend.html', trained=False, train=False)

# Route for training the model
@app.route('/train', methods=['POST'])
def train_model():
    global preds, acc, trained, minmax, X_train, X_test, y_train, y_test, model
    # Check if a file was uploaded or not
    if 'csv-file' not in request.files:
        message = "No file uploaded!"
        return render_template('frontend.html', message=message)
    # Reads the uploaded CSV file in the label name of csv-file from frontend
    df = pd.read_csv(request.files['csv-file'])
    # Computes min-max normalization ranges for each feature
    for i in df.columns[1:]:
        minmax.append([df[i].min(), df[i].max()])
    # Performs min-max normalization on the features
    for i in df.columns[1:]:
        df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
    # Splits the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2, random_state=42)
    # Trains the random forest model
    model, preds, acc = randomforest(X_train, y_train)
    # Provides the feedback to the user wheather the file is uploaded or not
    message = "File uploaded and model trained successfully!"
    # Renders the 'frontend.html' 
    return render_template('frontend.html', trained=True, model=model, preds=preds, acc=acc, train=True, message=message)

# Function to train the random forest model
def randomforest(X_train, y_train):
    global preds, acc, trained, minmax, X_test, y_test, model
    # Initializes and train the random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    # Makes the predictions on the test set
    rf_preds = rf_model.predict(X_test)
    # Computes accuracy score
    rf_acc = accuracy_score(y_test, rf_preds)
    return rf_model, rf_preds, rf_acc

# Route for predicting churn
@app.route('/predict', methods=['POST'])
def predict():
    global model, minmax
    if request.method == 'POST':
        # Extracting the feature values from the form and perform min-max normalization
        features = []  
        features.append((float(request.form['E-Comm-Tenure']) - minmax[0][0]) / (minmax[0][1] - minmax[0][0]))  # Normalizing 'Tenure'
        features.append(float(request.form['Preferred-Login-Device']))  # No normalization needed for 'PreferredLoginDevice'
        features.append((float(request.form['Warehouse-To-Home']) - minmax[3][0]) / (minmax[3][1] - minmax[3][0]))  # Normalizing 'WarehouseToHome'
        features.append(float(request.form['Preferred-Payment-Mode']))  # No normalization needed for 'PreferredPaymentMode'
        features.append((float(request.form['Hour-Spend-On-App']) - minmax[4][0]) / (minmax[4][1] - minmax[4][0]))  # Normalizing 'HourSpendOnApp'
        features.append((float(request.form['Number-Of-Device-Registered']) - minmax[5][0]) / (minmax[5][1] - minmax[5][0]))  # Normalizing 'NumberOfDeviceRegistered'
        features.append((float(request.form['Satisfaction-Score']) - minmax[6][0]) / (minmax[6][1] - minmax[6][0]))  # Normalizing 'SatisfactionScore'
        features.append(float(request.form['City-Tier-Level']))  # No normalization needed for 'CityTier'
        features.append(float(request.form['CashbackAmount']))  # No normalization needed for 'CashbackAmount'
        features.append((float(request.form['Marital-Status']) - minmax[7][0]) / (minmax[7][1] - minmax[7][0]))  # Normalizing 'MaritalStatus'
        features.append(float(request.form['Complain']))  # No normalization needed for 'Complain'
        features.append(float(request.form['Preferred-Order-Category']))  # No normalization needed for 'PreferedOrderCategory'
        # Reshapes the features
        features = [features]
        # Makes prediction using the trained model
        prediction = model.predict(features)[0]
        # Returns the prediction to frontend to display the results
        return str(prediction)

# Runs the Flask application
if __name__ == '__main__':
    app.run(debug=True)