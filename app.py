import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sqlite3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
#from user import user, init_app

conn = sqlite3.connect('database.db')

# Set the Seaborn style
sns.set(style='whitegrid')

conn = sqlite3.connect('database.db')

# Set the Seaborn style
sns.set(style='whitegrid')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
#init_app(app)
#app.register_blueprint(user.bp)

# Route to render the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to render the upload form
@app.route('/upload-form')
def uploadform():
    upload_message = "No file selected."
    return render_template('upload_form.html', upload_message=upload_message)

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the dashboard page
@app.route('/dashboard')
def dashboard():
    # Load the processed data from the CSV file
    processed_file_path = 'static/pred_expenses.csv'
    df_processed = pd.read_csv(processed_file_path)
    # Render the dashboard template with the processed data
    return render_template('dashboard.html', data=df_processed)

# Route to fetch the updated data for the chart
@app.route('/get-updated-data')
def get_updated_data():
    processed_file_path = 'static/pred_expenses.csv'
    df_processed = pd.read_csv(processed_file_path)
    updated_data = df_processed[['Month', 'RF_Predicted_Expenses', 'GB_Predicted_Expenses', 'DL_Predicted_Expenses']].to_dict(orient='list')
    return jsonify(updated_data)


# Route to handle the file upload and process the data
@app.route('/upload', methods=['POST'])
def uploadfile():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']

    if file.filename == '':
        return "No file selected."

    if file and allowed_file(file.filename):
        # Read the uploaded file as a pandas DataFrame
        expenses_data = pd.read_csv(file)

        # Process the expense data
        # For demonstration, let's assume we want to calculate the total expenses for each month
        expenses_data['Total_Expenses'] = expenses_data.groupby('Month')['Expenses'].transform('sum')

        # Create a Pandas DataFrame from the data
        df_expenses = pd.DataFrame(expenses_data)

        # Set the month column as the index
        df_expenses.set_index('Month', inplace=True)

        # Handling missing values
        imputer = SimpleImputer(strategy='mean')
        df_expenses['Expenses'] = imputer.fit_transform(df_expenses[['Expenses']])

        # Prepare the data for regression
        X = np.arange(1, len(df_expenses.index) + 1).reshape(-1, 1)
        y = df_expenses['Expenses'].values

        # Feature scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the Random Forest and Gradient Boosting models
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_scaled, y)
        gb_regressor.fit(X_scaled, y)

        # Generate predictions for future expenses using deep learning model
        future_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        X_future = np.arange(len(df_expenses.index) + 1, len(df_expenses.index) + len(future_months) + 1).reshape(-1, 1)
        X_future_scaled = scaler.transform(X_future)

        # Generate predictions for Random Forest and Gradient Boosting models
        y_pred_rf = rf_regressor.predict(X_future_scaled)
        y_pred_gb = gb_regressor.predict(X_future_scaled)

        # Define and train the deep learning model
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(1, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled.reshape(-1, 1, 1), y, epochs=100, verbose=0)

        # Generate predictions using the trained deep learning model
        y_pred_dl = model.predict(X_future_scaled.reshape(-1, 1, 1)).flatten()
        y_pred_dl_real = scaler.inverse_transform(y_pred_dl.reshape(-1, 1)).flatten()


        # Create a DataFrame for future expenses with predictions from all models
        df_future_expenses = pd.DataFrame({'Month': future_months, 'RF_Predicted_Expenses': y_pred_rf, 'GB_Predicted_Expenses': y_pred_gb, 'DL_Predicted_Expenses': y_pred_dl})
        df_future_expenses.set_index('Month', inplace=True)

        # Concatenate the historical and future expenses
        df_all_expenses = pd.concat([df_expenses, df_future_expenses])

        # Save the processed data as a CSV file
        processed_file_path = 'static/pred_expenses.csv'
        df_all_expenses.to_csv(processed_file_path, index=False)

        # Define a set of attractive colors for the chart
        bar_colors = cm.tab20.colors[:len(df_all_expenses.index)]
        line_colors = cm.tab20c.colors[:len(future_months)]

        # Set custom colors for each model's predictions
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Plot the monthly expenses with future predictions as a bar chart
        ax = df_all_expenses.plot(kind='bar', color=[colors[0]] * len(df_expenses.index) + [colors[1]] * len(future_months), legend=False)
        plt.title('Monthly Expenses with Future Predictions')
        plt.xlabel('Month')
        plt.ylabel('Amount (£)')

        # Add color labels to the plot
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[0]), plt.Rectangle((0, 0), 1, 1, color=colors[1]), plt.Rectangle((0, 0), 1, 1, color=colors[2])]
        labels = ['Historical Expenses', 'Future Predictions']
        plt.legend(handles, labels)

        # Line plot for future predictions
        x_line = np.arange(len(df_expenses.index) + 1, len(df_expenses.index) + len(future_months) + 1)

        for i, month in enumerate(future_months):
            x_month = np.array([x_line[i]])
            y_rf = np.array([df_future_expenses.loc[month]['RF_Predicted_Expenses']])
            y_gb = np.array([df_future_expenses.loc[month]['GB_Predicted_Expenses']])
            y_dl = np.array([df_future_expenses.loc[month]['DL_Predicted_Expenses']])

            plt.plot(x_month, y_rf, color=colors[0], marker='o', linestyle='--', label=f'RF {month}')
            plt.plot(x_month, y_gb, color=colors[1], marker='o', linestyle='--', label=f'GB {month}')
            plt.plot(x_month, y_dl, color=colors[2], marker='o', linestyle='--', label=f'DL {month}')

        # Save the chart as an image
        plt.savefig('static/chart.png')
        plt.close()

        # Render the template with the chart image and processed data file path
        return render_template('dashboard.html', chart_path='chart.png', data_path=processed_file_path, data=df_all_expenses)
    
    return "Invalid file format."


if __name__ == '__main__':
    app.run(debug=True)
