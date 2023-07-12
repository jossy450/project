import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.cm as cm

app = Flask(__name__)

# Route to render the home page
@app.route('/')
def home():
    return render_template('/home.html')

# Route to render the upload form
@app.route('/upload-form')
def uploadform():
    upload_message = "No file selected."
    return render_template('/upload_form.html', upload_message=upload_message)

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
    return render_template('/dashboard.html', data=df_processed)

# Route to handle the file upload and process the data
@app.route('/upload', methods=['POST'])
def upload():
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

        # Generate predictions for future expenses
        future_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        X_future = np.arange(len(df_expenses.index) + 1, len(df_expenses.index) + len(future_months) + 1).reshape(-1, 1)
        X_future_scaled = scaler.transform(X_future)

        y_pred_rf = rf_regressor.predict(X_future_scaled)
        y_pred_gb = gb_regressor.predict(X_future_scaled)

        # Determine the best prediction model for each month
        best_models = []
        for i, month in enumerate(future_months):
            rf_pred = y_pred_rf[i]
            gb_pred = y_pred_gb[i]
            best_model = 'Random Forest' if rf_pred < gb_pred else 'Gradient Boosting'
            best_models.append(best_model)

        # Create a DataFrame for future expenses with best prediction models
        df_future_expenses = pd.DataFrame({'Month': future_months, 'RF_Predicted_Expenses': y_pred_rf, 'GB_Predicted_Expenses': y_pred_gb, 'Best_Model': best_models})
        df_future_expenses.set_index('Month', inplace=True)

        # Concatenate the historical and future expenses
        df_all_expenses = pd.concat([df_expenses, df_future_expenses])

        # Save the processed data as a CSV file
        processed_file_path = 'static/pred_expenses.csv'
        df_all_expenses.to_csv(processed_file_path, index=False)

        # Define a set of attractive colors for the chart
        bar_colors = cm.tab20.colors[:len(df_all_expenses.index)]
        line_colors = cm.tab20c.colors[:len(future_months)]

        # Plot the monthly expenses with future predictions as a bar chart
        ax = df_all_expenses.plot(kind='bar', color=bar_colors, legend=False)
        plt.title('Monthly Expenses with Future Predictions')
        plt.xlabel('Month')
        plt.ylabel('Amount (£)')

        # Add color labels to the plot
        handles = [plt.Rectangle((0, 0), 1, 1, color=bar_colors[i]) for i in range(len(df_all_expenses.index))]
        labels = df_all_expenses.index.tolist()
        plt.legend(handles, labels)

        # Line plot for future predictions
        x_line = np.arange(len(df_expenses.index) + 1, len(df_expenses.index) + len(future_months) + 1)

        for i, month in enumerate(df_future_expenses.index):
            x_month = np.array([x_line[i]])
            y_rf = np.array([df_future_expenses.loc[month]['RF_Predicted_Expenses']])
            y_gb = np.array([df_future_expenses.loc[month]['GB_Predicted_Expenses']])

            plt.plot(x_month, y_rf, color=line_colors[i], marker='o', linestyle='--', label=f'RF {month}')
            plt.plot(x_month, y_gb, color=line_colors[i], marker='o', linestyle='--', label=f'GB {month}')

        # Save the chart as an image
        plt.savefig('static/chart.png')
        plt.close()

        # Render the template with the chart image and processed data file path
        return render_template('dashboard.html', chart_path='chart.png', data_path=processed_file_path, data=df_all_expenses)

    return "Invalid file format."


if __name__ == '__main__':
    app.run(debug=True)