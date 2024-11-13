import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # For saving and loading the model
import tkinter as tk
from tkinter import ttk
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset and prepare it
df = pd.read_csv('car_price.csv').drop(columns=['Unnamed: 0'])

# Convert price to numeric values
def convert_price(price):
    if 'Lakh' in price:
        return float(price.replace(' Lakh', '')) * 1e5
    elif 'Crore' in price:
        return float(price.replace(' Crore', '')) * 1e7
    else:
        return 0.0

# Apply conversions to the DataFrame
df['car_prices_in_rupee'] = df['car_prices_in_rupee'].apply(convert_price)
df['kms_driven'] = df['kms_driven'].apply(lambda x: int(re.sub(r'[^\d]', '', x)))
df['engine'] = df['engine'].apply(lambda x: int(x.replace(' cc', '')))
df['Seats'] = df['Seats'].apply(lambda x: int(x.replace(' Seats', '')))

# Prepare features and target variable
X = df.drop('car_prices_in_rupee', axis=1)
y = df['car_prices_in_rupee']

# Define categorical features for preprocessing
categorical_features = ['car_name', 'fuel_type', 'transmission', 'ownership']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Define the Linear Regression model and create a pipeline
model = LinearRegression()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

# Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save the trained model pipeline
joblib.dump(pipeline, 'car_price_model.pkl')  # Save model to a file

# Load the model for use in the Tkinter application
pipeline = joblib.load('car_price_model.pkl')  # Load the saved model

# Predict on the test set and calculate metrics
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the accuracy metrics in the command prompt
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Function to predict using the loaded model
def predict():
    # Get user input values
    car_name = car_name_var.get()
    manufacture = int(manufacture_var.get())
    fuel_type = fuel_type_var.get()
    seats = int(seats_var.get())
    kms_driven = int(kms_driven_var.get())
    transmission = transmission_var.get()
    ownership = ownership_var.get()
    engine = int(engine_var.get())

    # Create a DataFrame for prediction input
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
    user_df = pd.DataFrame([user_input], columns=X.columns)
    
    # Predict price using the loaded model and display result
    predicted_price = pipeline.predict(user_df)[0]
    result_label.config(text=f"Predicted Price: ₹{predicted_price:,.2f}")

# Function to display the graph
def show_graph():
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_title("Actual vs. Predicted Car Prices")
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")

    # Display the graph in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Initialize the Tkinter window
window = tk.Tk()
window.title("Car Price Prediction")
window.geometry("600x800")

# Car Model Dropdown
car_name_var = tk.StringVar()
car_name_label = tk.Label(window, text="Select Car Model:")
car_name_label.pack()
car_name_dropdown = ttk.Combobox(window, textvariable=car_name_var, values=list(df['car_name'].unique()))
car_name_dropdown.pack()

# Year of Manufacture
manufacture_var = tk.StringVar()
manufacture_label = tk.Label(window, text="Enter Year of Manufacture:")
manufacture_label.pack()
manufacture_entry = tk.Entry(window, textvariable=manufacture_var)
manufacture_entry.pack()

# Fuel Type Dropdown
fuel_type_var = tk.StringVar()
fuel_type_label = tk.Label(window, text="Select Fuel Type:")
fuel_type_label.pack()
fuel_type_dropdown = ttk.Combobox(window, textvariable=fuel_type_var, values=list(df['fuel_type'].unique()))
fuel_type_dropdown.pack()

# Number of Seats Dropdown
seats_var = tk.StringVar()
seats_label = tk.Label(window, text="Select Number of Seats:")
seats_label.pack()
seats_dropdown = ttk.Combobox(window, textvariable=seats_var, values=[4, 5, 6, 7])
seats_dropdown.pack()

# Kms Driven Entry
kms_driven_var = tk.StringVar()
kms_driven_label = tk.Label(window, text="Enter Kms Driven:")
kms_driven_label.pack()
kms_driven_entry = tk.Entry(window, textvariable=kms_driven_var)
kms_driven_entry.pack()

# Transmission Type Dropdown
transmission_var = tk.StringVar()
transmission_label = tk.Label(window, text="Select Transmission Type:")
transmission_label.pack()
transmission_dropdown = ttk.Combobox(window, textvariable=transmission_var, values=list(df['transmission'].unique()))
transmission_dropdown.pack()

# Ownership Type Dropdown
ownership_var = tk.StringVar()
ownership_label = tk.Label(window, text="Select Ownership Type:")
ownership_label.pack()
ownership_options = [opt for opt in df['ownership'].unique() if opt != "0th owner"]
ownership_dropdown = ttk.Combobox(window, textvariable=ownership_var, values=ownership_options)
ownership_dropdown.pack()

# Engine CC Entry
engine_var = tk.StringVar()
engine_label = tk.Label(window, text="Enter Engine CC:")
engine_label.pack()
engine_entry = tk.Entry(window, textvariable=engine_var)
engine_entry.pack()

# Predict Button
predict_button = tk.Button(window, text="Predict Price", command=predict)
predict_button.pack()

# Button to Show Graph
graph_button = tk.Button(window, text="Show Graph", command=show_graph)
graph_button.pack()

# Label to Display the Result
result_label = tk.Label(window, text="")
result_label.pack()

# Run the Tkinter event loop
window.mainloop()
