from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector
import pandas as pd 
import numpy as np
from flask import render_template, request
import joblib
from keras.models import load_model 
import joblib
import matplotlib.pyplot as plt 

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='stock'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/') 
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register(): 
    if request.method == "POST": 
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password) 
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,) 
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')




@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    algo_data = []
    algorithm = None
    selected_bank = None

    bank_list = ['SBIN.NS', 'HDFCBANK.NS', 'AXISBANK.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS']

    model_data = {
        "ARIMA": {
            "SBIN.NS": {"MSE": 163.8910, "RMSE": 12.8020, "MAE": 7.1460, "R2": 0.9833},
            "HDFCBANK.NS": {"MSE": 394.2685, "RMSE": 19.8562, "MAE": 13.3131, "R2": 0.9473},
            "AXISBANK.NS": {"MSE": 250.8132, "RMSE": 15.8371, "MAE": 11.3257, "R2": 0.9611},
            "ICICIBANK.NS": {"MSE": 154.0381, "RMSE": 12.4112, "MAE": 8.3373, "R2": 0.9730},
            "INDUSINDBK.NS": {"MSE": 572.0950, "RMSE": 23.9185, "MAE": 16.9462, "R2": 0.9170}
        },
        "LSTM": {
            "SBIN.NS": {"MSE": 388.4666, "RMSE": 19.7096, "MAE": 14.7313, "R2": 0.961075},
            "HDFCBANK.NS": {"MSE": 1309.8256, "RMSE": 36.1915, "MAE": 27.7293, "R2": 0.829607},
            "AXISBANK.NS": {"MSE": 538.1541, "RMSE": 23.1981, "MAE": 17.6353, "R2": 0.911235},
            "ICICIBANK.NS": {"MSE": 579.4595, "RMSE": 24.0720, "MAE": 18.5197, "R2": 0.897609},
            "INDUSINDBK.NS": {"MSE": 1127.0959, "RMSE": 33.5723, "MAE": 25.1528, "R2": 0.79929}
        },
        "GRU": {
            "SBIN.NS": {"MSE": 228.8589, "RMSE": 15.1281, "MAE": 10.8090, "R2": 0.977068},
            "HDFCBANK.NS": {"MSE": 571.9101, "RMSE": 23.9146, "MAE": 16.8484, "R2": 0.925601},
            "AXISBANK.NS": {"MSE": 349.1555, "RMSE": 18.6857, "MAE": 13.8501, "R2": 0.942409},
            "ICICIBANK.NS": {"MSE": 210.3482, "RMSE": 14.5034, "MAE": 10.3858, "R2": 0.962831},
            "INDUSINDBK.NS": {"MSE": 885.3617, "RMSE": 29.7550, "MAE": 23.5326, "R2": 0.842339}
        },
        "S-LSTM": {
            "SBIN.NS": {"MSE": 791.3302, "RMSE": 28.1306, "MAE": 22.0919, "R2": 0.920708},
            "HDFCBANK.NS": {"MSE": 1407.8185, "RMSE": 37.5209, "MAE": 26.8671, "R2": 0.816859},
            "AXISBANK.NS": {"MSE": 650.6375, "RMSE": 25.5076, "MAE": 19.2092, "R2": 0.892681},
            "ICICIBANK.NS": {"MSE": 902.7654, "RMSE": 30.0461, "MAE": 24.8462, "R2": 0.840481},
            "INDUSINDBK.NS": {"MSE": 1677.7112, "RMSE": 40.9599, "MAE": 34.1024, "R2": 0.701240}
        }
    }

    if request.method == 'POST':
        algorithm = request.form['algorithm']
        selected_bank = request.form['bank']

        if algorithm in model_data and selected_bank in model_data[algorithm]:
            metrics = model_data[algorithm][selected_bank]
            algo_data.append({
                "Bank": selected_bank,
                "MSE": round(metrics["MSE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "MAE": round(metrics["MAE"], 4),
                "Accuracy": round(metrics["R2"] * 100, 2) 
            }) 

    return render_template(
        'algorithm.html',
        algo_data=algo_data,
        algorithm=algorithm,
        selected_bank=selected_bank,
        bank_list=bank_list
    )







from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

df=pd.read_csv('backdata.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
# Function to create sequences for GRU
def create_sequences(data, time_steps=60):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)

# Function to predict future stock prices
def predict_future_stock(model, scaler, data, time_steps, future_days):
    last_data = data[-time_steps:]  # Get the last 60 days of data
    last_data = np.reshape(last_data, (1, time_steps, 1))  # Reshape for GRU input

    predicted_prices = []
    for _ in range(future_days):
        predicted_price = model.predict(last_data)
        predicted_prices.append(predicted_price[0][0])

        # Update the data for the next prediction
        predicted_price_reshaped = np.reshape(predicted_price, (1, 1, 1))  # Reshape predicted_price to be 3D
        last_data = np.append(last_data[:, 1:, :], predicted_price_reshaped, axis=1)

    # Inverse transform to the original scale
    predicted_prices_inv = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    return predicted_prices_inv

# Load models and scalers for different banks (make sure these models exist)
def load_models_and_scaler(bank_name):
    model_filename = f"gru_{bank_name}.h5"  # Change to match actual model filenames
    scaler_filename = f"scaler_{bank_name}.pkl"  # Change to match actual scaler filenames

    model = load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    
    return model, scaler

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        # Get data from the form
        bank_name = request.form['bank_name']
        num_days = int(request.form['num_days'])

        # Load model and scaler for selected bank
        model, scaler_X = load_models_and_scaler(bank_name)

        # Load data and preprocess
        data = df[[bank_name]].dropna().values
        data_scaled = scaler_X.fit_transform(data)

        # Predict future stock prices
        predicted_prices = predict_future_stock(model, scaler_X, data_scaled, time_steps=60, future_days=num_days)

        # Create future dates from last date in dataset
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=num_days + 1, freq='D')[1:]  # Skip the start date

        # Pair formatted dates and predicted prices
        predictions_with_dates = list(zip(future_dates.strftime('%Y-%m-%d'), predicted_prices.flatten()))

        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-100:], df[bank_name].iloc[-100:], label="Actual Prices", color="blue")
        plt.plot(future_dates, predicted_prices, label="Predicted Prices", color="red", linestyle="dashed")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"GRU Forecast for {bank_name} - Next {num_days} Days")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot image
        plot_filename = f"static/{bank_name}_forecast_plot.png"
        plt.savefig(plot_filename)

        # Return to template
        return render_template('prediction.html',
                               predictions=predictions_with_dates,
                               plot_filename=plot_filename,
                               num_days=num_days)

    return render_template('prediction.html')




@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug = True)