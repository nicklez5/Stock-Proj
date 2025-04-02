
from sqlalchemy import URL
from templates.auth.reset_password_email_content import (
    reset_password_email_html_content)
import requests
import json
import datetime
from datetime import date,timedelta
from bitcoin_value import currency

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from newsapi import NewsApiClient
from flask import Flask, make_response, render_template, render_template_string, url_for, request, redirect
from flask_login import LoginManager, login_required, login_user, logout_user, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from itsdangerous import BadSignature, SignatureExpired

from itsdangerous import URLSafeTimedSerializer
import base64
import math
# Machine learning imports
import numpy as np
from dotenv import load_dotenv
import os
import math
from numpy import array
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras import layers,models
from yahoofinancials import YahooFinancials
from copy import deepcopy
load_dotenv()

plt.style.use('fivethirtyeight')


# Machine learning devices



lstm_image = ""
regular_image = ""
preferred_stock_predictions = []
app = Flask(__name__)




app.secret_key = os.getenv('SECRET_KEY_FLASK')
newsapi = NewsApiClient(api_key='73250150a7eb41ef98e7c198b1fec41c')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PW")
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config["RESET_PASS_TOKEN_MAX_AGE"] = 100000
db = SQLAlchemy(app)

with app.app_context():
    db.create_all()
    db.session.commit()

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

person_stocks = db.Table(
    "person_stocks",
    db.Column("person_id", db.Integer, db.ForeignKey("person.id")),
    db.Column("stock_id", db.Integer, db.ForeignKey("stock.id")),
)
preferred_stock_name = ""


class Person(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.datetime.now())
    money = db.Column(db.Integer, default=4000)
    stockz = db.relationship('Stock', lazy='subquery',
                             secondary=person_stocks, backref='persons')

    def generate_reset_password_token(self):
        serializer = URLSafeTimedSerializer(os.getenv('SECRET_KEY_FLASK'))

        return serializer.dumps(self.email)

    def __repr__(self):
        return '<Person %r>' % self.id

    @classmethod
    def is_user_name_taken(cls, username):
        return db.session.query(db.exists().where(Person.username == username)).scalar()

    @classmethod
    def is_email_taken(cls, email):
        return db.session.query(db.exists().where(Person.email == email)).scalar()

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.id

    def get_username(self):
        return self.username
    
    def get_money(self):
        return self.money
    
    def set_money(self, value):
        self.money = value

    def set_password(self, password2):
        self._password = password2


    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4
        )

    @staticmethod
    def validate_reset_password_token(token: str, user_id: int):
        # user = db.session.get(Person,user_id)
        user = Person.query.filter_by(id=user_id).first()
        # print(user)
        if user is None:
            return None

        serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY_FLASK"))

        try:
            token_user_email = serializer.loads(
                token
            )

        except (BadSignature, SignatureExpired):
            print("Bad signature")
            return None
        if token_user_email != user.email:
            print("Something happened here")
            return None
        return user
    

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Person, user_id)

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    date_purchased = db.Column(db.DateTime, default=datetime.datetime.now())

    def __repr__(self):
        return f'<Stock "{self.name}">'


# @login_manager.user_loader
# def load_user(user_id):
#     return db.session.get(Person,user_id)
#     return Person.query.get(user_id)


def get_sources_and_domains():
    all_sources = newsapi.get_sources()['sources']
    sources = []
    domains = []
    for e in all_sources:
        id = e['id']
        domain = e['url'].replace("http://", "")
        domain = domain.replace("https://", "")
        domain = domain.replace("www.", "")
        slash = domain.find('/')
        if slash != -1:
            domain = domain[:slash]
        sources.append(id)
        domains.append(domain)
    sources = ", ".join(sources)
    domains = ", ".join(domains)
    return sources, domains


@app.route("/")
def home_no_login():
    return render_template("/home/Login.html")


@app.route("/news", methods=['GET'])
def home():
    if request.method == "GET":
        sources,domains = get_sources_and_domains()
        top_headlines = newsapi.get_top_headlines(sources=sources, language="en")
        
        total_results = top_headlines['totalResults']
        if total_results > 100:
            total_results = 100
        all_headlines = newsapi.get_top_headlines(
             language="en",sources=sources)['articles']
        #print(all_headlines)
        # username = session['username']
        new_user = current_user
        return render_template("/home/home.html", all_headlines=all_headlines, current_user=new_user)


def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n = 3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)
    print(first_date)
    target_date = first_date

    dates = []
    X, Y = [], []
    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return
        
        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day),month=int(month),year=int(year))
        if last_time:
            break
        target_date = next_date

        if target_date == last_date:
            last_time = True
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0,n):
        X[:,i]
        ret_df[f'Target-{n-i}'] = X[:,i]
    ret_df['Target'] = Y
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:,0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1] , 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)

def lstm(stock_name, from_date, to_date):

    bytes_me = io.BytesIO()
    from_date1 = str_to_datetime(from_date)
    from_date2 = (from_date1 - datetime.timedelta(days=1825)).date()
    api_key = os.environ.get("12_day_api_key")
    response = requests.get(f"https://api.twelvedata.com/time_series?apikey={api_key}&interval=1day&symbol={stock_name}&end_date={to_date}")
    data = response.json()
    filename = "stock.json"
    with open(filename,"w") as file:
        json.dump(data,file,indent=4)
    data = json.load(open("stock.json"))
    df = pd.DataFrame(data["values"])
    df = df.iloc[::-1]
    print(df)
    length_data = len(df)
    split_ratio = 0.7
    length_train = round(length_data * split_ratio)
    length_validation = length_data - length_train
    print("Data length :", length_data)
    print("Train data length :",length_train)
    print("Validation data length :" , length_validation)
    train_data = df[:length_train].iloc[:, :2]
    train_data['datetime'] = pd.to_datetime(train_data['datetime'])
    print(train_data)
    validation_data = df[length_train:].iloc[:,:2]
    validation_data["datetime"] = pd.to_datetime(validation_data['datetime'])
    print(validation_data)
    dataset_train = train_data.open.values
    print(dataset_train.shape)
    dataset_train = np.reshape(dataset_train,(-1,1))
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train_scaled = scaler.fit_transform(dataset_train)
    print(dataset_train_scaled.shape)
    plt.subplots(figsize=(15,6))
    plt.plot(dataset_train_scaled)
    plt.xlabel("Days as 1st, 2nd, 3rd..")
    plt.ylabel("Open Price")
    plt.savefig("temp")
    X_train = []
    y_train = []
    time_step = 50
    for i in range(time_step, length_train):
        X_train.append(dataset_train_scaled[i-time_step:i,0])
        y_train.append(dataset_train_scaled[i,0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    print("Shape of X_train before reshape :",X_train.shape)
    print("Shape of y_train before reshape :", y_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    print("Shape of X_train after reshape :",X_train.shape)
    print("Shape of y_train after reshape :",y_train.shape)
    print(X_train[0])
    print(y_train[0])
    regressor = keras.models.Sequential()
    regressor.add(keras.layers.SimpleRNN(units = 50,
                                        activation = 'tanh',
                                        return_sequences = True,
                                        input_shape = (X_train.shape[1],1))
                                        )
    regressor.add(keras.layers.Dropout(0.2))
    regressor.add(keras.layers.SimpleRNN(units = 50,
                                        activation='tanh',
                                        return_sequences = True)
                                        )
    regressor.add(keras.layers.Dropout(0.2))
    regressor.add(keras.layers.SimpleRNN(units = 50,
                                        activation='tanh',
                                        return_sequences = True)
                                        )
    regressor.add(keras.layers.Dropout(0.2))
    regressor.add(keras.layers.SimpleRNN(units = 50))
    regressor.add(keras.layers.Dropout(0.2))
    regressor.add(keras.layers.Dense(units = 1))
    regressor.compile(
        optimizer = 'adam',
        loss="mean_squared_error",
        metrics = ["accuracy"]
    )
    history = regressor.fit(X_train,y_train,epochs =50, batch_size = 32)
    print(history.history['loss'])
    plt.clf()
    plt.figure(figsize=(10,7))
    plt.plot(history.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Simple RNN model, Loss vs Epoch")
    plt.savefig("simplernn")
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.plot(history.history["accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracies")
    plt.title("Simple RNN model, Accuracy vs Epoch")
    plt.savefig("accuracy")

    y_pred = regressor.predict(X_train)
    y_pred = scaler.inverse_transform(y_pred)
    print(y_pred.shape)
    y_train = scaler.inverse_transform(y_train)
    print(y_train.shape)
    plt.figure(figsize=(30,10))
    plt.plot(y_pred, color="b",label="y_pred")
    plt.plot(y_train, color="g",label="y_train")
    plt.xlabel("Days")
    plt.ylabel("Open price")
    plt.title("Simple RNN model, Predictions with input X_train vs y_train")
    plt.legend()
    plt.savefig("Simple RNN model with predictions")
    plt.clf()
    dataset_validation = validation_data.open.values
    dataset_validation = np.reshape(dataset_validation, (-1,1))
    scaled_dataset_validation = scaler.fit_transform(dataset_validation)
    print("Shape of scaled validation dataset :", scaled_dataset_validation.shape)
    X_test = []
    y_test = []
    for i in range(time_step, length_validation):
        X_test.append(scaled_dataset_validation[i-time_step:i,0])
        y_test.append(scaled_dataset_validation[i,0])

    X_test , y_test = np.array(X_test), np.array(y_test)
    print("Shape of X_test before reshape :",X_test.shape)
    print("Shape of y_test before reshape :",y_test.shape)
    X_test = np.reshape(X_test , (X_test.shape[0],X_test.shape[1],1))
    y_test = np.reshape(y_test, (-1,1))
    print("Shape of X_test after reshape :",X_test.shape)
    print("Shape of y_test after reshape :",y_test.shape)
    y_pred_of_test = regressor.predict(X_test)
    y_pred_of_test = scaler.inverse_transform(y_pred_of_test)
    print("Shape of y_pred_of_test :",y_pred_of_test.shape)
    plt.figure(figsize=(30,10))
    plt.plot(y_pred_of_test,label="y_pred_of_test" , c = "orange")
    plt.plot(scaler.inverse_transform(y_test),label="y_test",c="g")
    plt.xlabel("Days")
    plt.ylabel("Open price")

    plt.title("Simple RNN model, Prediction with input X_test vs y_test")
    plt.legend()
    plt.savefig(bytes_me,format="png")
    plt.clf()
    plt.subplots(figsize =(30,12))
    plt.plot(train_data.datetime, train_data.open, label = "train_data", color = "b")
    plt.plot(validation_data.datetime, validation_data.open, label = "validation_data", color = "g")
    plt.plot(train_data.datetime.iloc[time_step:], y_pred, label = "y_pred", color = "r")
    plt.plot(validation_data.datetime.iloc[time_step:], y_pred_of_test, label = "y_pred_of_test", color = "orange")
    plt.xlabel("Days")
    plt.ylabel("Open price")
    plt.title("Simple RNN model, Train-Validation-Prediction")
    plt.legend()
    plt.savefig("Train_validation_Prediction")
    y_train = scaler.fit_transform(y_train)
    model_lstm = keras.models.Sequential()
    model_lstm.add(
        keras.layers.LSTM(64, return_sequences=True, input_shape = (X_train.shape[1],1))
    )
    model_lstm.add(
        keras.layers.LSTM(64, return_sequences = False)
    )
    model_lstm.add(keras.layers.Dense(32))
    model_lstm.add(keras.layers.Dense(1))
    model_lstm.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])
    history2 = model_lstm.fit(X_train,y_train,epochs=10,batch_size=10)
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.plot(history2.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("LSTM model, Accuracy vs Epoch")
    plt.savefig("LSTM model")
    plt.clf()
    plt.subplots(figsize =(30,12))
    plt.plot(scaler.inverse_transform(model_lstm.predict(X_test)), label = "y_pred_of_test", c = "orange" )
    plt.plot(scaler.inverse_transform(y_test), label = "y_test", color = "g")
    plt.xlabel("Days")
    plt.ylabel("Open price")
    plt.title("LSTM model, Predictions with input X_test vs y_test")
    plt.legend()
    plt.savefig("LSTM model, Predictions with input X_test vs y_test")
    print(df.iloc[-1])
    X_input = df.iloc[-time_step:].open.values               # getting last 50 rows and converting to array
    X_input = scaler.fit_transform(X_input.reshape(-1,1))      # converting to 2D array and scaling
    X_input = np.reshape(X_input, (1,50,1))                    # reshaping : converting to 3D array
    print("Shape of X_input :", X_input.shape)
    print(X_input)
    simple_RNN_prediction = scaler.inverse_transform(regressor.predict(X_input))
    LSTM_prediction = scaler.inverse_transform(model_lstm.predict(X_input))
    print("Simple RNN, Open price prediction for 3/18/2017      :", simple_RNN_prediction[0,0])
    print("LSTM prediction, Open price prediction for 3/18/2017 :", LSTM_prediction[0,0])

    


    bytes_me.seek(0)

    final_img = base64.b64encode(bytes_me.read()).decode()
    # plt.savefig(bytes_me, format="png")
    # bytes_me.seek(0)
    # final_img = base64.b64encode(bytes_me.read()).decode()

    return (LSTM_prediction[0,0], final_img)


@app.route('/info2', methods=['GET', 'POST'])
@login_required
def get_all_stocks():
    if request.method == 'GET':
        all_the_stocks = current_user.stockz
        current_amount_money = round(current_user.get_money(), 2)
        return render_template('/home/inventory.html', round=round, yf=yf, all_the_stocks=all_the_stocks, current_amount_money=current_amount_money)
    elif request.method == 'POST':
        id = request.form.get('val')
        stock_sold = Stock.query.filter(Stock.id == id).first()
        ticker_yahoo = yf.Ticker(stock_sold.name)
        data = ticker_yahoo.history()
        last_quote = data['Close'].iloc[-1]
        current_user.money = current_user.get_money() + (last_quote * (stock_sold.amount))
        current_user.stockz.remove(stock_sold)
        db.session.commit()
        current_amount_money = round(current_user.get_money(),2)
        all_the_stocks = current_user.stockz
        return render_template('/home/inventory.html', round=round, yf=yf, all_the_stocks=all_the_stocks, current_amount_money=current_amount_money)


@app.route('/buy', methods=['POST'])
def buy_me():
    error2 = ""
    last_quote = ""
    if request.method == 'POST':
        selected = request.form.get('currency')
        stock_amount = request.form.get('stock_amt')
        
        stock_name = preferred_stock_name
        print("Stock Name:" + stock_name)
        #print(selected)
        ticker_yahoo = yf.Ticker(preferred_stock_name)
        data = ticker_yahoo.history()
        last_quote = data['Close'].iloc[-1].tolist()

        last_quote = round(last_quote, 2)
        if selected == "USD":

            print(current_user.get_money())
            the_amount_wanted = float(stock_amount) * last_quote
            if (the_amount_wanted > current_user.get_money()):
                error2 = "Unable to get that amount, lack of money"
                return render_template("/stocks/stock.html",lstm_image=lstm_image, regular_image=regular_image, error=error2, preferred_stock_name=preferred_stock_name, preferred_stock_predictions=preferred_stock_predictions, last_quote=last_quote)
            else:
                total_amount = last_quote * float(stock_amount)
                current_user.money = current_user.get_money() - total_amount

                new_stock = Stock(name=preferred_stock_name,
                                  amount=stock_amount, price=last_quote)
                current_user.stockz.append(new_stock)
                db.session.commit()
                error2 = f'You have successfully purchased {stock_amount} of {
                    preferred_stock_name} with {total_amount} USD'
                return error2

                # find the stock name price
                # get the stock name price * amount
                # turn that into bitcoin
                # used ur max_amount_Btc to buy it
                # add it to your person object stocks.
            # print("Max amount able to buy of stock with bitcoin: " +
            #       str(max_amount_btc))
        
    return render_template("/stocks/stock.html",lstm_image=lstm_image, regular_image=regular_image, error=error2, preferred_stock_name=preferred_stock_name, preferred_stock_predictions=preferred_stock_predictions, last_quote=last_quote)


@app.route('/stocks', methods=['POST', 'GET'])
def stock():
    current_date = datetime.date.today()
    if request.method == 'POST':
        plt.clf()
        stock_name = request.form.get('keyword2')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        period = request.form.getlist('period')

        global preferred_stock_name
        preferred_stock_name = stock_name

        ticker_yahoo = yf.Ticker(preferred_stock_name)
        data = ticker_yahoo.history()

        # data = ticker_yahoo.history()

        stock_price = data['Close'].iloc[-1]
        stock_price = round(stock_price, 2)
        usr_wallet_amount = current_user.money
        usr_wallet_amount = round(usr_wallet_amount, 2)
        
        if "d" in period:

            bytes_me = io.BytesIO()

            true_value, final_img = lstm(stock_name, from_date, to_date)
            #true_value = round(true_value, 2)
            plt.clf()

            # Get the data from stock api
            ticker = stock_name
            api_key = os.environ.get("12_day_api_key")
            print(api_key)
            url = f"https://api.twelvedata.com/time_series?apikey={api_key}&interval=1day&symbol={stock_name}&start_date={from_date}&end_date={to_date}"
            response = requests.get(url)
            data = response.json()
            with open("sample.json","w") as outfile:
                json.dump(data,outfile,indent = 4)

            # Load the data into f
            data = json.load(open('sample.json'))
            df = pd.DataFrame(data["values"])
            df = df[::-1]
            df2 = df[['datetime','close']]


            df2.index = df2.pop('datetime')

            # Plot
            plt.plot(df2.index, df2['close'])
            plt.title("Stock")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.savefig(bytes_me, format="png")
            bytes_me.seek(0)

            my_base_64_pngData = base64.b64encode(bytes_me.read()).decode()
            global regular_image
            regular_image = my_base_64_pngData

            # This was the first option
            return render_template('/stocks/stock.html',round=round,usr_wallet_amount=usr_wallet_amount, my_base_64_pngData=my_base_64_pngData, final_img=final_img, stock_name=stock_name, current_date=current_date, true_value=true_value, stock_price=stock_price)

        elif 'w' in period:

            # Get img
            bytes_me = io.BytesIO()

            true_value, final_img = lstm(stock_name, from_date, to_date)
            # Get data from api
            plt.clf()
            api_key = os.environ.get("12_day_api_key")
            url = f"https://api.twelvedata.com/time_series?apikey={api_key}&interval=1week&symbol={stock_name}&start_date={from_date}&end_date={to_date}"
            response = requests.get(url)
            data = response.json()
            with open("sample.json","w") as outfile:
                json.dump(data,outfile,indent=4)


            data = json.load(open('sample.json'))
            df = pd.DataFrame(data["values"])
            df = df[::-1]
            df2 = df[['datetime','close']]


            df2.index = df2.pop('datetime')

            # Plot
            plt.plot(df2.index, df2['close'])
            plt.title("Stock")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.savefig(bytes_me, format="png")
            bytes_me.seek(0)

            my_base_64_pngData = base64.b64encode(bytes_me.read()).decode()
            regular_image = my_base_64_pngData

            return render_template('/stocks/stock.html',round=round,usr_wallet_amount=usr_wallet_amount, my_base_64_pngData=my_base_64_pngData, final_img=final_img, stock_name=stock_name, current_date=current_date, true_value=true_value, stock_price=stock_price)

        elif "m" in period:

            bytes_me = io.BytesIO()
            # image

            true_value, final_img = lstm(stock_name, from_date, to_date)
            # true_value = [np.round(x) for x in true_value]
            plt.clf()
            ticker = stock_name
            api_key = os.environ.get("12_day_api_key")
            print(api_key)
            url = f"https://api.twelvedata.com/time_series?apikey={api_key}&interval=1month&symbol={stock_name}&start_date={from_date}&end_date={to_date}"
            print(url)
            response = requests.get(url)
            data = response.json()
            # Get data from api
            with open("sample.json", "w") as outfile:
                json.dump(data, outfile,indent=4)

            # Load the data into f
            data = json.load(open('sample.json'))
            df = pd.DataFrame(data["values"])
            df = df[::-1]
            df2 = df[['datetime','close']]


            df2.index = df2.pop('datetime')

            # Plot
            plt.plot(df2.index, df2['close'])
            plt.title("Stock")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.savefig(bytes_me, format="png")
            bytes_me.seek(0)

            my_base_64_pngData = base64.b64encode(bytes_me.read()).decode()
            regular_image = my_base_64_pngData

            return render_template('/stocks/stock.html',round=round,usr_wallet_amount=usr_wallet_amount, my_base_64_pngData=my_base_64_pngData, final_img=final_img, stock_name=stock_name, current_date=current_date, true_value=true_value, stock_price=stock_price)
        else:
            bytes_me = io.BytesIO()
            true_value, final_img = lstm(stock_name, from_date, to_date)
            true_value = round(true_value, 2)
            plt.clf()
            ticker = stock_name
            resp321 = YahooFinancials(ticker)
            resp = resp321.get_historical_price_data(start_date=from_date,end_date=to_date,time_interval="day")

            with open("sample.json", "w") as outfile:
                json.dump(resp, outfile,indent=4)

            # Load the data into f
            f = open('sample.json')

            # Load the data from file
            data = json.load(f)

            # Data into a dataframe
            df = pd.DataFrame(data)
            x = df[ticker]['prices']
            df2 = pd.DataFrame(x)
            df2 = df2[['formatted_date', 'close']]

            df2.index = df2.pop('formatted_date')

            # Plot
            plt.plot(df2.index, df2['close'])
            plt.title("Stock")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.savefig(bytes_me, format="png")
            bytes_me.seek(0)

            my_base_64_pngData = base64.b64encode(bytes_me.read()).decode()
            regular_image = my_base_64_pngData

            return render_template('/stocks/stock.html',round=round,usr_wallet_amount=usr_wallet_amount, my_base_64_pngData=my_base_64_pngData, final_img=final_img, stock_name=stock_name, current_date=current_date, true_value=true_value, stock_price=stock_price)
    else:
        empty_table = []
        username = current_user.get_username()
        usr_wallet_amount = current_user.get_money()
        new_user = Person.query.filter_by(username=username).first()
        return render_template('/stocks/stock.html', usr_wallet_amount=usr_wallet_amount, current_user=new_user,round=round)
    return render_template('/stocks/stock.html')

def wallet_xmr_btc_eth():
    usr_wallet_amount = current_user.money
    usr_wallet_amount = round(usr_wallet_amount, 2)
    bitcoin_max_amount = currency("USD")
    bth_wallet = usr_wallet_amount / bitcoin_max_amount
    bth_wallet = round(bth_wallet,9)
    # date = datetime.datetime.now()
    today = date.today()
    print("Was i here")
    
    ticker_yahoo = yf.Ticker("ETH-USD")
    data = ticker_yahoo.history()
    eth_price = data['Close'].iloc[-1]
    eth_price = round(eth_price,2)
    eth_wallet = usr_wallet_amount / eth_price
    eth_wallet = round(eth_wallet,9)

    ticker_yahoo = yf.Ticker("XMR-USD")
    data = ticker_yahoo.history()
    xmr_price = data['Close'].iloc[-1]
    xmr_price = round(xmr_price,2)
    xmr_wallet = usr_wallet_amount / xmr_price
    xmr_wallet = round(xmr_wallet,9)
    #print(xmr_price)
    return (bth_wallet,eth_wallet,xmr_wallet)
    
@app.route('/info', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        username_content = request.form.get('username')
        email_content = request.form.get('email')
        password_content = request.form.get('password')
        password_content_2 = request.form.get('password2')
        if Person.is_user_name_taken(username_content):
            username_validation = False
            return render_template('/auth/info.html', username_validation=username_validation)
        elif Person.is_email_taken(email_content):
            email_validation = False
            return render_template('/auth/info.html', email_validation=email_validation)
        if password_content == password_content_2:
            new_person = Person(username=username_content,
                                password=password_content,
                                email=email_content)

            try:
                with app.app_context():
                    db.session.add(new_person)
                    db.session.commit()
                    login_user(new_person, remember=True)
                    sources, domains = get_sources_and_domains()
                    top_headlines = newsapi.get_top_headlines(
                        sources=sources, language="en")
                    total_results = top_headlines['totalResults']
                    if total_results > 100:
                        total_results = 100
                    all_headlines = newsapi.get_top_headlines(
                        sources=sources, language="en", page_size=total_results)['articles']

                    response = make_response(render_template(
                        '/home/home.html', all_headlines=all_headlines, new_person=new_person))
                    response.set_cookie("Person1", username_content)
                    return render_template('/home/home.html', all_headlines=all_headlines, new_person=new_person)
            except Exception as error:
                print("An error occured:", error)

        else:
            validation_password = False
            return render_template('/auth/info.html', validation_password=validation_password)
    else:
        Person1 = request.cookies.get('Person1')
        return render_template('/auth/info.html', Person1=Person1)
    return render_template('/home/home.html')


@app.route("/reset_password", methods=["GET", "POST"])
def reset_password_request():
    error = None
    if request.method == "POST":
        email = request.form.get("email")
        user = Person.query.filter_by(email=email).first()
        if user:
            send_reset_password_email(user)
            error = "Instructors to reset your password were sent to your email address, if it exists in our system"
        else:
            error = "Email does not exists in database"
    return render_template("/auth/ResetPassword.html", error=error)


@app.route("/reset_password/<token>/<int:user_id>", methods=["GET", "POST"])
def reset_password(token, user_id):
    error = None
    if current_user.is_authenticated:
        return redirect("/news")
    user = Person.validate_reset_password_token(token, user_id)
    if not user:
        error = "User does not exists"
        return render_template("auth/reset_password_error.html", title="Reset Password error", error=error)
    if request.method == "POST":
        password1 = request.form.get("password")
        password2 = request.form.get("password2")
        if password1 == password2:

            try:
                with app.app_context():
                    email = user.email
                    user2 = Person.query.filter_by(email=email).first()
                    user2.set_password(password2)

                    db.session.commit()
                    return render_template(
                        "/auth/reset_password_success.html", title="Reset Password Success", current_user=user2
                    )
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

        else:
            error = "None matching passwords"
            return render_template(
                "/auth/reset_password_error.html", error=error, title="Reset Password Failed"
            )
    return render_template("/auth/ResetPasswordFinal.html", error=error, user=user)


@app.route("/login", methods=['GET', 'POST'])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = Person.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                login_user(user, remember=True)
                user.is_authenticated()
                return redirect('/')
            else:
                error = "Invalid Credentials. Please try again."
        else:
            return "Email does not exist."
    return render_template("/auth/login.html", user=current_user, error=error)


def send_reset_password_email(user):
    reset_password_url = url_for(
        "reset_password",
        token=user.generate_reset_password_token(),
        user_id=user.id,
        _external=True,
    )
    email_body = render_template_string(
        reset_password_email_html_content, reset_password_url=reset_password_url
    )
    with mail.connect() as conn:
        message = Message(
            subject="Reset your password",
            html=email_body,
            recipients=[user.email],
            sender=os.getenv("MAIL_USERNAME"))

        conn.send(message)


@app.route('/delete/<int:id>')
def delete(id):
    person_to_delete = Person.query.get_or_404(id)
    try:
        db.session.delete(person_to_delete)
        db.session.commit()
        return redirect('/info')
    except:
        return 'There was a problem deleting that task'


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port='5000')
