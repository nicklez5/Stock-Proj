
from sqlalchemy import URL
from templates.auth.reset_password_email_content import (
    reset_password_email_html_content)
import json
from bitcoin_value import currency
import datetime
from datetime import date
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
import io
from flask import Flask, flash, make_response, render_template, render_template_string, url_for, request, redirect, session
from flask_login import LoginManager, login_required, login_user, logout_user, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from itsdangerous import BadSignature, SignatureExpired, TimedSerializer
from eodhd import APIClient
from itsdangerous import URLSafeTimedSerializer
from io import BytesIO
import base64
# Machine learning imports
import numpy as np
from dotenv import load_dotenv
import os
import tensorflow as tf
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
load_dotenv()

plt.style.use('fivethirtyeight')


# Machine learning devices


load_dotenv()

lstm_image = ""
regular_image = ""
preferred_stock_predictions = []
app = Flask(__name__)



app.secret_key = os.getenv('SECRET_KEY_FLASK')
newsapi = NewsApiClient(os.getenv('news_api'))
api = APIClient(os.getenv("api_client"))
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


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Person, user_id)


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

    def __repr__(self):
        return '<User %r>' % (self.username)

    def set_password(self, password2):
        self._password = password2

    def set_money(self, money2):
        self.money = money2

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
    top_headlines = newsapi.get_top_headlines(country="us", language="en")
    total_results = top_headlines['totalResults']
    if total_results > 100:
        total_results = 100
    all_headlines = newsapi.get_top_headlines(
        country="us", language="en", page_size=total_results)['articles']
    # username = session['username']
    new_user = current_user
    return render_template("/home/home.html", all_headlines=all_headlines, current_user=new_user)


def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


def lstm(stock_name, from_date, to_date):

    bytes_me = io.BytesIO()
    df = yf.download(tickers=stock_name, start=from_date, end=to_date)
    # print(df)
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD($)', fontsize=18)
    plt.savefig(fname="final.png")

    # Create a new dataframe with only the Close column
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create a new array containg scaled values from index 1543 to 2003
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    #print(rmse)
    xyz = predictions[len(predictions) - 1] + rmse
    #print("Final Predictions: " + str(xyz))
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig(bytes_me, format="png")
    bytes_me.seek(0)
    final_img = base64.b64encode(bytes_me.read()).decode()

    return (xyz[0], final_img)


@app.route('/info2', methods=['GET', 'POST'])
def get_all_stocks():
    if request.method == 'GET':
        all_the_stocks = current_user.stockz
        current_amount_money = round(current_user.money, 2)
        return render_template('/home/inventory.html', round=round, yf=yf, all_the_stocks=all_the_stocks, current_amount_money=current_amount_money)
    elif request.method == 'POST':
        id = request.form.get('val')
        stock_sold = Stock.query.filter(Stock.id == id).first()
        ticker_yahoo = yf.Ticker(stock_sold.name)
        data = ticker_yahoo.history()
        last_quote = data['Close'].iloc[-1]
        current_user.money = current_user.money + (last_quote * (stock_sold.amount))
        current_user.stockz.remove(stock_sold)
        db.session.commit()
        current_amount_money = round(current_user.money, 2)
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

            print(current_user.money)
            the_amount_wanted = float(stock_amount) * last_quote
            if (the_amount_wanted > current_user.money):
                error2 = "Unable to get that amount, lack of money"
                return render_template("/stocks/stock.html",lstm_image=lstm_image, regular_image=regular_image, error=error2, preferred_stock_name=preferred_stock_name, preferred_stock_predictions=preferred_stock_predictions, last_quote=last_quote)
            else:
                total_amount = last_quote * float(stock_amount)
                current_user.money = current_user.money - total_amount

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
            print("Max amount able to buy of stock with bitcoin: " +
                  str(max_amount_btc))
        
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
            true_value = round(true_value, 2)
            plt.clf()

            # Get the data from stock api
            
            resp = api.get_eod_historical_stock_market_data(
                symbol=stock_name, period='d', from_date=from_date, to_date=to_date)
            with open("sample.json", "w") as outfile:
                json.dump(resp, outfile)

            # Load the data into f
            f = open('sample.json')
            data = json.load(f)
            df = pd.DataFrame(data)

            df = df[['date', 'close']]
            # print(df)
            df['date'] = df['date'].apply(str_to_datetime)
            df.index = df.pop('date')
            # plt.subplot(1,2,2)
            plt.plot(df.index, df['close'])
            # Plot
            # Save the plotting image
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
            true_value = round(true_value, 2)
            # Get data from api
            plt.clf()
            resp = api.get_eod_historical_stock_market_data(
                symbol=stock_name, period='w', from_date=from_date, to_date=to_date)
            with open("sample.json", "w") as outfile:
                json.dump(resp, outfile)

            f = open('sample.json')
            data = json.load(f)

            # Put that data into a dictionary aka dataframe
            # df = pd.DataFrame.from_dict(data)

            df = pd.DataFrame(data)

            df = df[['date', 'close']]

            df['date'] = df['date'].apply(str_to_datetime)
            df.index = df.pop('date')
            # Plot

            plt.plot(df.index, df['close'])

            # Save the plot and insert it into html
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
            true_value = round(true_value, 2)
            plt.clf()
            # Get data from api
            resp = api.get_eod_historical_stock_market_data(
                symbol=stock_name, period='m', from_date=from_date, to_date=to_date)
            with open("sample.json", "w") as outfile:
                json.dump(resp, outfile)

            # Load the data into f
            f = open('sample.json')

            # Load the data from file
            data = json.load(f)

            # Data into a dataframe
            df = pd.DataFrame(data)

            df = df[['date', 'close']]

            df['date'] = df['date'].apply(str_to_datetime)

            df.index = df.pop('date')

            # Plot
            plt.plot(df.index, df['close'])
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
            # Get data from api
            resp = api.get_eod_historical_stock_market_data(
                symbol=stock_name, from_date=from_date, to_date=to_date)
            with open("sample.json", "w") as outfile:
                json.dump(resp, outfile)

            # Load the data into f
            f = open('sample.json')

            # Load the data from file
            data = json.load(f)

            # Data into a dataframe
            df = pd.DataFrame(data)

            df = df[['date', 'close']]

            df['date'] = df['date'].apply(str_to_datetime)

            df.index = df.pop('date')

            # Plot
            plt.plot(df.index, df['close'])
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
        username = current_user.username
        usr_wallet_amount = current_user.money
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
                    top_headlines = newsapi.get_top_headlines(
                        country="us", language="en")
                    total_results = top_headlines['totalResults']
                    if total_results > 100:
                        total_results = 100
                    all_headlines = newsapi.get_top_headlines(
                        country="us", language="en", page_size=total_results)['articles']

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
    app.run(debug=True)
