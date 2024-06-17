import io
from urllib import response
from flask import Flask, flash, make_response, render_template, render_template_string, url_for, request, redirect,session
from flask_login import LoginManager, login_required, login_user, logout_user, current_user,UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail,Message
from flask_bcrypt import Bcrypt
from itsdangerous import BadSignature, SignatureExpired, TimedSerializer
from eodhd import APIClient
from itsdangerous import URLSafeTimedSerializer
from IPython.display import display
from io import BytesIO 
import base64
import pickle
import pandas as pd
import requests
import os

from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient
from newsapi import NewsApiClient

import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




import json
from templates.auth.reset_password_email_content import (reset_password_email_html_content)
from sqlalchemy import URL
load_dotenv()


app = Flask(__name__)
bcrypt = Bcrypt(app)
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
    db.create_all();
    db.session.commit();

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

person_stocks = db.Table(
    "person_stocks",
    db.Column("person_id",db.Integer, db.ForeignKey("person.id")),
    db.Column("stock_id", db.Integer, db.ForeignKey("stock.id")),
)


class Person(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100),nullable=False)
    email = db.Column(db.String(100),nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.datetime.now())
    money = db.Column(db.Integer,default=4000)
    stockz = db.relationship('Stock',lazy='subquery', secondary=person_stocks, backref='persons')
    def generate_reset_password_token(self):
        serializer = URLSafeTimedSerializer(os.getenv('SECRET_KEY_FLASK'))
        
        return serializer.dumps(self.email)
    def __repr__(self):
        return '<Person %r>' % self.id
    
    @classmethod
    def is_user_name_taken(cls,username):
        return db.session.query(db.exists().where(Person.username == username)).scalar()
    
    @classmethod
    def is_email_taken(cls,email):
        return db.session.query(db.exists().where(Person.email==email)).scalar()

    
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
        #user = db.session.get(Person,user_id)
        user = Person.query.filter_by(id=user_id).first()
        #print(user)
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
    name = db.Column(db.String(100),nullable=False)

    start_date = db.Column(db.DateTime,default='2023-01-01')
    end_date = db.Column(db.DateTime,default=datetime.datetime.now())



    def __repr__(self):
        return f'<Stock "{self.name}">'

def send_reset_password_email(user):
    reset_password_url = url_for(
        "reset_password",
        token=user.generate_reset_password_token(),
        user_id=user.id,
        _external = True,
    )
    email_body = render_template_string(
        reset_password_email_html_content,reset_password_url=reset_password_url
    )
    with mail.connect() as conn:
        message = Message(
        subject="Reset your password",
        html=email_body,
        recipients=[user.email],
        sender=os.getenv("MAIL_USERNAME"))

        conn.send(message)
    
    

@app.route("/reset_password/<token>/<int:user_id>", methods=["GET","POST"])
def reset_password(token,user_id):
    error = None
    if current_user.is_authenticated:
        return redirect("/news")
    user = Person.validate_reset_password_token(token,user_id)
    if not user:
        error = "User does not exists"
        return render_template("auth/reset_password_error.html",title="Reset Password error",error=error)
    if request.method == "POST":
        password1 = request.form.get("password")
        password2 = request.form.get("password2")
        if password1 == password2:
            
            try:
                with app.app_context():
                    email = user.email;
                    user2 = Person.query.filter_by(email=email).first()
                    user2.password = password2
                    session['username'] = user2.username
                    db.session.commit()
                    return render_template(
                        "/auth/reset_password_success.html",title="Reset Password Success", current_user=user2
                    )
            except Exception as err: 
                print(f"Unexpected {err=}, {type(err)=}")
                raise
                
            
        else:
            error = "None matching passwords"
            return render_template(
                "/auth/reset_password_error.html", error=error,title="Reset Password Failed"
            )
    return render_template("/auth/ResetPasswordFinal.html",error=error,user=user)
@app.route("/reset_password", methods=["GET","POST"])
def reset_password_request():
    error = None
    if request.method == "POST":
        email = request.form.get("email")
        user = Person.query.filter_by(email=email).first()
        if user:
            send_reset_password_email(user)
            error="Instructors to reset your password were sent to your email address, if it exists in our system"
        else:
            error = "Email does not exists in database"
        


    return render_template("/auth/ResetPassword.html",error=error)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)
  
  target_date = first_date
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    print(len(df_subset))
    print(n)
    if len(df_subset) <= n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Person,user_id)
    return Person.query.get(user_id)

@app.route("/login", methods=['GET','POST'])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = Person.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                login_user(user,remember=True)
                return redirect('/')
            else:
                error = "Invalid Credentials. Please try again."
        else:
            return "Email does not exist."
    return render_template("/auth/login.html",user=current_user,error=error)

def get_sources_and_domains():
    all_sources = newsapi.get_sources()['sources']
    sources = []
    domains = []
    for e in all_sources:
        id = e['id']
        domain = e['url'].replace("http://", "")
        domain = domain.replace("https://", "")
        domain = domain.replace("www.","")
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
    top_headlines = newsapi.get_top_headlines(country="us",language="en")
    total_results = top_headlines['totalResults']
    if total_results > 100:
        total_results = 100
    all_headlines = newsapi.get_top_headlines(country="us", language="en", page_size=total_results)['articles']
    username = session['username']
    new_user = Person.query.filter_by(username=username).first()
    return render_template("/home/home.html", all_headlines=all_headlines,current_user = new_user)

def str_to_datetime(s):
    split = s.split('-')
    year, month,day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year,month=month,day=day)

@app.route('/stocks', methods=['POST', 'GET'])
def stock():
    current_time = None;
    if request.method == 'POST':
        stock_name = request.form.get('keyword2')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        period = request.form.getlist('period')
        order = request.form.getlist('order')
        if "d" in period:
            if "a" in order:
                bytes_me = io.BytesIO()

                ## Get the data from stock api
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='d',from_date=from_date,to_date=to_date,order='a',fmt=json)
                #resp = api.get_live_stock_prices(date_from=from_date, date_to=to_date, ticker=stock_name)
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)

                #Load the data into f
                f = open('sample.json')
                data = json.load(f)

                #df = pd.DataFrame.from_dict(data)
                df = pd.DataFrame(data)
                csv_file = "output.csv"

                df.to_csv(csv_file,index=False)
                df = pd.read_csv("output.csv")
                
                df = df[['date','close']]
                
                df['date'] = df['date'].apply(str_to_datetime)
                
                df.index = df.pop('date')
                
                plt.plot(df.index, df['close'])
                print(df)
                windowed_df = df_to_windowed_df(df,from_date,to_date,n=3)
                print(windowed_df)
                #Plot 
                
                
                #Save the plotting image
                plt.title("Stock")
                plt.xlabel("Time")
                plt.ylabel("Price")
                
                plt.savefig(bytes_me,format="png")
                bytes_me.seek(0)
                my_base64_pngData = base64.b64encode(bytes_me.read()).decode()
                plt.close()
                
                my_table = df.to_html(classes=['date','close'])
                
                #Same as here
                #windowed_df = df_to_windowed_df(df,from_date, to_date, n=3)
                #result = windowed_df.to_html()
                #print(windowed_df)
                #df = pd.DataFrame.from_dict(data)
                #my_table = df.to_html(classes=['date','open','high','low','close','adjusted_close','volume'])
                temp_stock = Stock(name=stock_name,start_date=from_date,end_date=to_date)


                #This was the first option
                return render_template('/stocks/stock.html',my_base64_pngData=my_base64_pngData,stock_name=stock_name)

                #return render_template('/stocks/stock.html',my_table=my_table)
            elif "d" in order:
                
                bytes_me = io.BytesIO()
                #Get the data from api
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='d',from_date=from_date,to_date=to_date,order='d')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)

                #Put the data to a dictionary
                #df = pd.DataFrame.from_dict(data)
                df = pd.DataFrame(data)
                csv_file = "output.csv"
                df.to_csv(csv_file,index=False)
                df = pd.read_csv("output.csv")
                df = df[['date','close']]
                df['date'] = df['date'].apply(str_to_datetime)
                
                df.index = df.pop('date')

                #Plot
                plt.plot(df.index, df['close'])
                
                #plt.show()
                plt.title("Stock")
                plt.xlabel("Time")
                plt.ylabel("Price")
                plt.savefig(bytes_me,format="png")
                bytes_me.seek(0)
                my_base64_pngData = base64.b64encode(bytes_me.read()).decode()
                plt.close()

                
                

                #dataframe to csv
                #df.to_csv("stock.csv",sep="\t")

                #Unnecessary table for printing it out from CRUD
                my_table = df.to_html(classes=['date','close'])
                #df = pd.read_csv('stock.csv')
                
                #windowed_df = df_to_windowed_df(df,from_date, to_date, n=3)
                #result = windowed_df.to_html()
                #print(windowed_df)
                #
                return render_template('/stocks/stock.html',my_base64_pngData=my_base64_pngData,stock_name=stock_name)
        elif 'w' in period:
                if "a" in order:
                    #Get img
                    bytes_me = io.BytesIO()

                    #Get data from api
                    resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='w',from_date=from_date,to_date=to_date,order='a')
                    with open("sample.json","w") as outfile:
                        json.dump(resp,outfile)
                    f = open('sample.json',)
                    data = json.load(f)

                    #Put that data into a dictionary aka dataframe
                    #df = pd.DataFrame.from_dict(data)

                    df = pd.DataFrame(data)
                    csv_file = "output.csv"
                    df.to_csv(csv_file,index=False)
                    df = pd.read_csv("output.csv")
                    df = df[['date','close']]
                    df['date'] = df['date'].apply(str_to_datetime)
                    df.index = df.pop('date')
                    
                    #Plot
                    plt.plot(df.index, df['close'])
                    
                    #Save the plot and insert it into html
                    plt.title("Stock")
                    plt.xlabel("Time")
                    plt.ylabel("Price")
                    plt.savefig(bytes_me,format="png")
                    bytes_me.seek(0)
                    my_base64_pngData = base64.b64encode(bytes_me.read()).decode()
                    plt.close()
                    

                    #Plot url
                    #plot_url = base64.b64encode(img.getvalue()).decode('utf-8')

                    #Dataframe to csv
                    #df.to_csv("stock.csv",sep="\t")
                    #my_table = df.to_html(classes=['date','close'])

                    #read the csv into dataframe 
                    #df = pd.read_csv('stock.csv')

                    #Convert the dataframe into a windowed dataframe
                    #windowed_df = df_to_windowed_df(df,from_date, to_date, n=3)
                    #result = windowed_df.to_html()
                    #print(windowed_df)
                    #print(result)   
                    return render_template('/stocks/stock.html',my_base64_pngData=my_base64_pngData,stock_name=stock_name)
                    #return render_template('/stocks/stock.html',my_table=my_table)
                elif "d" in order:

                    #Get img
                    #img = BytesIO()
                    bytes_me = io.BytesIO()
                    #Get data from api
                    resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='w',from_date=from_date,to_date=to_date,order='d')
                    with open("sample.json","w") as outfile:
                        json.dump(resp,outfile)

                    #Load the data into f
                    f = open('sample.json',)
                    data = json.load(f)

                    #df = pd.DataFrame.from_dict(data)
                    df = pd.DataFrame(data)
                    csv_file = "output.csv"
                    df.to_csv(csv_file,index=False)
                    df = pd.read_csv(csv_file)
                    df = df[['date','close']]
                    df['date'] = df['date'].apply(str_to_datetime)
                    df.index = df.pop('date')

                    #Plot the points on the dataframe
                    plt.plot(df.index, df['close'])
                    #plt.show()

                    plt.title("Stock")
                    plt.xlabel("Time")
                    plt.ylabel("Price")
                    plt.savefig(bytes_me,format="png")
                    bytes_me.seek(0)
                    my_base64_pngData = base64.b64encode(bytes_me.read()).decode()
                    plt.close()
                    
                    #Plot save the figure in the png format
                    

                    #Plot url
                    #plot_url = base64.b64encode(img.getvalue()).decode('utf8')

                    #Dataframe to csv
                    #df.to_csv("stock.csv",sep="\t")
                    #my_table = df.to_html(classes=['date','close'])

                    #
                    #df = pd.read_csv('stock.csv')

                    #windowed_df = df_to_windowed_df(df,from_date, to_date, n=3)
                    #result = windowed_df.to_html()
                    #print(windowed_df)
                    return render_template('/stocks/stock.html',my_base64_pngData=my_base64_pngData,stock_name=stock_name)

        elif "m" in period:
            if "a" in order:

                #image 
                bytes_me = io.BytesIO()

                #Get data from api
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='m',from_date=from_date,to_date=to_date,order='a')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)

                #Load the data into f
                f = open('sample.json',)

                #Load the data from file 
                data = json.load(f)

                #Data into a dataframe
                df = pd.DataFrame(data)
                csv_file = "output.csv"
                df.to_csv(csv_file,index=False)
                df = pd.read_csv(csv_file)
                df = df[['date','close']]
                df['date'] = df['date'].apply(str_to_datetime)
                df.index = df.pop('date')

                #Plot
                plt.plot(df.index, df['close'])
                plt.title("Stock")
                plt.xlabel("Time")
                plt.ylabel("Price")
                plt.savefig(bytes_me,format="png")
                bytes_me.seek(0)
                my_base64_pngData = base64.b64encode(bytes_me.read()).decode()
                plt.close()

                #Save the plotting image
               

                #put the url into the html
                #plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                #my_table = df.to_html(classes=['date','close'])

                #read stock.csv 
                #df = pd.read_csv('stock.csv')
                #windowed_df = df_to_windowed_df(df,from_date, to_date, n=3)
                #result = windowed_df.to_html()
                #print(windowed_df)

                
                #print(result)
                
                return render_template('/stocks/stock.html',my_base64_pngData=my_base64_pngData,stock_name=stock_name)

                #return render_template('/stocks/stock.html', my_table=my_table)
            elif "d" in order:
                bytes_me = io.BytesIO()

                #Get the data from api
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='m',from_date=from_date,to_date=to_date,order='d')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',"r")
                data = json.load(f)

                print(data)
                #Put that data to a dictionary
                df = pd.DataFrame(data)

                csv_file = "output.csv"
                df.to_csv(csv_file, index=False)
                df = pd.read_csv(csv_file)
                df = df[['date','close']]
                df['date'] = df['date'].apply(str_to_datetime)
                df.index = df.pop('date')
                plt.plot(df.index, df['close'])
                #from_date = '2021-03-25'
                #to_date = '2022-03-23'
                windowed_df = df_to_windowed_df(df,from_date,to_date,n=3)
                print(windowed_df)
                #Plot
                
                plt.title("Stock")
                plt.xlabel("Time")
                plt.ylabel("Price")
                plt.savefig(bytes_me,format="png")
                bytes_me.seek(0)
                my_base64_pngData = base64.b64encode(bytes_me.read()).decode()

                plt.close()

                # df = pd.DataFrame()
                # df = pd.read_csv("MSFT(3).csv")
                
                # df = df[['Date','Close']]
                # df['Date'] = df['Date'].apply(str_to_datetime)
                # df.index = df.pop('Date')
                # from_date = '2022-03-28'
                # to_date = "2023-06-12"
                # windowed_df = df_to_windowed_df(df,from_date,to_date,n=3)
                # print(windowed_df)
                    
                #Plot url
                #plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                
                
                #my_table = df.to_html(classes=['date','close'])
                #df = pd.read_csv('stock.csv')
                #windowed_df = df_to_windowed_df(df,from_date, to_date, n=3)
                #result = windowed_df.to_html()
                #print(windowed_df)
                
                return render_template('/stocks/stock.html',my_base64_pngData=my_base64_pngData,stock_name=stock_name)

            
    else:
        empty_table = []
        username = session["username"]
        print(username)
        #date = datetime.datetime.now()
        
        new_user = Person.query.filter_by(username=username).first()
        return render_template('/stocks/stock.html',current_user=new_user)
    return render_template('/stocks/stock.html')

@app.route('/info', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        username_content = request.form.get('username')
        email_content = request.form.get('email')
        password_content = request.form.get('password')
        password_content_2 = request.form.get('password2')
        if Person.is_user_name_taken(username_content):
            username_validation = False
            return render_template('/auth/info.html',username_validation=username_validation)
        elif Person.is_email_taken(email_content):
            email_validation = False
            return render_template('/auth/info.html',email_validation=email_validation)
        if password_content == password_content_2:
            new_person = Person(username=username_content, 
                          password=password_content, 
                          email=email_content)
            
            try:
                with app.app_context():
                    db.session.add(new_person)
                    db.session.commit()
                    login_user(new_person, remember=True)
                    top_headlines = newsapi.get_top_headlines(country="us",language="en")
                    total_results = top_headlines['totalResults']
                    if total_results > 100:
                        total_results = 100
                    all_headlines = newsapi.get_top_headlines(country="us", language="en", page_size=total_results)['articles']
                    
                    response = make_response(render_template('/home/home.html',all_headlines=all_headlines,new_person=new_person))
                    response.set_cookie("Person1",username_content)
                    return render_template('/home/home.html',all_headlines=all_headlines,new_person=new_person)
            except Exception as error:
                print("An error occured:", error)
                
        else:
            validation_password = False
            return render_template('/auth/info.html',validation_password=validation_password)
    else:
        Person1 = request.cookies.get('Person1')
        return render_template('/auth/info.html',Person1=Person1)
    return render_template('/home/home.html')

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