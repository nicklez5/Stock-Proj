from urllib import response
from flask import Flask, flash, make_response, render_template, render_template_string, url_for, request, redirect,session
from flask_login import LoginManager, login_required, login_user, logout_user, current_user,UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail,Message
from flask_bcrypt import Bcrypt
from itsdangerous import BadSignature, SignatureExpired, TimedSerializer
from eodhd import APIClient
from itsdangerous import URLSafeTimedSerializer

import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient
from newsapi import NewsApiClient
from datetime import datetime

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
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
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
    end_date = db.Column(db.DateTime,default=datetime.now())



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
    user_name = user.username;
    user_email = user.email;
    user_date = user.date_created;
    user_money = user.money;
    user_stockz = user.stockz;
    if not user:
        error = "User does not exists"
        return render_template("auth/reset_password_error.html",title="Reset Password error",error=error)
    error = None
    if request.method == "POST":
        password1 = request.form.get("password")
        password2 = request.form.get("password2")
        if password1 == password2:
            Person.query.filter_by(username=user_name).delete()
            newperson = Person(id=user.id, username=user_name, password=password2,
                               email=user_email, date_created = user_date, money = user_money,
                               stockz = user_stockz
                               )
            try:
                with app.app_context():
                    db.session.add(newperson)
                    db.session.commit()
                    return render_template(
                        "/auth/reset_password_success.html",title="Reset Password Success"
                    )
            except Exception as error:
                print("An error occured: ", error)
            
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
    

@login_manager.user_loader
def load_user(user_id):
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
    new_person = request.cookies.get('Person1')
    return render_template("/home/home.html", all_headlines=all_headlines,new_person=current_user)
@app.route('/stocks', methods=['POST', 'GET'])
def stock():
    if request.method == 'POST':
        stock_name = request.form.get('keyword2')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        period = request.form.getlist('period')
        order = request.form.getlist('order')
        if "d" in period:
            if "a" in order:
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='d',from_date=from_date,to_date=to_date,order='a')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)
                df = pd.DataFrame.from_dict(data)
                my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                temp_stock = Stock(name=stock_name,start_date=from_date,end_date = to_date)
                
                return render_template('/stocks/stock.html',my_table=my_table)
            elif "d" in order:
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='d',from_date=from_date,to_date=to_date,order='d')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)
                df = pd.DataFrame.from_dict(data)
                my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                
                return render_template('/stocks/stock.html',my_table=my_table)
        elif 'm' in period:
                if "a" in order:
                    resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='m',from_date=from_date,to_date=to_date,order='a')
                    with open("sample.json","w") as outfile:
                        json.dump(resp,outfile)
                    f = open('sample.json',)
                    data = json.load(f)
                    df = pd.DataFrame.from_dict(data)
                    my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                    
                    return render_template('/stocks/stock.html',my_table=my_table)
                elif "d" in order:
                    resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='m',from_date=from_date,to_date=to_date,order='d')
                    with open("sample.json","w") as outfile:
                        json.dump(resp,outfile)
                    f = open('sample.json',)
                    data = json.load(f)
                    df = pd.DataFrame.from_dict(data)
                    my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                    
                    return render_template('/stocks/stock.html',my_table=my_table)
        elif "y" in period:
            if "a" in order:
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='y',from_date=from_date,to_date=to_date,order='a')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)
                df = pd.DataFrame.from_dict(data)
                my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                print(data)
                return render_template('/stocks/stock.html',my_table=my_table)
            elif "d" in order:
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='y',from_date=from_date,to_date=to_date,order='d')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)
                df = pd.DataFrame.from_dict(data)
                my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                print(data)
                return render_template('/stocks/stock.html',my_table=my_table)
            
    elif request.method == 'GET':
        empty_table = []
        return render_template('/stocks/stock.html',my_table=empty_table)
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