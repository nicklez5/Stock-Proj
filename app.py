from urllib import response
from flask import Flask, make_response, render_template, render_template_string, url_for, request, redirect,session
from flask_login import LoginManager, login_required, login_user, logout_user, current_user,UserMixin
from flask_sqlalchemy import SQLAlchemy
from eodhd import APIClient
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Email
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient
from newsapi import NewsApiClient
from datetime import datetime
import json

from sqlalchemy import URL
load_dotenv()


app = Flask(__name__)
app.secret_key = 'super secret key'
newsapi = NewsApiClient(api_key='230b4a51a01f4de2ba0329e873fa5fe3')
api = APIClient("6652bd3e397aa5.61249582")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
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

class ResetPasswordRequestForm(FlaskForm):
    email = StringField("Email",validators=[DataRequired(),Email()])
    submit = SubmitField("Request Password Reset")
    
class Person(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100),nullable=False)
    email = db.Column(db.String(100),nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    money = db.Column(db.Integer,default=4000)
    stockz = db.relationship('Stock',lazy='subquery', secondary=person_stocks, backref='persons')
    
    def __repr__(self):
        return '<Person %r>' % self.id
    
    @classmethod
    def is_user_name_taken(cls,username):
        return db.session.query(db.exists().where(Person.username == username)).scalar()
    
    @classmethod
    def is_email_taken(cls,email):
        return db.session.query(db.exists().where(Person.email==email)).scalar()

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4
        )
class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100),nullable=False)

    start_date = db.Column(db.DateTime,default='2023-01-01')
    end_date = db.Column(db.DateTime,default=datetime.now())



    def __repr__(self):
        return f'<Stock "{self.name}">'

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
    return render_template("login.html",user=current_user,error=error)
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
def home2():
    return render_template("home2.html")
@app.route("/news", methods=['GET'])
def home():
    top_headlines = newsapi.get_top_headlines(country="us",language="en")
    total_results = top_headlines['totalResults']
    if total_results > 100:
        total_results = 100
    all_headlines = newsapi.get_top_headlines(country="us", language="en", page_size=total_results)['articles']
    new_person = request.cookies.get('Person1')
    return render_template("home.html", all_headlines=all_headlines,new_person=new_person)
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
                
                return render_template('stock.html',my_table=my_table)
            elif "d" in order:
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='d',from_date=from_date,to_date=to_date,order='d')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)
                df = pd.DataFrame.from_dict(data)
                my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                
                return render_template('stock.html',my_table=my_table)
        elif 'm' in period:
                if "a" in order:
                    resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='m',from_date=from_date,to_date=to_date,order='a')
                    with open("sample.json","w") as outfile:
                        json.dump(resp,outfile)
                    f = open('sample.json',)
                    data = json.load(f)
                    df = pd.DataFrame.from_dict(data)
                    my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                    
                    return render_template('stock.html',my_table=my_table)
                elif "d" in order:
                    resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='m',from_date=from_date,to_date=to_date,order='d')
                    with open("sample.json","w") as outfile:
                        json.dump(resp,outfile)
                    f = open('sample.json',)
                    data = json.load(f)
                    df = pd.DataFrame.from_dict(data)
                    my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                    
                    return render_template('stock.html',my_table=my_table)
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
                return render_template('stock.html',my_table=my_table)
            elif "d" in order:
                resp = api.get_eod_historical_stock_market_data(symbol=stock_name, period='y',from_date=from_date,to_date=to_date,order='d')
                with open("sample.json","w") as outfile:
                    json.dump(resp,outfile)
                f = open('sample.json',)
                data = json.load(f)
                df = pd.DataFrame.from_dict(data)
                my_table = df.to_html(classes=['date','open','high','low','close','adjusted close','volume'])
                print(data)
                return render_template('stock.html',my_table=my_table)
            
    elif request.method == 'GET':
        empty_table = []
        return render_template('stock.html',my_table=empty_table)
    return render_template('stock.html')

@app.route('/info', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        username_content = request.form['username']
        email_content = request.form['email']
        password_content = request.form['password']
        password_content_2 = request.form['password2']
        if Person.is_user_name_taken(username_content):
            username_validation = False
            return render_template('info.html',username_validation=username_validation)
        elif Person.is_email_taken(email_content):
            email_validation = False
            return render_template('info.html',email_validation=email_validation)
        if password_content == password_content_2:
            new_person = Person(username=username_content, 
                          password=password_content, 
                          email=email_content)
            
            try:
                with app.app_context():
                    db.session.add(new_person)
                    db.session.commit()
                    login_user(new_person, remeber=True)
                    top_headlines = newsapi.get_top_headlines(country="us",language="en")
                    total_results = top_headlines['totalResults']
                    if total_results > 100:
                        total_results = 100
                    all_headlines = newsapi.get_top_headlines(country="us", language="en", page_size=total_results)['articles']
                    response = make_response(render_template('home.html',all_headlines=all_headlines,new_person=new_person))
                    response.set_cookie("Person1",username_content)
                    return response
            except:
                return 'There was an issue adding your person'
        else:
            validation_password = False
            return render_template('info.html',validation_password=validation_password)
    else:
        Person1 = request.cookies.get('Person1')
        return render_template('info.html',Person1=Person1)
    return render_template('home.html')

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