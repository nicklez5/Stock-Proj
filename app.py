from flask import Flask, make_response, render_template, render_template_string, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from eodhd import APIClient
import pandas as pd
import requests
from newsdataapi import NewsDataApiClient
from newsapi import NewsApiClient
from datetime import datetime
import json

from sqlalchemy import URL
app = Flask(__name__)

api = APIClient("6652bd3e397aa5.61249582")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

person_stocks = db.Table(
    "person_stocks",
    db.Column("person_id",db.Integer, db.ForeignKey("person.id")),
    db.Column("stock_id", db.Integer, db.ForeignKey("stock.id")),
)
class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    #money = db.Column(db.Integer,default=4000)
    stockz = db.relationship('Stock',lazy='subquery', secondary=person_stocks, backref='persons')
    
    def __repr__(self):
        return f'<Person "{self.name}">'
class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100),nullable=False)

    start_date = db.Column(db.DateTime,default='2023-01-01')
    end_date = db.Column(db.DateTime,default=datetime.now())



    def __repr__(self):
        return f'<Stock "{self.name}">'
    
@app.route("/", methods=['POST'])
def home():
    api = NewsDataApiClient(apikey='pub_45577c20ab4b3b9164fe39620ed39bdfef60c')
    response = api.news_api()
    print(response)
@app.route('/stocks', methods=['POST'])
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
        my_table = []
        return render_template('stock.html',my_table=empty_table)
    return render_template('stock.html')

@app.route('/person', methods=['POST','GET'])
def index():
    pass
        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)