from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from newsapi import NewsApiClient
from datetime import datetime
app = Flask(__name__)
newsapi = NewsApiClient(api_key='230b4a51a01f4de2ba0329e873fa5fe3')
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

    open_price = db.Column(db.Integer, default=0)
    high_price = db.Column(db.Integer, default=0)
    low_price = db.Column(db.Integer, default=0)
    adj_close = db.Column(db.Integer, default=0)
    volume = db.Column(db.Integer,default = 0)


    def __repr__(self):
        return f'<Stock "{self.name}">'
    
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

@app.route("/", methods=['GET','POST'])
def home():
    persons = Person.query.all()
    if request.method == "POST":

        sources, domains = get_sources_and_domains()
        keyword = request.form["keyword"]
        related_news = newsapi.get_everything(q=keyword,
                                              sources=sources,
                                              domains=domains,
                                              language='en',
                                              sort_by='relevancy')
        no_of_articles = related_news['totalResults']
        if no_of_articles > 100:
            no_of_articles = 100
        all_articles = newsapi.get_everything(q=keyword,
                                              sources=sources,
                                              domains=domains,
                                              language='en',
                                              sort_by = 'relevancy',
                                              page_size = no_of_articles)['articles']
        return render_template("home.html", all_articles = all_articles,
                               keyword=keyword, persons=persons)
    else:
        top_headlines = newsapi.get_top_headlines(country="us", language="en")
        total_results = top_headlines['totalResults']
        if total_results > 100:
            total_results = 100
        all_headlines = newsapi.get_top_headlines(country="us", language="en", page_size=total_results)['articles']
        return render_template("home.html", all_headlines = all_headlines, persons=persons)
    return render_template("home.html")
@app.route('/person', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        person_content = request.form['person_content']
        new_person = Person(content=person_content)
        try: 
            db.session.add(new_person)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding a person'
    else:
        persons = Person.query.order_by(Person.date_created).all()
        return render_template('index.html',persons=persons)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)