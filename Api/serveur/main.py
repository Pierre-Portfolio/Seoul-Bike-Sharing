#import librairies
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from flask import Flask , render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#Charge the Dataset
SeoulBikeDf = pd.read_csv("./static/csv/SeoulBikeData.csv", encoding="latin1")

#Transformation quantitative variable to vector
SeoulBikeDf['Functioning Day'] = SeoulBikeDf['Functioning Day'].replace(to_replace=['No', 'Yes'], value=[0, 1])
SeoulBikeDf['Holiday'] = SeoulBikeDf['Holiday'].replace(to_replace=['No Holiday', 'Holiday'], value=[0, 1])
SeoulBikeDf['Seasons'] = SeoulBikeDf['Seasons'].replace(to_replace=['Winter', 'Spring', 'Summer', 'Autumn'], value=[0, 1, 2, 3])

#We transfrom Date data into 4 new columns :  day, month, years and dayofweek
SeoulBikeDf['Years'] = pd.to_datetime(SeoulBikeDf['Date']).dt.year
SeoulBikeDf['Month'] = pd.to_datetime(SeoulBikeDf['Date']).dt.month
SeoulBikeDf['Day'] = pd.to_datetime(SeoulBikeDf['Date']).dt.day

#Delete columns
SeoulBikeDf = SeoulBikeDf.drop(['Date', 'Dew point temperature(°C)', 'Years'], axis = 1)

#Target & features
x = SeoulBikeDf.drop(['Rented Bike Count'], axis = 1)
y = SeoulBikeDf["Rented Bike Count"]

#Create train & test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    
#Scale
scaler = StandardScaler()
scaler.fit(x_train)       
x_train = scaler.transform(x_train, copy = False)
x_test  = scaler.transform(x_test, copy = False)

#params
algo = XGBRegressor()
hyperparametres = {
    "max_depth" : [4],
    "gamma" : [0.077, 0.5, 0.75, 1]
}

#Fitting Model         
grid = GridSearchCV(algo, hyperparametres, n_jobs=-1)
grid.fit(x_train, y_train)

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def homepredict():
    return render_template('index.html', prediction = request.form['Hour'])

app.run(host='127.0.0.1', port=8080, debug=False)