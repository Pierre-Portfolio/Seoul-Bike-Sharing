#Import librairies
print("Loading Librairies ...")
import pandas as pd
import numpy as np
from math import *

from sklearn.ensemble import BaggingRegressor
from flask import Flask , render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#Charge the Dataset
print("Loading Dataset ...")
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
algo = BaggingRegressor()
hyperparametres = {
    "n_estimators"         : [10, 25],
    'bootstrap'            : [True,False]
}

#Fitting Model         
grid = GridSearchCV(algo, hyperparametres, n_jobs=-1)
grid.fit(x_train, y_train)

app = Flask(__name__)
print("Loading : Done")

#Launch the main page
@app.route("/")
def home():
    return render_template('index.html')

#Launch on the button event
@app.route('/', methods=['POST'])
def homepredict():
    predictDf = [[int(request.form['Hour']), float(request.form['Temperature']), int(request.form['Humidity']), float(request.form['WindSpeed']), float(request.form['Visibility']), float(request.form['SolarRadiation']), float(request.form['Rainfall']), float(request.form['Snowfall']), int(request.form['Season']), int(request.form['Holiday']), int(request.form['FunctioningDay']), int(request.form['Month']), int(request.form['Day'])]]
    result = grid.predict(predictDf)
    return render_template('index.html', prediction = "The prediction for this features is : " + str(ceil(result[0])))

app.run(host='127.0.0.1', port=8080, debug=False)