#this code uses an ensemble of different algorithms
# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib
from sklearn import model_selection
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import re
import numpy as np
import pandas as pd
import urllib
from sklearn.cross_validation import train_test_split
# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Cross validation
from sklearn import cross_validation, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

# Convert text to vector
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, render_template, json, request
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/heartDisease")
def heartDiseaseInput():
    return render_template('heartDisease.html')

@app.route('/heartDiseasePredictionPage',methods=['POST'])
def heartDiseaseClassifier():
    # read the posted values from the UI
    _username = request.form['username']
    _age = request.form['age']
    _sex = request.form['sex']
    _cpt = request.form['chest_pain_types']
    _bp = request.form['resting_BP']
    _cholesterol = request.form['serum_cholesterol']
    _sugar = request.form['bloodSugar']
    _restEcg = random.randint(0,2)
    _maxHeartRate = request.form['maxHeartRate']

    data = pd.read_csv("heart_disease.csv",header=0)
    features=list(data.columns[0:13])
    train, test = train_test_split(data, test_size = 0.1)
    X_train = train[features]
    y_train = train.outcome
    X_test = [_age,_sex,_cpt,_bp,_cholesterol,_sugar,_restEcg,_maxHeartRate,0,2.5,3,0,6]
    print ("input")
    #print (list(X_test))
    my_test = np.asarray(X_test)
    #print (X_train)
    #print (y_train)
    print (my_test.shape)
    heart_dt = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    #print(heart_dt)
    y_pred = heart_dt.predict(my_test.reshape(1,-1))
    #print(y_pred)
    #output=0;
    output= int(y_pred)
    print(output)
    print(_username)
    if output == 0:
        return render_template('heartDiseasePrediction.html',value ="Congratulations!!" +_username+" You are safe from heart disease.Keep visiting nearby hospitals for regular checkups.")
    else:
        return render_template('heartDiseasePrediction.html',value = _username+"There are high chances of having heart disease. Please visit nearby hospital soon.")


if __name__ == "__main__":
    app.run()


