import os
import numpy as np
import flask
import json
import requests
from flask import Flask, render_template, request

# creating instance of the class
app = Flask(__name__)

# to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

# prediction function
def ValuePredictor(jsondata):
	headers = {'Content-Type': 'application/json'}
	uri = "http://20.185.111.89:80/score"
	resp = requests.post(uri, jsondata, headers = headers)
	finalclass = int(resp.text.strip('[]'))
	return finalclass

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
    	list1=[]
    	description = request.form['description']
    	quantity = request.form['quantity']
    	unitprice = request.form['unitprice']
    	timestamp = request.form['timestamp']
    	dictionary = {"description": description, "quantity": quantity, "unitprice": unitprice, "timestamp": timestamp}
    	list1.append(dictionary)
    	final_dict = {"data": list1}
    	jsondata = json.dumps(final_dict)
    	result = ValuePredictor(jsondata)
    	country = {
    	36: 'United Kingdom',
    	13: 'France',
    	0: 'Australia',
    	24: 'Netherlands',
    	14: 'Germany',
    	25: 'Norway',
    	10: 'EIRE',
    	33: 'Switzerland',
    	31: 'Spain',
    	26: 'Poland',
    	27: 'Portugal',
    	19: 'Italy',
    	3: 'Belgium',
    	22: 'Lithuania',
    	20: 'Japan',
    	17: 'Iceland',
    	6: 'Channel Islands',
    	9: 'Denmark',
    	7: 'Cyprus',
    	32: 'Sweden',
    	1: 'Austria',
    	18: 'Israel',
    	12: 'Finland',
    	2: 'Bahrain',
    	15: 'Greece',
    	16: 'Hong Kong',
    	30: 'Singapore',
    	21: 'Lebanon',
    	35: 'United Arab Emirates',
    	29: 'Saudi Arabia',
    	8: 'Czech Republic',
    	5: 'Canada',
    	37: 'Unspecified',
    	4: 'Brazil',
    	34: 'USA',
    	11: 'European Community',
    	23: 'Malta',
    	28: 'RSA'
    	}

    	if result in country:
    		prediction = country[result]
    		return render_template("result.html", prediction = prediction)

    	else:
    		prediction = "Not found"
    		return render_template("result.html", prediction = prediction)


if __name__ == '__main__':
	app.run(debug = True, threaded=True)