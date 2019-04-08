import requests
jsondata = """{"data": [
    {"description": "YELLOW COAT RACK PARIS FASHION", 
    "quantity": "100", 
    "unitprice": "4.95", 
    "timestamp": "03-12-2010 11:26"
    }]}"""
headers = {'Content-Type': 'application/json'}
uri = "http://20.185.111.89:80/score"
resp = requests.post(uri, jsondata, headers = headers)
country_value = int(resp.text.strip('[]'))
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

if country_value in country:
	prediction = country[country_value]
	print prediction 
else:
	prediction = "Not found"
	print prediction