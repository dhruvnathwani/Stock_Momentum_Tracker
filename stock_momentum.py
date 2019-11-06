import pandas as pd
import requests
import json
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import time

def get_current_price(stock):

  url = 'https://www.alphavantage.co/query?'

  function = 'TIME_SERIES_INTRADAY'

  interval = '1min'

  symbol = str(stock)

  key = '2UZM8DILNESN9BB1'

  request = requests.get(url +
                               'function=' + function +
                               '&symbol=' + symbol +
                               '&interval=' + interval +
                               '&apikey='+ key)



  # need to reformat the function in order to comnbine it to have an index to parse the json tree with
  function_reformatted = function.split("_")

  function_reformatted = str(function_reformatted[0]) + " " + str(function_reformatted[1])

  function_reformatted = function_reformatted.title()

  function_reformatted = function_reformatted + " (" + str(interval) + ")"

  request = request.json()

  request = json.dumps(request,sort_keys = True, indent = 4)

  request = json.loads(request)

  top_node = request[function_reformatted]

  # get a list of all the indexes

  indexer = []

  for x in top_node:
    indexer.append(x)

  # Use the first indexer value to access the most recent price
  current_stock_price = top_node[indexer[0]]['1. open']


  return current_stock_price

def get_previous_prices(stock):

  url = 'https://www.alphavantage.co/query?'

  function = 'TIME_SERIES_INTRADAY'

  interval = '1min'

  symbol = str(stock)

  key = '2UZM8DILNESN9BB1'

  request = requests.get(url +
                               'function=' + function +
                               '&symbol=' + symbol +
                               '&interval=' + interval +
                               '&apikey='+ key)



  # need to reformat the function in order to comnbine it to have an index to parse the json tree with
  function_reformatted = function.split("_")

  function_reformatted = str(function_reformatted[0]) + " " + str(function_reformatted[1])

  function_reformatted = function_reformatted.title()

  function_reformatted = function_reformatted + " (" + str(interval) + ")"

  request = request.json()

  request = json.dumps(request,sort_keys = True, indent = 4)

  request = json.loads(request)

  top_node = request[function_reformatted]

  # get a list of all the indexes

  indexer = []

  for x in top_node:
    indexer.append(x)

  # Use the first indexer value to access the most recent price

  recent_prices = []

  for x in indexer:
    price = top_node[x]['1. open']

    recent_prices.append(price)


  return recent_prices

def predict_next_price(forecast):

  recent_prices = get_previous_prices(stock)

  shape = len(recent_prices)

  df = np.array(recent_prices)

  df = np.flip(df)

  df = pd.DataFrame(df.reshape(shape), columns = ['Prices'])

  df['Predictions'] = df['Prices'].shift(-forecast)

  # Now we need to create the x variable

  x = np.array(df.drop(['Predictions'],1))

  x = x[:-forecast]

  # Now we need to create the y variable

  y = np.array(df['Predictions'])

  y = y[:-forecast]

  # Split the data into training and testing (80/20)

  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
  # Create and train the models we will be using

  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  svr_rbf.fit(x_train,y_train)
  svm_confidence = float(svr_rbf.score(x_test,y_test))


  lr = LinearRegression()
  lr.fit(x_train,y_train)
  lr_confidence = float(lr.score(x_test,y_test))

  svr_ridge = SVR(kernel = 'linear', C = 1e3, gamma = 0.1)
  svr_ridge.fit(x_train, y_train)
  svr_ridge_confidence = float(svr_ridge.score(x_test,y_test))


  # Choose the correct model to use by deciding which has the higher r^2

  optimal = float(max(svr_ridge_confidence, lr_confidence, svm_confidence))

  if optimal == svm_confidence:
    method = svr_rbf
    method_used = 'SVR RBF'
  elif optimal == lr_confidence:
    method = lr
    method_used = 'Linear Regression'
  elif optimal == svr_ridge_confidence:
    method = svr_ridge
    method_used = 'SVR Ridge'

  print("Method Used: " + str(method_used))

  # set x_forecast equal to the last 'n' forecast rows of the original data set from Prices column

  # In simpler terms, find the next value we want to forecast, as this will be the only real value without a prediction next to it as the algorithm can not give predictions for values that have not occurred

  x_forecast = np.array(df.drop(['Predictions'],1))[-forecast:]

  current_price = x_forecast

  # Create prediction for the next stock price

  predicted_price = method.predict(x_forecast)

  return current_price, predicted_price

# We need a variable to choose our stock

stock = str(input("Please choose a ticker symbol: "))

#stock = 'SBUX'

# We need to create a forecast variable in order to create the prediction column
# The prediction column will create a value that is 'n' values ahead of the current value

forecast = 1

# Setting counting variables for function running below
total_predictions = 0
correct_predictions = 0

output = 0

output_final = 0




# We beed to create a loop while the market is open
now = datetime.now()

while now.hour < 14:

    print("------------------------------------------------------------")

    isvalid = False
    while not isvalid:
        try:
            current_price, predicted_price = predict_next_price(forecast)
            isvalid = True
        except:
            print('Please input a valid ticker symbol')
            stock = str(input("Please choose a ticker symbol: "))

    print("------------------------------------------------------------")

    total_predictions = total_predictions + 1

    print ("Current Price: " + str(current_price[0][0]))
    print("Predicted Price: " + str(predicted_price[0]))

    current_price = float(current_price[0][0])
    predicted_price = float(predicted_price[0])

    print(" ")

    if predicted_price > current_price:
        print ("Stockbot predicts the price will INCREASE by: " + str("{0:.4%}".format((float(predicted_price - float(current_price)))/float(current_price))))
        output = 1

    elif current_price > predicted_price:
        print("Stockbot predicts the price will DECREASE by: " + str("{0:.4%}".format((float(current_price - float(predicted_price))/float(current_price)))))
        output = 2

    elif float(predicted_price) == float(current_price):
        print("Stockbot predicts the price will stay the same")
        output = 3



    time.sleep(60)

    price_post_prediction = float(get_current_price(stock))

    up_down = price_post_prediction - current_price

    if up_down > 0:
        output_final = 1

    elif up_down < 0:
        output_final = 2

    elif up_down == 0:
        output_final = 3

    up_down = str(up_down)

    up_down = up_down.format("{0:.4f}")

    up_down = float(up_down)

    print(" ")

    print("Actual Price: " + str(price_post_prediction))

    print(" ")

    if output == output_final:
        correct_predictions = correct_predictions + 1
        if ((predicted_price - current_price) - up_down) != 0:
            print("Stockbot was correct on the trend but was off by: " + str(abs(up_down)))
        elif ((predicted_price - current_price)- up_down) == 0:
            print("Stockbot was perfectly accurate!")
    else:
        print("Stockbot was incorrect on the trend, and off on the prediction by: " + str(abs(up_down)))

    print(" ")

    percentage_correct = correct_predictions / total_predictions

    percentage_correct = "{0:.2%}".format(percentage_correct)

    print("Accuracy: " + str(percentage_correct))

    print("------------------------------------------------------------")
