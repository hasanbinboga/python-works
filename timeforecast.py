import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization            
import os
from influxdb import InfluxDBClient
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta 
influxdb_token = os.getenv('influxdbToken')
client = InfluxDBClient(url="http://influxdb2:80", token=influxdb_token, org = 'influxdata')
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

start_date = datetime(2023, 12, 1)
end_date = datetime(2023, 12, 31)
date_interval = timedelta(days=7)

interval_count = (end_date - start_date) // date_interval
date_range_array = [start_date + i * date_interval for i in range(interval_count + 1)]

results_dict = {}

for date in date_range_array:
    formatted_start_date = date.strftime('%Y-%m-%dT%H:%M:%SZ')
    formatted_end_date = (date + date_interval).strftime('%Y-%m-%dT%H:%M:%SZ')
    if formatted_end_date + str(date_interval) > str(end_date):
        break
    print(formatted_start_date)
    print(formatted_end_date)
    query = f'''from(bucket: "jenkins-uici")
        |> range(start: {formatted_start_date}, stop: {formatted_end_date})
        |> filter(fn: (r) => r["_measurement"] == "jenkins_data" and r["project_name"] =~ /^(?i)(linuxBuild)$/ and r["project_namespace"] =~ /Bots/ )
        |> filter(fn: (r) => r._field == "build_number" or r._field == "_time" )
        |> distinct(column:"_value")
        |> yield()'''

    result = client.query_api().query(org='influxdata', query=query)
    print(result)
    sino_results = []
    result_array = []
    for table in result:
        for record in table.records:
            value = record.get_value()
            sino_results.append(value)
    if sino_results:
        count = len(sino_results)
        results_dict[date.strftime('%B %d')] = count
    results = [(formatted_end_date, count, 'builds') for formatted_end_date, count in results_dict.items()]
    print(results)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
tf.random.set_random_seed(7)
# load the dataset
results_df = pd.DataFrame(results, columns=['Date', 'Builds'])
results_df['Date'] = pd.to_datetime(results_df['Date'])
results_df.set_index('Date', inplace=True)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(results_df['Builds'].values.reshape(-1, 1))
'''dataframe = read_csv('py_uici/Internship_Project/Linux.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')'''
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(100):
	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(dataset)
plt.savefig('sino2.png')
plt.show()
plt.close()