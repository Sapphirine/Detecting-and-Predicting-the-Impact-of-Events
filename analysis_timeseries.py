
# coding: utf-8

# In[66]:


# !pip install luminol psycopg2-binary
# !pip install git+https://github.com/takuti/anompy.git


# In[1]:


# !pip show luminol


# In[2]:


# import luminol
import psycopg2
import pandas as pd
import time
from luminol import anomaly_detector
import datetime


# In[200]:


def getTicks(security, field, from_date=None, to_date=None):
  if from_date is None:
    stmt = 'select tickdate, value from import_raw_ticks where security = \'' + security + '\' and field = \'' + field + '\' order by tickdate'
  else:
    stmt = 'select tickdate, value from import_raw_ticks where tickdate between \'' + from_date + '\' and \'' + to_date + '\' and security = \'' + security + '\' and field = \'' + field + '\' order by tickdate'
  print stmt
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  cur = conn.cursor()

  cur.execute(stmt)  
  outputDF = pd.DataFrame(cur.fetchall(), columns=['tickdate', 'tickvalue'])
#   output = output.sort_values(0)
  conn.commit()
  conn.close()
  outputDF = outputDF.dropna()  

  return outputDF
  
#   print stmt == sql_stmt
  
def get_anomalies(timeseries, lag_size=10,future_size=5,offset = .4):
  anomlist = [False]*len(timeseries)
  for i in range(lag_size, len(timeseries)-future_size-1):
    cur_avg = (sum(timeseries[i-lag_size:i]) + sum(timeseries[i+1:i+future_size+1])) / (lag_size+future_size)
#     print i, timeseries[i], cur_avg, abs(timeseries[i] - cur_avg) / cur_avg > offset
    anomlist[i] = abs(timeseries[i] - cur_avg) / cur_avg > offset
  return anomlist


# In[203]:


outputDF = getTicks('AAPL UW Equity', 'VOLUME') #, '2013-01-15', '2015-02-15')

outputDF['is_anomaly'] = get_anomalies(outputDF['tickvalue'])
anomalyDF = outputDF[outputDF['is_anomaly']==True]
print outputDF.shape, anomalyDF.shape

plt.plot(outputDF['tickdate'], outputDF['tickvalue'],'b-', anomalyDF['tickdate'], anomalyDF['tickvalue'],'rs')


# In[188]:


# # sql_stmt = 'select tickdate, cast(value as real)  from import_raw_ticks where security = \'S5IOIL Index\' and field = \'PX_LAST\' '


# sql_stmt = 'select tickdate, value from import_raw_ticks where tickdate between \'2013-01-15\' and \'2015-02-15\' and security = \'AAPL UW Equity\' and field = \'VOLUME\' order by tickdate'
# if True:
#   conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
#   cur = conn.cursor()

#   cur.execute(sql_stmt)  
#   outputDF = pd.DataFrame(cur.fetchall(), columns=['tickdate', 'tickvalue'])
# #   output = output.sort_values(0)
#   conn.commit()
#   conn.close()  

#   print output
# outputDF.shape


# In[189]:


outputDF.plot(x='tickdate', y='tickvalue', legend=False, title='AAPL US Equity')


# In[190]:




# print ll

# if True:
#   outputDF['is_anomaly'] = False
#   for i in range(lag_size, len(outputDF.index)-future_size-1):
#     cur_avg = (sum(outputDF['tickvalue'][i-lag_size:i]) + sum(outputDF['tickvalue'][i+1:i+future_size+1])) / (lag_size+future_size)
# #     print i, outputDF['tickvalue'][i], cur_avg, abs(outputDF['tickvalue'][i] - cur_avg) / cur_avg > offset
#     outputDF['is_anomaly'][i] = abs(outputDF['tickvalue'][i] - cur_avg) / cur_avg > offset




# In[191]:





# In[192]:


plt.plot(outputDF['tickdate'], outputDF['tickvalue'],'b-', anomalyDF['tickdate'], anomalyDF['tickvalue'],'rs')


# In[154]:


outputDF[outputDF['is_anomaly']==True].count()


# In[130]:


# import random
# series = [random.random() for i in range(10)]
# ttt = outputDF['tickvalue'].tolist()
# print type(series), type(ttt)
# print series[0], ttt[0]
# # print series,ttt
# print outputDF.head()


# In[ ]:


# # ttt = series
# from anompy.detector.average import AverageDetector
# from anompy.detector.smoothing import ExponentialSmoothing, DoubleExponentialSmoothing, TripleExponentialSmoothing

# # detector = ExponentialSmoothing(ttt[0], alpha=0.9,threshold=.5)

# detector = AverageDetector(ttt[0], window_size=11,threshold=.5)


# forecasted_series = detector.detect(ttt)
# print len(forecasted_series), len(forecasted_series[0])
# print ttt[6:], forecasted_series
# print detector.average


# In[58]:


# output = output.set_index(0)
outputDF['tickepoch'] = outputDF['tickdate'].apply(lambda x: time.mktime(time.strptime(str(x), '%Y-%m-%d %H:%M:%S')))
outputDF = outputDF.set_index('tickepoch')
outputDF.shape


# In[64]:


# to_dict adds another wrapper dict for each column
ts_dict = outputDF.to_dict()['tickvalue']
bit_params = {'precision': 1, 'lag_window_size':1, 'future_window_size':1, 'chunk_size':10}
detector = anomaly_detector.AnomalyDetector(ts_dict, algorithm_name='bitmap_detector', algorithm_params=bit_params)
anomalies = detector.get_anomalies()
print len(anomalies)


# In[67]:


# anomaly = anomalies[0]
# print anomaly
# print str(datetime.datetime.fromtimestamp(anomaly.start_timestamp))


# In[8]:


import matplotlib.pyplot as plt


# In[61]:


anomaly_list = []
for anomaly in anomalies:
#   print anomaly, str(datetime.datetime.fromtimestamp(anomaly.exact_timestamp))
  anomaly_list.append([datetime.datetime.fromtimestamp(anomaly.exact_timestamp), anomaly.anomaly_score])
anomalyDF = pd.DataFrame(anomaly_list, columns=['anodate', 'anoscore'])
# anomalyDF
plt.plot(outputDF['tickdate'], outputDF['tickvalue'],'b-', anomalyDF['anodate'], anomalyDF['anoscore'],'rs')


# In[45]:


fig, ax1 = plt.subplots()
# plt.xticks(outputDF['tickdate'])

ax1.plot(outputDF['tickdate'], outputDF['tickvalue'],'b-')
ax1.set_xlabel('AAPL US Equity - Volume')
ax1.set_ylabel('Close Price', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(anomalyDF['anodate'], anomalyDF['anoscore'],'rs')
ax2.set_ylabel('Anomaly Score', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()

