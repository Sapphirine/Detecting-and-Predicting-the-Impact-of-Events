
# coding: utf-8

# In[1]:


get_ipython().system(u'pip install luminol psycopg2-binary')


# In[1]:


# !pip show luminol


# In[2]:


# import luminol
import psycopg2
import pandas as pd
import time
from luminol import anomaly_detector
import datetime


# In[1]:


# from luminol import anomaly_detector


# In[6]:


# luminol.anomaly_detector.AnomalyDetector(ts)


# In[5]:


# luminol.modules.anomaly_detector.AnomalyDetecor(s)


# In[4]:


# sql_stmt = 'select tickdate, cast(value as real)  from import_raw_ticks where security = \'S5IOIL Index\' and field = \'PX_LAST\' '
sql_stmt = 'select score, count(1) from (select round(CAST (anamoly_score as numeric), 2) as score from tick_anomalies) x group by x.score;        '
if True:
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  cur = conn.cursor()

  cur.execute(sql_stmt)  
  outputDF = pd.DataFrame(cur.fetchall(), columns=['tickdate', 'tickvalue'])
#   output = output.sort_values(0)
  conn.commit()
  conn.close()  
  
#   print output


# In[9]:


outputDF


# In[7]:


outputDF = outputDF.set_index('tickdate')


# In[11]:


outputDF.hist('tickvalue')


# In[8]:


outputDF.plot(x='tickdate', y='tickvalue', legend=False, title='Anomaly Score')


# In[40]:


# output = output.set_index(0)
outputDF['tickepoch'] = outputDF['tickdate'].apply(lambda x: time.mktime(time.strptime(str(x), '%Y-%m-%d %H:%M:%S')))
outputDF = outputDF.set_index('tickepoch')


# In[41]:


outputDF.head()


# In[42]:


# to_dict adds another wrapper dict for each column
ts_dict = outputDF.to_dict()['tickvalue']
detector = anomaly_detector.AnomalyDetector(ts_dict)
anomalies = detector.get_anomalies()



# In[43]:


anomaly = anomalies[0]
print anomaly
print str(datetime.datetime.fromtimestamp(anomaly.start_timestamp))


# In[44]:


import matplotlib.pyplot as plt


# In[47]:


anomaly_list = []
for anomaly in anomalies:
#   print anomaly, str(datetime.datetime.fromtimestamp(anomaly.exact_timestamp))
  anomaly_list.append([datetime.datetime.fromtimestamp(anomaly.exact_timestamp), anomaly.anomaly_score])
anomalyDF = pd.DataFrame(anomaly_list, columns=['anodate', 'anoscore'])
anomalyDF


# In[46]:


anomalyDF['anodate'][1] = datetime.datetime.fromtimestamp(1353333339).date()
# anomalyDF.loc[2] = [datetime.datetime.fromtimestamp(1353765339).date(), 3.267453]
anomalyDF


# In[48]:



plt.plot(outputDF['tickdate'], outputDF['tickvalue'],'b-', anomalyDF['anodate'], anomalyDF['anoscore'],'rs')


# In[49]:


fig, ax1 = plt.subplots()
# plt.xticks(outputDF['tickdate'])

ax1.plot(outputDF['tickdate'], outputDF['tickvalue'],'b-')
ax1.set_xlabel('S5OILG Index (Industry) Oil Gas and Consumable Industry Group GICS')
ax1.set_ylabel('Tick Volume', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(anomalyDF['anodate'], anomalyDF['anoscore'],'rs')
ax2.set_ylabel('Anomaly Score', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()

