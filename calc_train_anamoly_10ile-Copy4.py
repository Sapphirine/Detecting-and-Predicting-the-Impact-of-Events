
# coding: utf-8

# In[1]:


# !pip install psycopg2_binary


# In[2]:


import psycopg2
import pandas as pd
import time
import datetime


# In[3]:


def processAnomalies(fromid, toid):
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  connw = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")  
  
  cursec = conn.cursor()
  cursec.execute('select * from domain_security')
  securityDF = pd.DataFrame(cursec.fetchall(), columns=[desc[0] for desc in cursec.description])
  
  cur = conn.cursor()
  curw = connw.cursor()

  sql_stmt = "select tickdate, security, id, anamoly_score from tick_anomalies where id between " + str(fromid) + " and " + str(toid) + " order by id;"  
  print sql_stmt
  cur.execute(sql_stmt)  

  count = 0
  row = cur.fetchone()
  sql_stmt = ''
  while row:
    anodate, security, anomid, anoscore = row
    security_id = int(securityDF[securityDF['security']==security]['security_id'])
#     print anodate.date(), security, security_id
    sql_stmt = "select tickdate, field, value from import_raw_ticks "       "where security=E'" + security + "'and field IN (E'PX_LAST', E'VOLUME') and "       "tickdate between to_timestamp(E'" + str(anodate.date()) + "','YYYY-MM-DD') - interval '30 day' and to_timestamp(E'"+str(anodate.date())+"','YYYY-MM-DD') + interval '+10 day' order by field, tickdate; "
#     print sql_stmt
    
    curread = conn.cursor()
    curread.execute(sql_stmt)
    resultDF = pd.DataFrame(curread.fetchall(), columns=[desc[0] for desc in curread.description])
    #print resultDF
#     print anodate
    priceDF = resultDF.loc[resultDF['field'] == 'PX_LAST']
#     print priceDF
    ffidx = priceDF.index[priceDF['tickdate']==anodate]
    fidx = ffidx[0] if ffidx.shape[0] > 0 else 0
#     print fidx, fvalue
    if fidx > 10 and fidx < priceDF.shape[0]-2:
#       print priceDF['value'].loc[fidx-9:fidx]
      label = +1 if priceDF['value'][fidx+1] > priceDF['value'][fidx] else -1
      volDF = resultDF.loc[resultDF['field'] == 'VOLUME']
      vvidx = volDF.index[volDF['tickdate']==anodate]
      vidx = vvidx[0] if vvidx.shape[0] > 0 else 0
#       print vidx
#       print volDF
      if vidx > 10:
        sql_insert = "insert into train_tick_data values (" + ",".join(map(lambda x: str(x),[anomid, "'"+str(anodate)+"'", anoscore, "'"+security+"'", security_id, label] + priceDF['value'].loc[fidx-9:fidx].tolist() + volDF['value'].loc[vidx-9:vidx].tolist())) + ");"
#         print sql_insert
        curw.execute(sql_insert)
        
    
    row = cur.fetchone()
    count = count + 1
    if count % 1000 == 0:
      connw.commit()
      print str(datetime.datetime.now().time()), 'Count:', count
    

  
#   output = output.sort_values(0)
  conn.commit()
  conn.close()  
  connw.commit()
  connw.close()  
  
  print ("DONE..........")
  
  
processAnomalies(350001, 400000)  


# In[4]:




# #print resultDF
# print anodate
# priceDF = resultDF.loc[resultDF['field'] == 'PX_LAST']
# print priceDF
# fidx = priceDF.index[priceDF['tickdate']==anodate][0]
# fvalue = priceDF['value'][fidx]
# print fidx, fvalue
# if fidx > 10 and fidx < priceDF.shape[0]-2:
#   label = +1 if priceDF['value'][fidx+1] > fvalue else -1
#   print label
#   print priceDF['value'].loc[fidx-9:fidx]
#   volDF = resultDF.loc[resultDF['field'] == 'VOLUME']
#   vidx = volDF.index[volDF['tickdate']==anodate][0]
#   print vidx
#   print volDF
#   if vidx > 10:
#     print volDF['value'].loc[vidx-9:vidx]


# In[5]:


# import pandas as pd
# import numpy as np
# df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
#                    'B': 'one one two three two two one three'.split(),
#                    'C': np.arange(8), 'D': np.arange(8) * 2})
# print(df)


# In[6]:


# df.index[df['C']==7].shape[0]

