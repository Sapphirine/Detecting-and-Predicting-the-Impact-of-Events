
# coding: utf-8

# In[3]:


# !pip3 install psycopg2-binary


# In[4]:


import tensorflow as tf
import psycopg2
import pandas as pd
import time
import datetime
import json


# In[12]:


sql_stmt = 'select * from train_tick_data limit 1000'
# 18932534
if True:
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  cur = conn.cursor()

  cur.execute(sql_stmt)  
  trainDF = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
#   output = output.sort_values(0)
  conn.commit()
  conn.close()  
  
  print(trainDF.shape)



# In[13]:


trainDF.columns.values


# In[26]:


feature_columns = []
feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(key='security_id',
          vocabulary_list=list(range(1,753))))
tickcols = ['price_9', 'price_8', 'price_7', 'price_6', 'price_5','price_4', 'price_3', 'price_2', 'price_1', 'price_0', 
            'vol_9','vol_8', 'vol_7', 'vol_6', 'vol_5', 'vol_4', 'vol_3', 'vol_2','vol_1', 'vol_0']                       
for col in tickcols:
  feature_columns.append(tf.feature_column.numeric_column(key=col))

print ("total features: ", len(feature_columns))


# In[ ]:


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[20, 20],
    n_classes=3)


# In[ ]:


get_ipython().system(u'top')

