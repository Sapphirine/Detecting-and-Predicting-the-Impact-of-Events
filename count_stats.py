
# coding: utf-8

# In[1]:


get_ipython().system(u'pip install psycopg2-binary ')


# In[2]:


import psycopg2
import pandas as pd


# In[5]:


sql_stmt = 'select date(newsdate), count(1) from import_news_articles where fulltext is not null group by date(newsdate) ;'
# sql_stmt = 'select date(newsdate), title, count(1) from import_news_articles where newsdate between \'2010-10-01\' and \'2010-10-31\' group by date(newsdate), title having count(1) > 1'
#sql_stnt = 'select title, fulltext, date(newsdate), count(*) from import_news_articles group by title, fulltext, date(newsdate) having count(1) > 1'

if True:
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  cur = conn.cursor()

  cur.execute(sql_stmt)  
  output = pd.DataFrame(cur.fetchall())

  conn.commit()
  conn.close()  
  
  print output
  


# In[4]:



titleDF = output
titleDF.plot(x=0, y=1, legend=False, title = 'Only Titles of Articles')

# fulltextDF.set_index(0)
# fulltextDF.plot()


# In[6]:


fulltextDF = output
fulltextDF.plot(x=0, y=1, legend=False, title = 'Full Text of Articles')


# In[65]:


merged =  titleDF.set_index(0).merge(fulltextDF.set_index(0), how='outer', left_index=True, right_index=True)


# In[66]:


print merged.shape, fulltextDF.shape, titleDF.shape


# In[70]:


merged.plot(x=2, y=[0,1])


# In[69]:


merged

