
# coding: utf-8

# In[2]:


# !pip install --upgrade google-cloud-language psycopg2-binary


# In[21]:


from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import psycopg2
import pandas as pd
import time
import datetime
import json


# In[4]:


sql_stmt = 'select * from import_news_articles where id = 18946092 '
# 18932534
if True:
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  cur = conn.cursor()

  cur.execute(sql_stmt)  
  outputDF = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
#   output = output.sort_values(0)
  conn.commit()
  conn.close()  
  
  print outputDF
  


# In[36]:


text1 = 'Apple Inc. (AAPL)  plunged the most in more than four years after posting the slowest profit growth since 2003 and the  weakest  sales increase in 14 quarters, fueling concern that mounting costs and competition may curtail growth.  The  shares  dropped 12 percent to $450.50 at the close in  New York , the biggest decline since September 2008. Fiscal first-quarter profit rose less than 1 percent to $13.1 billion, or $13.81 a share. Sales climbed 18 percent to $54.5 billion, compared with 73 percent growth in the same period a year ago.  Yesterday’s results underscored the rising costs of product overhauls amid competition from  Samsung Electronics Co. (005930)  in the saturating smartphone market. Even as Chief Executive Officer  Tim Cook  guided Apple to record revenue and iPad and iPhone sales, investors worry about management’s ability to keep producing hit products more than a year after the death of co- founder  Steve Jobs .  “People are concerned about how quickly sales are falling off after the initial product launches and whether the company can deliver new and interesting products to reignite growth,” said  Walter Piecyk , co-head of research at BTIG LLC in New York.  Related Stories: Apple suppliers also declined. Assembler Hon Hai Precision Industry Co. dropped 2.9 percent in  Taiwan  and speaker-maker AAC Technologies Holding Inc. plunged 6 percent in  Hong Kong . Samsung, which makes chips for Apple, fell 1.4 percent in  Seoul .  Suppliers Tumble  Circuit-maker Cirrus Logic Inc. retreated 11 percent to $26.71 at the close in New York, while Fusion-io Inc., a producer of flash-memory, decreased 1.1 percent to $21.28.  At least 20 analysts cut their price estimates for Apple shares. The average target is $635.65, down from $716.94 the day before the earnings report, according to data compiled by Bloomberg.  Apple introduced the iPhone 5, iPad mini and a restyled Mac to draw customers in time for the first fiscal quarter, typically its most lucrative.  Through yesterday, the company’s share price decline had shaved about $175 billion from its market capitalization since a September peak and means the company may lose its status as the world’s most valuable company to  Exxon Mobil Corp. (XOM)  Apple’s market value at the close of trading today was $423 billion, compared with Exxon’s $416.5 billion.  Largest Investors  Apple’s ten largest investors hold a combined 272.7 million shares, about 29 percent of the total, and include BlackRock Inc., T. Rowe Price Group Inc. and JPMorgan Chase & Co., according to data compiled by Bloomberg. That number of shares was worth $122.8 billion based on today’s closing price, down from $191.4 billion when Apple closed at a record high in September.  For the fiscal second quarter, now under way, Apple forecast sales of $41 billion to $43 billion. That compares with predictions by analysts for revenue of $45.5 billion. Gross margin will be 37.5 percent to 38.5 percent, Apple said yesterday in a statement. Operating costs for equipment, retail stores and data centers will be $3.8 billion to $3.9 billion.  Apple changed the way it provides financial outlooks to investors, after years of  exceeding  quarterly profit estimates by an average of 26 percent. Rather than providing “conservative” forecasts, Apple expects to report results within its predicted range, Chief Financial Officer  Peter Oppenheimer  said on a conference call.  Smartphone Competitors  Competition from smartphone vendors using  Google Inc. (GOOG) ’s Android software, as well as the lack of a new breakthrough product since the iPad’s 2010 debut, has led Daniel Morris, chief investment officer of Morris Capital Advisors LLC, to reduce his holdings.  “It does raise some red flags,” said Morris, who first bought Apple before the iPhone was introduced.  Cook and Oppenheimer defended the company’s performance, saying iPhone, iPad mini and iMac sales were held back because the company couldn’t manufacture enough keep up with demand.  “There are a few factors that are impacting the year-over- year results, making the strong performance of the business a little harder to see,” Oppenheimer said.  Cook also addressed reports suggesting iPhone demand is waning because the company cut orders for some components.  “It would be impossible to interpret what it would mean for our entire business,” Cook said.  Era Ends  Even so, the results indicate that the period of rapid growth that Apple experienced in the six years after Jobs introduced the iPhone may be coming to an end. Samsung and others have followed Apple’s lead in to the era of mobile touch- screen devices and are grabbing market share by introducing smartphones in various designs and prices. Earlier this month, Samsung reported profit rose 89 percent to 8.8 trillion won ($8.3 billion).  “This confirms some of the worries that some of the investors had,” said  Jack Ablin , chief investment officer at BMO Private Bank. “It’s a gigantic company, so incremental growth is just that much more difficult.”  To grab a larger portion of the global smartphone market, some Apple investors argue the company needs to introduce a lower-priced iPhone that can appeal to customers in  developing countries  where carriers don’t subsidize handset purchases.  IPhone Options  Cook said Apple still offers consumers that option by continuing to sell older models at reduced prices. The iPhone 4 was selling out throughout the quarter, Cook said. In  China , the world’s largest phone market, sales grew 67 percent to $6.8 billion. IPhone sales more than doubled in China, where Apple opened 4 stores during the quarter.  Sales of the iPhone, Apple’s biggest source of  revenue  and profit, reached 47.8 million units, matching the prediction by analysts surveyed by Bloomberg. The company also sold 22.9 million iPads, above the projected 22.4 million units.  While Apple revamped its Mac personal-computer lineup during the quarter, selling 4.1 million Macs, that wasn’t enough to beat analysts’ estimates for 5.1 million units. Apple sold 12.7 million iPods, more than the projection for 11.4 million units. Cook said Apple couldn’t manufacture enough Macs to keep up with demand, holding back sales.  Some investors are looking at the drop-off in Apple’s stock since a September high as a buying opportunity.  “This is not by any stretch of the imagination a broken company,” said  David Rolfe , chief investment officer at Wedgewood Partners.  Cook displayed similar confidence when asked about future products.  “We’re working on some incredible stuff,” he said. “We feel great about what we’ve got in store.”  To contact the reporter on this story: Adam Satariano in  San Francisco  at  asatariano1@bloomberg.net   To contact the editor responsible for this story: Tom Giles at  tgiles5@bloomberg.net'


# In[5]:


newstext = outputDF['fulltext'][0]
# newstext = text1
newstext 


# In[30]:


def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {} - {}'.format(
            index, sentence_sentiment, sentence.text))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0


# def analyze(text):
if True:
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    document = types.Document(content=newstext, type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)
    ent_out = client.analyze_entity_sentiment(document=document)
    # Print the results
    print_result(annotations)


# if __name__ == '__main__':
#     analyze(newstext)


# In[7]:


ent_out.entities[0]


# In[24]:


entities = ent_out.entities

# entity types from enums.Entity.Type
entity_type = ('UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
               'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER')


# In[25]:




ent = entities[0]
print str(ent)
str(entity_type[ent.type])


# In[29]:


mylist = []
for ent in entities:
  print 'MYENT:', ent.name, ent.sentiment.score, ent.sentiment.magnitude, entity_type[ent.type], ent.salience #, str(ent.metadata)
#   print ent.name, ent.sentiment.score, ent.sentiment.magnitude
  mylist.append([ent.name, ent.sentiment.score, ent.sentiment.magnitude])
appledf1 = pd.DataFrame(mylist, columns=['entity', 'score','magnitude'])
# appledf1.to_csv('spx_oil.csv')


# In[82]:


for ent in entities:
  print ent.name, ent.sentiment.score, ent.sentiment.magnitude


# In[83]:


# !ls apple*csv
# !gsutil cp spx_oil*csv gs://finnews-to-spx-prediction/


# In[2]:



sql_stmt = 'select newsdate, title, fulltext from import_news_articles where newsdate between \'2013-01-15\' and \'2013-02-15\' '

if True:
  conn = psycopg2.connect(database="postgres", user = "postgres", password = "swapnil", host = "35.190.146.57", port = "5432")
  cur = conn.cursor()

  cur.execute(sql_stmt)  
  outputDF = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
#   output = output.sort_values(0)
  conn.commit()
  conn.close()  
  
  print outputDF


# In[3]:


outputDF['newsdate'].apply(lambda x: x.date()).unique()


# In[4]:


dates = outputDF['newsdate'].apply(lambda x: x.date()).unique()

countDF = pd.DataFrame(0, index=dates, columns=['tcount', 'tfull']);
countDF


# In[5]:


for index, row in outputDF.iterrows():
  if 'Apple' in row['title'] or 'apple' in row['title']:
    countDF['tcount'][row['newsdate'].date()] = countDF['tcount'][row['newsdate'].date()] + 1
  if row['fulltext'] is not None and ('Apple' in row['fulltext'] or 'apple' in row['fulltext']):
    countDF['tfull'][row['newsdate'].date()] = countDF['tfull'][row['newsdate'].date()] + 1
  


# In[8]:


# sql_stmt = 'select tickdate, cast(value as real)  from import_raw_ticks where security = \'S5IOIL Index\' and field = \'PX_LAST\' '
sql_stmt = 'select tickdate, value from import_raw_ticks where tickdate between \'2013-01-15\' and \'2013-02-15\' and security = \'AAPL UW Equity\' and field = \'VOLUME\' order by tickdate'
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


# In[10]:


countDF = countDF.join(outputDF.set_index('tickdate'), how='outer')


# In[11]:


countDF['totalRef'] = countDF['tcount'] + countDF['tfull']


# In[12]:



countDF


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


fig, ax1 = plt.subplots()
# plt.xticks(outputDF['tickdate'])

ax1.plot(countDF.index, countDF['totalRef'],'b-')
ax1.set_xlabel('AAPL US Equity References Vs Tick Volume')
ax1.set_ylabel('Full + Titles', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(countDF.index, countDF['tickvalue'],'r-')
ax2.set_ylabel('Tick Volume', color='r')
ax2.tick_params('y', colors='r')


fig.tight_layout()
plt.show()

