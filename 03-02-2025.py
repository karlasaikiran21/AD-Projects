#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Reviews.csv',nrows=500)
df.head(3)


# In[2]:


df.Summary.head()


# In[3]:


df.Text.head()


# In[4]:


#!pip install textblob


# In[5]:


from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Text'] = df['Text'].str.replace('[^\w\s]', ' ')
stop = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['Text'] = df['Text'].apply(lambda x: str(TextBlob(x).correct()))
df['Text'] = df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.Text.head()


# In[6]:


reviews =df
reviews.dropna(inplace=True)
reviews.Score.hist(bins=5,grid=False)
plt.show()
print(reviews.groupby('Score').count().Id)


# In[7]:


score_1=reviews[reviews['Score']==1].sample(n=18)
score_2=reviews[reviews['Score']==2].sample(n=18)
score_3=reviews[reviews['Score']==3].sample(n=18)
score_4=reviews[reviews['Score']==4].sample(n=18)
score_5=reviews[reviews['Score']==5].sample(n=18)


# In[10]:


reviews_sample=pd.concat([score_1,score_2,score_3,score_4,score_5],axis=0)
reviews_sample.reset_index(drop=True,inplace=True)
print(reviews_sample.groupby('Score').count().Id)


# In[14]:


from wordcloud import WordCloud
reviews_str=" ".join(reviews_sample['Summary'].to_numpy())
wordcloud=WordCloud(background_color='white').generate(reviews_str)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[15]:


negative_reviews=reviews_sample[reviews_sample['Score'].isin([1,2])]
positive_reviews=reviews_sample[reviews_sample['Score'].isin([4,5])]
negative_reviews_str=negative_reviews.Summary.str.cat()
positive_reviews_str=negative_reviews.Summary.str.cat()


# In[16]:


get_ipython().system('pip install WordCloud')


# In[19]:


wordcloud_negative=WordCloud(background_color='white')\
    .generate(negative_reviews_str)
wordcloud_positive=WordCloud(background_color='white')\
    .generate(positive_reviews_str)
fig=plt.figure(figsize=(10,10))
ax1=fig.add_subplot(211)
ax1.imshow(wordcloud_negative,interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Reviews with Negative Scores',fontsize=20)
ax2=fig.add_subplot(212)
ax2.imshow(wordcloud_positive,interpolation='bilinear')
ax2.axis=("off")
ax1.set_title('Reviews with positive Scores',fontsize=20)
plt.show()


# In[20]:


get_ipython().system('pip install vaderSentiment')


# In[21]:


import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
plt.style.use('fivethirtyeight')
cp=sns.color_palette()
analyzer=SentimentIntensityAnalyzer()
emptyline=[]
for row in df['Text']:
    vs=analyzer.polarity_scores(row)
    emptyline.append(vs)


# In[22]:


df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()


# In[25]:


df_c=pd.concat([df.reset_index(drop=True),df_sentiments],axis=1)
df_c.head(3)


# In[26]:


df_c['Sentiment']=np.where(df_c['compound']>=0,'positive','Negative')
df_c.head(3)


# In[27]:


result=df_c['Sentiment'].value_counts()
print(result)
result.plot(kind='bar',rot=30)


# In[ ]:




