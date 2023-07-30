#!/usr/bin/env python
# coding: utf-8

# In[200]:


import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
warnings.filterwarnings('ignore')


# In[201]:


df = pd.read_csv("C:/Users/Spectre Raven/Downloads/threads_reviews.csv")
df.dtypes


# In[202]:


df.isna().sum()


# In[203]:


df['source'].value_counts()


# In[204]:


plt.pie(df['source'].value_counts(),labels=["Google Play","App Store"], autopct ='%1.1f%%')
plt.show()


# In[205]:


google_rating = df['rating'].loc[df['source']=='Google Play']
apple_rating = df['rating'].loc[df['source']=='App Store']


# In[206]:


sns.histplot(x='rating',data=df)


# In[207]:


df['rating'] = df['rating'].map({1:-1,2:-1,3:0,4:1,5:1})
df['rating']


# In[208]:


df['review_description'] =  df['review_description'].str.replace('[^a-zA-Z0-9]+'," ")


# In[209]:


df['review_description']


# In[210]:


negative=df.loc[df['rating']==-1][['review_description','rating']]
negative


# In[211]:


wordcloud = WordCloud(width=800,height=400, background_color='red').generate(" ".join(negative['review_description'].values))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)


# In[212]:


neutral=df.loc[df['rating']==0][['review_description','rating']]
neutral


# In[213]:


wordcloud = WordCloud(width=800,height=400, background_color='blue').generate(" ".join(neutral['review_description'].values))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)


# In[214]:


positive = df.loc[df['rating']==1][['review_description','rating']]
positive


# In[215]:


wordcloud = WordCloud(width=800,height=400, background_color='green').generate(" ".join(positive['review_description'].values))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)


# In[216]:


ps = PorterStemmer()
df['review_description'] = df['review_description'].apply(lambda x:[ps.stem(i.lower()) for i in word_tokenize(x)]).apply(lambda y:" ".join(y))


# In[217]:


df['review_description']


# In[218]:


from nltk.corpus import stopwords
st = stopwords.words('english')
df['review_description'] = df['review_description'].apply(lambda x:[i for i in word_tokenize(x) if i.lower() not in st]).apply(lambda y:" ".join(y))


# In[219]:


df['review_description']


# In[220]:


df['review_description']=df['review_description'].apply(lambda x:[i for i in word_tokenize(x) if len(i)>2]).apply(lambda y:" ".join(y))
df['review_description']


# In[221]:


tf = TfidfVectorizer()
X = tf.fit_transform(df['review_description'])
X.shape


# In[222]:


y = df['rating'].values
y


# In[223]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[196]:


model = RandomForestClassifier()
model.fit(X,y)


# In[224]:


pickle.dump(model, open('model.pkl','wb'))
pickle.dump(tf, open('tf.pkl', 'wb'))


# In[195]:


print(tf.transform(["Threads is a poor app"]))


# In[230]:


op = model.predict(tf.transform(["Threads is a poor app"]))


# In[231]:


op[0]


# In[ ]:




