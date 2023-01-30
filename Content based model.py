#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error


# In[2]:


books = pd.read_csv(r'data\books.csv')
tags = pd.read_csv(r'data\tags.csv')
book_tags = pd.read_csv(r'data\book_tags.csv')
ratings = pd.read_csv(r'data\ratings.csv')
to_read= pd.read_csv(r'data\to_read.csv')


# In[3]:


books.rename(columns = {'book_id':'id'}, inplace = True)
books.rename(columns = {'goodreads_book_id':'book_id'}, inplace = True)
ratings= ratings[['book_id','user_id','rating']]


# In[4]:


ratings


# In[5]:


tags


# In[6]:


book_tags


# In[7]:


books


# In[8]:


books.columns


# In[9]:


books=books.drop(['books_count', 'ratings_count', 'best_book_id','work_id',
            'original_publication_year', 'language_code',
            'work_ratings_count','ratings_1', 'ratings_2','ratings_3',
            'ratings_4','ratings_5','image_url','small_image_url',
            'isbn','isbn13', 'work_text_reviews_count', 'average_rating'], axis=1)
books


# In[10]:


len(book_tags['goodreads_book_id'].unique())


# In[11]:


l_list=book_tags.duplicated()
count=0
for i in range(0,len(l_list)):
    if (l_list[i]== True):
        print(book_tags.values[i,:])
        count=count+1
count   


# In[12]:


book_tags=book_tags.drop_duplicates()


# In[13]:


book_tags


# In[14]:


book_tags[book_tags['goodreads_book_id']==1]


# In[15]:


b_tags=pd.merge(tags,book_tags,left_on='tag_id',right_on='tag_id',how='inner')
b_tags[b_tags['tag_id']==595]


# In[16]:


b_tags


# In[17]:


len(b_tags['goodreads_book_id'].unique())


# In[18]:


data = pd.merge(b_tags,books,left_on='goodreads_book_id',right_on='book_id',how='inner')
data


# In[19]:


data=data.drop(['tag_id','goodreads_book_id','count'],axis=1)
data


# In[20]:


data_list = data.groupby(by='book_id')['tag_name'].apply(set)
data_list


# In[21]:


books['tags'] = books['book_id'].apply(lambda x: ' '.join(data_list[x]))


# In[22]:


original_books=books.copy()


# In[23]:


#Text Standardization
def remove_stopwords(data):
    stop_words=stopwords.words('english')
    tokens=word_tokenize(data)
    result=''
    for c in tokens:
        if c not in stop_words:
            result+=c+' '
    return result        


# In[24]:


set(string.punctuation)


# In[25]:


def remove_punctuation(data):
    punct = set(string.punctuation)
    result=''
    for c in data:
        if c not in punct:
            result+=c
        else:
            result+=" "
    return result


# In[26]:


def Stemming(data):
    stemmer=PorterStemmer()
    tokens=word_tokenize(data)
    result=''
    for c in tokens:        
        result+=stemmer.stem(c)+' '
    return result 


# In[27]:


def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)


# In[28]:


def remove_digits(data):
    gigits = ['0','1','2','3','4','5','6','7','8','9']
    result=''
    for c in data:
        if c not in gigits:
            result+=c
    return result 


# In[30]:


def text_standarization(data):
    data=[i.lower() for i in data]
    print('v')
    data=[remove_punctuation(i) for i in data]
    print('vvv')
    data=[remove_stopwords(i) for i in data]
    print('vv')
    data=[Stemming(i) for i in data]
    print('vvvv')
    data=[unidecode(i) for i in data]
    print('vvvvv')
#     data=[remove_digits(i) for i in data]
    return data


# In[31]:


books['title']=text_standarization(list(books['title']))
books['tags']=text_standarization(list(books['tags']))
books


# In[32]:


a=books['authors']
a=[i.lower() for i in a]
a=[i.replace(" ", "") for i in a]  
a=[i.replace(",", " ") for i in a]
a=[unidecode(i) for i in a]
books['authors']=a
books


# In[33]:


books['content']=books['authors']+' '+books['title']+' '+books['tags']
books


# In[34]:


books['tags']


# In[35]:


vectorizer = TfidfVectorizer(ngram_range=(1, 1))
item_features = vectorizer.fit_transform(list(books['content']))
features=vectorizer.get_feature_names_out()
features


# In[36]:


print(item_features.shape)


# In[37]:


item_features


# In[38]:


df=pd.DataFrame(item_features.todense(),columns=features)
df.insert(0, "id",books['id'])
df.insert(1, "book_id",books['book_id'])
df


# In[39]:


features_similarity = cosine_similarity(item_features,item_features)
features_similarity


# In[40]:


len(np.unique(ratings['user_id']))


# In[41]:


users_id=np.unique(ratings['user_id'])
users_id


# In[42]:


def user_train_test(data,user_id):
    user_ratings=[]
    user_train=[]
    user_test=[]
    for j in range(0,data.shape[0]):
        if((data.values[j,:][1])==user_id):
            user_ratings.append(data.values[j,:])
            print(j)
    user_ratings=np.asarray(user_ratings)
    user_ratings = np.unique(user_ratings, axis=0)
    test_size=max(1,int(user_ratings.shape[0]*0.2))
    d = user_ratings
    for k in range(0,test_size):
        user_test.append(d[0])
        d=np.delete(d,0,axis=0)
    user_train = d
    user_test = np.asarray(user_test)
    return user_ratings,user_train,user_test


# In[43]:


def recommend_book(book_title,sim_matrix,o_books,n):
    #get the book_id
    book_row=o_books[o_books['title']==book_title]
    book_id=np.asarray(book_row['book_id'])[0]
    book_index=np.asarray(book_row['id'])[0]-1
    book_sim=sim_matrix[book_index]
    book_sim=np.delete(book_sim,book_index,axis=0)
    top_sim=np.flip(np.argsort(book_sim)[-n:])
    top_books={}
    for i in range(0,top_sim.shape[0]):
        bk_row=o_books[o_books['id']==(top_sim[i])+1]
        b_title=o_books[o_books['id']==(top_sim[i])+1]
        bk_id=int(bk_row['book_id'])
        bk_title=b_title['title']
        bk_title=np.asarray(bk_title)        
        value=bk_title[0]+" , "+str(book_sim[top_sim[i]])
        top_books[int(bk_id)]=value
    return book_id,book_index,top_books


# In[44]:


recommend_book("Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",features_similarity,original_books,5)


# In[45]:


def recommend_user_books(user_id,sim_matrix,o_books,train,n):
#     _,train,_ = user_train_test(ratings,user_id)
    train=np.delete(train,1,axis=1)
    top_book={} 
    result={}
    all_books_ids=o_books['id'].values
    for i in range(0,all_books_ids.shape[0]):
        sum_newbook=0
        sum_sim=0
        bk_row=o_books[o_books['id']==(all_books_ids[i])]
        bk_id=int(bk_row['book_id'])
        bk_index= all_books_ids[i]
        bk_title=bk_row['title']
        bk_title=np.asarray(bk_title)
        if (bk_index in train[:,0]):
            continue
        for i in range(0,train.shape[0]):
            b_id=train[i][0]-1
            b_rat=train[i][1]
#             if(b_rat<4):
#                 continue
            sum_newbook=sum_newbook+(sim_matrix[bk_index-1][b_id]*b_rat)
            sum_sim=sum_sim+sim_matrix[bk_index-1][b_id]
        key=str(bk_index)+" , "+bk_title[0]
        top_book[key]=sum_newbook
    values_list=list(top_book.values())
    top_sim=np.flip(np.argsort(values_list)[-n:])
#     print(top_sim.size)
#     print(list(top_book.items())[1][0])
#     print(list(top_book.items())[1][1])
    for k in range(0,top_sim.size):
        index = top_sim[k]
        result[list(top_book.items())[index][0]]=list(top_book.items())[index][1]
#         result.update(list(top_book.items())[index])
    return result


# In[49]:


def new_user_evaluate(user_id,sim_matrix,o_books,ratings):
    _,train,test = user_train_test(ratings,user_id)
    if((train.shape[0]<1)|(test.shape[0]<1)):
        return 0,0,0,0,0
    m = test.shape[0]
    true_ids=[]
    t_score=[]
    p_score=[]
    for j in range(0,len(list(test))):
        if(list(test)[j][2]>=4):
            true_ids.append(list(test)[j][0]) 
    if(len(true_ids)==0):
        return 0,0,0,0,0
    n=len(true_ids)
    result=recommend_user_books(user_id,sim_matrix,o_books,train,n)
    r=list(result.keys())
    predicted_ids=[]
    for i in range(0,len(r)):
        predicted_ids.append(int(r[i].split(",")[0]))      
    correct=[]
    for i in range(0,len(predicted_ids)):
        if(predicted_ids[i] in true_ids):
            correct.append(predicted_ids[i])
            p_score.append(1)
        else:
            p_score.append(0)
        t_score.append(1)    
    print(t_score) 

    f=f1_score(t_score,p_score)
    return correct,len(correct),len(true_ids),m,f


# In[50]:


import random
randomlist =random.sample(range(1, 53424), 1000)


# In[52]:


avg_mse_=[]
f_score=[]
# acc=[]
for i in range(1,len(randomlist)//4):
    c,l,n,b,f=new_user_evaluate(randomlist[i],features_similarity,original_books,ratings)
    if((c==0)&(l==0)&(n==0)):
        continue
    avg_mse_.append(l/n)
    f_score.append(f)
#     acc.append(acc_)
    print("iteration ",i,":  ",randomlist[i],"  ",c,l,n,b,f)
print('finalllly!')    
print(mean(avg_mse_))
print(mean(f_score))


# ### Try another model with books genres 

# In[53]:


genres=["adolescence","adult","aeroplanes","amish","animals","anthologies","art-and-photography","artificial-intelligence",
"aviation","biblical","biography-memoir","bird-watching","canon","christian","colouring-books","comics-manga","conservation",
"dark","death","diary","disability","dyscalculia","emergency-services","feminism","femme-femme","fiction","football","freight"
"futurism","futuristic","gender","gender-and-sexuality","gettysburg-campaign","graphic-novels-comics",
"graphic-novels-comics-manga","graphic-novels-manga","history-and-politics","holiday","hugo-awards","infant-mortality",
"inspirational","jewellery","lapidary","lgbt","live-action-roleplaying","love","mary-shelley","medical","moroccan","museology"
"native-americans","new-york","non-fiction","novella","occult","paranormal-urban-fantasy","pediatrics","percy-bysshe-shelley",
"planetary-science","poetry","polyamory","pornography","prayer","preservation","productivity","race","relationship","roman",
"romantic","satanism","science-fiction-fantasy","science-nature","sequential-art","sex-and-erotica","sexuality","singularity"
,"soccer","social","space","spirituality","surreal","teaching","textbooks","the-americas","the-united-states-of-america",
"transport","tsars","unfinished","united-states","urban","war","wildlife""witchcraft","women-and-gender-studies","womens",
"wwii-related-fiction"]


# In[54]:


s_genres=text_standarization(genres)


# In[55]:


s_genres


# In[56]:


t_s_genres=[]
for i in s_genres:
    t_s_genres.append(word_tokenize(i))
t_s_genres    


# In[57]:


f_genres=[]
for elem in t_s_genres:
    f_genres.extend(elem)
f_genres 


# In[58]:


tags_genres=[]
for i in list(books['tags']):
    words=word_tokenize(i)
    st=""
    t_list=[]
    for j in words:
        if j in f_genres:
            if j not in t_list:
                st+=j.rstrip()+" "
                t_list.append(j.rstrip())
    tags_genres.append(st.rstrip())
tags_genres    


# In[59]:


books.insert(6,"tages_genres",tags_genres)
books


# In[60]:


books['content_genres']=books['authors']+' '+books['title']+' '+books['tages_genres']
books


# In[61]:


vectorizer_tags = TfidfVectorizer(ngram_range=(2,2))
item_features_tags = vectorizer_tags.fit_transform(list(books['tages_genres']))
features_tags=vectorizer_tags.get_feature_names_out()
features_tags


# In[62]:


features_tags.shape


# In[63]:


item_features_tags.shape


# In[64]:


df_tags=pd.DataFrame(item_features_tags.todense(),columns=features_tags)
df_tags.insert(0, "id",books['id'])
df_tags.insert(1, "book_id",books['book_id'])
df_tags


# In[65]:


features_genres_similarity = cosine_similarity(item_features_tags,item_features_tags)
features_genres_similarity


# In[66]:


recommend_book("Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",features_genres_similarity,original_books,5)


# In[ ]:




