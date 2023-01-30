
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Embedding,dot
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[421]:


from numpy.random import seed
seed(1)
tf.random.set_seed(2)
tf.keras.utils.set_random_seed(
    2
)


# In[422]:


dtf_products = pd.read_csv("/Users/ayahassan/Desktop/graduation_project/goodreads_10k/books.csv")
print(dtf_products.shape)


# In[423]:


# dtf_products.drop_duplicates(keep='first')
dtf_products.drop_duplicates(subset=['title','authors'],keep='first',inplace=True)
print(dtf_products.shape)


# In[424]:


df = pd.read_csv("/Users/ayahassan/Desktop/graduation_project/new dataset/ratings.csv",low_memory=False)
print(df.head(10))
print(df.shape)
#remove duplicate ratings for the same book by the same user
df = df.drop_duplicates(
  subset = ['user_id', 'book_id'],
  keep = 'first').reset_index(drop = True)
print(df.shape)




# In[425]:


# drop the row if it has at least one NaN value
df.dropna(axis=0)
print(df.shape)


# In[426]:


X_book, X_user,y_old = df.values[:,1],df.values[:,0],df.values[:,2] 
user_ids = np.unique(X_user).tolist()
book_ids = np.unique(X_book).tolist()
num_users = len(user_ids)
num_books = len(book_ids)
print(X_book.shape)
print(X_user.shape)
print (y_old)
print(y.shape)


# In[427]:


min_rating = min(y_old)
max_rating = max(y_old)
print(
    "Number of distinct users: {}, Number of distinct books: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_books, min_rating, max_rating
    )
)
X = df[["book_id","user_id"]].values
xmax, ymax = X.max(axis=0)
print(f"X max, y max {xmax} {ymax}")
print(df.max())
print(df.min())
print(y_old)
print(y_old.shape)
print(X.shape)
print(X)


# In[428]:


# print(len(df[df["user"] == 1]))
dictionary={}
for i in user_ids:
    dictionary[i] = len(df[df["user_id"] ==i])


# In[429]:


print(dictionary)


# In[430]:


plt.bar(dictionary.keys(), dictionary.values(), 120, color='g')


# In[431]:


filtered_vals = [v for _, v in dictionary.items() if v != 0]
average = sum(filtered_vals) / len(filtered_vals)
print(average)
print(max(dictionary.values()))
print(min(dictionary.values()))


# In[432]:


less = 0
more = 0
for k,v in enumerate(dictionary):
    if (v > 100):
        more+=1
    else:
        less+=1
        
print(more)
print(less)


# In[ ]:





# In[433]:


cleaning_list = []
for ind in user_ids:
    if dictionary[ind] < 100 :
        print(ind)
        cleaning_list+=[ind]
print(cleaning_list)


# In[434]:


df = df[~df['user_id'].isin(cleaning_list)]
print(df.shape)


# In[435]:


X_book, X_user,y_old = df.values[:,1],df.values[:,0],df.values[:,2] 
print(X_book.shape)
print(X_user.shape)
print (y_old)
print(y_old.shape)
user_ids = np.unique(X_user).tolist()
book_ids = np.unique(X_book).tolist()
num_users = len(user_ids)
num_books = len(book_ids)


# In[436]:


min_rating = min(y_old)
max_rating = max(y_old)
print(
    "Number of distinct users: {}, Number of distinct books: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_books, min_rating, max_rating
    )
)
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
bookencoded2book = {i: x for i, x in enumerate(book_ids)}
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
df["book"] = df['book_id'].map(book2book_encoded)
df["user"] = df['user_id'].map(user2user_encoded)
y_temp = df["book_id"].values
X = df[["book","user"]].values
xmax, ymax = X.max(axis=0)
print(f"X max {xmax} {ymax}")
print(df.max())
print(df.min())
print(y_old)
print(y_old.shape)
print(X.shape)
print(X)


# In[114]:


user_avg = {}
for i in df["user"]:
    avg = 0
    temp = df.loc[df['user'] == i]
    for j in temp["rating"]:
        avg = avg+j
    user_avg[i] = avg/temp.shape[0]


# In[437]:


user_avg


# In[438]:


# global_avg = 0
# for i in range(len(X)):
#     global_avg += dtf_products[dtf_products['id'] == y_temp[i]]["average_rating"]
# global_avg/len(X)


# In[439]:


print(max(user_avg.values()))
print(min(user_avg.values()))


# In[440]:


# global_avg
print(y_old)


# # Normalize the ratings

# In[512]:


y = np.zeros((len(X),), dtype=float)
# y[i]-user_avg[X[i][1]]-dtf_products.[dtf_products['id'] == y_temp[i]]["average_rating"]
for i in range(len(X)):
    y[i] = y_old[i] - user_avg[X[i][1]]


# In[513]:


print(y[:200])


# In[514]:


print(y_old[:200])


# In[515]:


print(X[:200])


# In[516]:


print(user_avg[1])


# In[ ]:





# In[517]:


print(list(user_avg.keys())[list(user_avg.values()).index(5)])
print(user_avg[4874])
print(list(user_avg.keys())[list(user_avg.values()).index(1)])
print(user_avg[35756])


# In[518]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)


# In[492]:


print(y)


# In[632]:


from keras.constraints import non_neg
class RecommenderNetPlain(keras.Model):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNetPlain, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
#             embeddings_initializer=tf.keras.initializers.Zeros(),
            embeddings_initializer= keras.initializers.he_normal( seed = 5),
            embeddings_regularizer=keras.regularizers.l2(1e-6),
#             embeddings_constraint=non_neg(),
        )
#         self.user_bias = layers.Embedding(
#             num_users,
#             1,
# #             embeddings_initializer=tf.keras.initializers.Zeros(),
#             embeddings_initializer= keras.initializers.he_normal( seed = 5),
#             embeddings_regularizer=keras.regularizers.l2(1e-6),
# #             embeddings_constraint=non_neg(),
#         )
        self.book_embedding = layers.Embedding(
            num_books,
            embedding_size,
            embeddings_initializer=keras.initializers.he_normal( seed = 5),
#             embeddings_initializer=tf.keras.initializers.Zeros(),
#             embeddings_initializer=tf.keras.initializers.RandomNormal(seed=1),
            embeddings_regularizer=keras.regularizers.l2(1e-6),
#             embeddings_constraint=non_neg(),
        )
        self.book_flatten = layers.Flatten()
        self.user_flatten = layers.Flatten()
#         self.book_bias = layers.Embedding(
#             num_books,
#             1,
# #             embeddings_initializer=tf.keras.initializers.Zeros(),
#             embeddings_initializer= keras.initializers.he_normal( seed = 5),
#             embeddings_regularizer=keras.regularizers.l2(1e-6),
# #             embeddings_constraint=non_neg(),
#         )
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,1])
        book_vector = self.book_embedding(inputs[:,0])
#         user_bias = self.user_bias(inputs[:,1])
#         book_bias = self.book_bias(inputs[:,0])
        book_flatten = self.book_flatten(book_vector)
        user_flatten = self.user_flatten(user_vector)
        prod = dot([user_flatten, book_flatten], axes=1, normalize=False)
#         temp = prod
        x = prod
        return x


# In[633]:


class RecommenderNetwithConcatenate(keras.Model):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNetwithConcatenate, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
#             embeddings_initializer="he_normal",
            embeddings_initializer=keras.initializers.he_normal( seed = 5),
            embeddings_regularizer=keras.regularizers.l2(1e-6),
#             embeddings_constraint=non_neg(),

        )
        self.book_embedding = layers.Embedding(
            num_books,
            embedding_size,
#             embeddings_initializer="he_normal",
            embeddings_initializer=keras.initializers.he_normal( seed = 5),
            embeddings_regularizer=keras.regularizers.l2(1e-6),
#             embeddings_constraint=non_neg(),
        )
        self.book_flatten = layers.Flatten()
        self.user_flatten = layers.Flatten()
        self.concatenated = layers.Concatenate()
        self.dense_1 = layers.Dense(512, activation="sigmoid")
        self.dense_2 = layers.Dense(248, activation="sigmoid")
#         self.dropout1 = layers.Dropout(0.2)
        self.dense_3 = layers.Dense(1, activation="linear")
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 1])
        book_vector = self.book_embedding(inputs[:, 0])
        book_flatten = self.book_flatten(book_vector)
        user_flatten = self.user_flatten(user_vector)
        concatenate = self.concatenated([user_flatten, book_flatten])
        dense_1 = self.dense_1(concatenate)
#         dropout1 = self.dropout1(dense_1)
        dense_2 = self.dense_2(dense_1)
        dense_3 = self.dense_3(dense_2)
        return dense_3
# class RecommenderNetwithConcatenate(keras.Model):
#     def __init__(self, num_users, num_books, embedding_size):
#         super(RecommenderNetwithConcatenate, self).__init__()
#         self.num_users = num_users
#         self.num_books = num_books
#         self.embedding_size = embedding_size
#         self.user_embedding = layers.Embedding(
#             num_users,
#             embedding_size,
#             embeddings_initializer="he_normal",
#             embeddings_regularizer=keras.regularizers.l2(1e-2),

#         )
# #         self.user_bias = layers.Embedding(num_users, 1)
#         self.book_embedding = layers.Embedding(
#             num_books,
#             embedding_size,
#             embeddings_initializer="he_normal",
#             embeddings_regularizer=keras.regularizers.l2(1e-2),
#         )
# #         self.book_bias = layers.Embedding(num_books, 1)
#         self.book_flatten = layers.Flatten()
#         self.user_flatten = layers.Flatten()
#         self.concatenated = layers.Concatenate()
#         self.dense_1 = layers.Dense(248, activation="relu")
#         self.dropout1 = layers.Dropout(0.5)
#         self.dense_2 = layers.Dense(128, activation="relu")
#         self.dropout2= layers.Dropout(0.5)
#         self.dense_3 = layers.Dense(1, activation="relu")
    

#     def call(self, inputs):
#         user_vector = self.user_embedding(inputs[:, 1])
# #         user_bias = self.user_bias(inputs[:, 1])
#         book_vector = self.book_embedding(inputs[:, 0])
# #         book_bias = self.book_bias(inputs[:, 0])
#         book_flatten = self.book_flatten(book_vector)
#         user_flatten = self.user_flatten(user_vector)
#         concatenate = self.concatenated([user_vector, book_vector])
# #         temp = prod+user_bias+book_bias
# #         dense_1 = self.dense_1(temp)
# #         dropout1 = self.dropout1(dense_1)
# #         dense_2 = self.dense_2(dense_1)
# #         dropout2 = self.dropout1(dense_2)
#         dense_1 = self.dense_1(concatenate)
#         dropout1 = self.dropout1(dense_1)
#         dense_2 = self.dense_2(dropout1)
#         dropout2 = self.dropout2(dense_2)
#         dense_3= self.dense_3(dropout2)





# In[634]:


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
#             embeddings_initializer="he_normal",
            embeddings_initializer=keras.initializers.he_normal( seed = 5),
            embeddings_regularizer=keras.regularizers.l2(1e-6),
#             embeddings_constraint=non_neg(),
        )
        self.book_embedding = layers.Embedding(
            num_books,
            embedding_size,
#             embeddings_initializer="he_normal",
            embeddings_initializer=keras.initializers.he_normal( seed = 5),
            embeddings_regularizer=keras.regularizers.l2(1e-6),
#             embeddings_constraint=non_neg(),
        )
        self.book_flatten = layers.Flatten()
        self.user_flatten = layers.Flatten()
        self.multiplied = layers.Multiply()
        self.dense_1 = layers.Dense(248, activation="sigmoid")
        self.dropout1 = layers.Dropout(0.5)
        self.dense_2 = layers.Dense(128, activation="sigmoid")
#         self.dropout2 = layers.Dropout(0.5)
        self.dense_3 = layers.Dense(64, activation="tanh")
        self.dense_4 = layers.Dense(32, activation="tanh")
        self.dense_5 = layers.Dense(1, activation="linear")
        self.concatenated1 = layers.Concatenate()
        self.concatenated2 = layers.Concatenate()
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 1])
        book_vector = self.book_embedding(inputs[:, 0])
        book_flatten = self.book_flatten(book_vector)
        user_flatten = self.user_flatten(user_vector)
        prod = self.multiplied([user_vector, book_vector])
        concatenated_1 = self.concatenated1([user_flatten, book_flatten])
        dense_1 = self.dense_1(concatenated_1)
        dropout1 = self.dropout1(dense_1)
        dense_2 = self.dense_2(dropout1)
#         dropout2 = self.dropout2(dense_2)
        dense_3 = self.dense_3(dense_2)
        dense_4 = self.dense_4(dense_3)
        concatenated_2 = self.concatenated2([prod, dense_4])
        dense_5 = self.dense_5(concatenated_2)
        return dense_5


# In[635]:


from tensorflow.keras import backend as K

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


# In[636]:


def training_RecommenderNetwithConcatenate (batch_size, embedding_size, num_users, num_books):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001,restore_best_weights = True)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 4, 
                                        restore_best_weights = True)
    model = RecommenderNetwithConcatenate(num_users, num_books, embedding_size)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics = [tf.keras.metrics.MeanSquaredError(
            name="mean_squared_error", dtype=None), soft_acc])
    history = model.fit(
        X_train,
        y_train,
        batch_size = batch_size,
        epochs = 60,
        validation_split = 0.1,
        callbacks = [reduce_lr,earlystopping])
    return model, history


# In[637]:


def training_RecommenderNet (batch_size, embedding_size, num_users, num_books):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, restore_best_weights = True)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 4, 
                                        restore_best_weights = True)
    model = RecommenderNet(num_users, num_books, embedding_size)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics = [tf.keras.metrics.MeanSquaredError(
            name="mean_squared_error", dtype=None), soft_acc])
    history = model.fit(
        X_train,
        y_train,
        batch_size = batch_size,
        epochs = 60,
        validation_split = 0.1,
        callbacks = [reduce_lr,earlystopping])
    return model, history


# In[638]:


def training_netplain (batch_size, embedding_size, num_users, num_books):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, restore_best_weights = True)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 4, 
                                        restore_best_weights = True)
    model = RecommenderNetPlain(num_users, num_books, embedding_size)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics = [tf.keras.metrics.MeanSquaredError(
        name="mean_squared_error", dtype=None), soft_acc])
    history = model.fit(
        X_train,
        y_train,
        batch_size = batch_size,
        epochs = 60,
        validation_split = 0.1,
        callbacks = [reduce_lr, earlystopping])
    return model, history


# In[639]:


def evaluation(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    return loss, acc


# # Choosing the best matrix factorization model

# ## testing different batch sizes and different embedding sizes for the RecommenderNetPlain

# In[640]:


print(y_train.shape)
print(X_train.shape)


# In[641]:


#plot the models losses
batch_sizes = [512, 1024, 2048]
models_bs_1 = {}
histories_bs_1 = {}
for b_s in batch_sizes:
    model, history = training_netplain(b_s, 30, num_users, num_books)
    models_bs_1[b_s] = model
    histories_bs_1[b_s] = history


# In[623]:


batch_sizes = [512, 1024, 2048]
for b_s in batch_sizes:
#     loss, acc = evaluation(models_bs_1[b_s], X_test, y_test)
    loss, acc, soft_acc = models_bs_1[b_s].evaluate(X_test, y_test)
    models_bs_1[b_s].save(f"my_model{models_bs_1[b_s]}")
    print(f'loss in the model with batch size = {b_s} is {loss} on test set')
    print(f'accuracy of the model with batch size = {b_s} is {acc} on test set')
    print(f'soft accuracy of the model with batch size = {b_s} is {soft_acc} on test set')
for b_s in batch_sizes:
    plt.plot(histories_bs_1[b_s].history['loss'], label=str(b_s))
    plt.legend()


# In[630]:


y_predicted = models_bs_1[1024].predict(X_test)


# In[631]:


y_predicted[:20]


# In[483]:


y_test[:20]


# In[642]:


# plot the models losses
embedding_sizes = [5, 10, 20, 30, 50, 80, 100]
# embedding_sizes = [50, 80, 100, 150, 200]
models_em_1 = {}
histories_em_1 = {}
for em in embedding_sizes:
    model, history = training_netplain(1024, em, num_users, num_books)
    models_em_1[em] = model
    histories_em_1[em] = history


# In[625]:


embedding_sizes = [5, 10, 20, 30, 50, 80, 100]
for em in embedding_sizes:
    loss, acc, soft_accu = models_em_1[em].evaluate(X_test, y_test)
    models_em_1[em].save(f"my_model{models_em_1[em]}")
    print(f'loss in the model with embedding size = {em} is {loss} on test set')
    print(f'accuracy of the model with embedding size = {em} is {acc} on test set')
    print(f'soft accuracy of the model with embedding size = {em} is {soft_accu} on test set')
for em in embedding_sizes:
    plt.plot(histories_em_1[em].history['loss'], label=str(em))
    plt.legend()


# ## testing different batch sizes and different embedding sizes for the RecommenderNetwithConcatenate which concatenates the embeddings of users and movies, then inputs the concatenation to a dense layer

# In[643]:


# #plot the models losses
batch_sizes = [1024, 2048, 4096]
models_bs_2 = {}
histories_bs_2 = {}
for b_s in batch_sizes:
    model, history = training_RecommenderNetwithConcatenate(b_s, 30, num_users, num_books)
    models_bs_2[b_s] = model
    histories_bs_2[b_s] = history


# In[ ]:


model, history = training_RecommenderNetwithConcatenate(b_s, 5, num_users, num_books)
models_bs_2[1024] = model
histories_bs_2[1024] = history


# In[626]:


batch_sizes = [1024, 2048, 4096]
for b_s in batch_sizes:
    loss, acc, soft_accu = models_bs_2[b_s].evaluate(X_test, y_test)
#     loss2, acc2 = evaluation(models_bs_2[b_s], X_train, y_train)
    models_bs_2[b_s].save(f"my_model{models_bs_2[b_s]}")
    print(f'loss in the model with batch size = {b_s} is {loss} on test set')
    print(f'accuracy of the model with batch size = {b_s} is {acc} on test set')
    print(f'soft accuracy of the model with batch size = {b_s} is {soft_accu} on test set')
for b_s in batch_sizes:
    plt.plot(histories_bs_2[b_s].history['loss'], label=str(b_s))
    plt.legend()


# In[644]:


# #plot the models losses
embedding_sizes = [5, 10, 20, 30, 50, 80, 100]
models_em_2 = {}
histories_em_2 = {}
for em in embedding_sizes:
    model, history = training_RecommenderNetwithConcatenate(1024, em, num_users, num_books)
    models_em_2[em] = model
    histories_em_2[em] = history


# In[ ]:


model, history = training_RecommenderNetwithConcatenate(1024, 20, num_users, num_books)
models_em_2[20] = model
histories_em_2[20] = history
model, history = training_RecommenderNetwithConcatenate(1024, 5, num_users, num_books)
models_em_2[5] = model
histories_em_2[5] = history


# In[627]:


embedding_sizes = [5, 10, 20, 30, 50, 80, 100]
for em in embedding_sizes:
    loss, acc, soft_accu = models_em_2[em].evaluate(X_test, y_test)
#     loss2, acc2 = evaluation(models_em_2[em], X_train, y_train)
    models_em_2[em].save(f"my_model{models_em_2[em]}")
    print(f'loss in the model with embedding size = {em} is {loss} on test set')
    print(f'accuracy of the model with embedding size = {em} is {acc} on test set')
    print(f'soft accuracy of the model with emb size = {em} is {soft_accu} on test set')
for em in embedding_sizes:
    plt.plot(histories_em_2[em].history['loss'], label=str(em))
    plt.legend()


# ###### testing different batch sizes and different embedding sizes for the RecommenderNet which calculates the dot product of the embeddings of users and movies, then inputs the concatenation to a neural network of 5 dense layers and 2 dropout layers

# In[645]:


#plot the models losses
batch_sizes = [2048, 4096, 8192]
models_bs_3 = {}
histories_bs_3 = {}
for b_s in batch_sizes:
    model, history = training_RecommenderNet(b_s, 30, num_users, num_books)
    models_bs_3[b_s] = model
    histories_bs_3[b_s] = history


# In[628]:


batch_sizes = [2048, 4096, 8192]
for b_s in batch_sizes:
    loss, acc, soft_accu = models_bs_3[b_s].evaluate(X_test, y_test)
    models_bs_3[b_s].save(f"my_model{models_bs_3[b_s]}")
#     loss2, acc2 = evaluation(models_bs_3[b_s], X_train, y_train)
    print(f'loss in the model with batch size = {b_s} is {loss} on test set')
    print(f'accuracy of the model with batch size = {b_s} is {acc} on test set')
    print(f'soft accuracy of the model with batch size = {b_s} is {soft_accu} on test set')
for b_s in batch_sizes:
    plt.plot(histories_bs_3[b_s].history['loss'], label=str(b_s))
    plt.legend()


# In[ ]:





# In[ ]:


model, history = training_RecommenderNet(2048, 10, num_users, num_books)
models_bs_3[2048] = model
histories_bs_3[2048] = history


# In[ ]:


#plot the models losses
embedding_sizes = [5, 10, 20, 30, 50, 80, 100]
models_em_3 = {}
histories_em_3 = {}
for em in embedding_sizes:
    model, history = training_RecommenderNet(1024, em, num_users, num_books)
    models_em_3[em] = model
    histories_em_3[em] = history


# In[ ]:


model, history = training_RecommenderNet(4096, 5, num_users, num_books)
models_em_3[5] = model
histories_em_3[5] = history


# In[629]:


embedding_sizes = [5, 10, 20, 30, 50, 80, 100]
for em in embedding_sizes:
    loss, acc, soft_accu = models_em_3[em].evaluate(X_test, y_test)
    models_em_3[em].save(f"my_model{models_em_3[em]}")
#     loss2, acc2 = evaluation(models_em_3[em], X_train, y_train)
    print(f'loss in the model with em size = {em} is {loss} on test set')
    print(f'accuracy of the model with em size = {em} is {acc} on test set')
    print(f'soft accuracy of the model with em size = {em} is {soft_accu} on test set')
for em in embedding_sizes:
    plt.plot(histories_em_3[em].history['loss'], label=str(em))
    plt.legend()


# # visualize the different models

# In[ ]:


from keras.utils.vis_utils import plot_model
batch_sizes1 = [512,1024, 2048, 4096]
embedding_sizes1 = [10, 20, 30, 40, 50]
embedding_sizes2 = [5, 10, 20]
# for em in embedding_sizes2:
# plot_model(models_em_2[5], to_file=f'model_plot{models_em_2[5]}.png', show_shapes=True, show_layer_names=True)
models_em_3[10].summary()
# for em in embedding_sizes1:
#     plot_model(models_em_1[bs], to_file=f'model_plot{models_em_1[em]}.png', show_shapes=True, show_layer_names=True)
#     plot_model(models_em_3[em], to_file=f'model_plot{models_em_3[em]}.png', show_shapes=True, show_layer_names=True)
    


# In[ ]:


get_ipython().system('pip install visualkeras')
import visualkeras
from PIL import ImageFont
visualkeras.layered_view(models_em_1[10], legend=True, font=font) 
visualkeras.layered_view(models_em_2[5], legend=True, font=font) 
visualkeras.layered_view(models_em_3[10], legend=True, font=font) 


# In[898]:


inputs = keras.Input(shape=(2,1))
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(num_users, 10)(u) # (N, 1, K)
m_embedding = Embedding(num_books, 10)(m) # (N, 1, K)
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
x = layers.Concatenate()([u_embedding, m_embedding]) # (N, 2K)
x = layers.Dense(512, activation="tanh")(x)
x = layers.Dense(248, activation="tanh")(x)
x = Dense(1, activation="relu")(x)
model_1 = keras.Model(inputs=(u,m), outputs=x)
# 
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(num_users, 5)(u) # (N, 1, K)
m_embedding = Embedding(num_books, 5)(m) # (N, 1, K)
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
x = dot([u_embedding, m_embedding], axes =1) # (N, 2K)
# x = Dense(1)(x)
model_2 = keras.Model(inputs=(u,m), outputs=x)
# 
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(num_users, 5)(u) # (N, 1, K)
m_embedding = Embedding(num_books, 5)(m) # (N, 1, K)
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
y = layers.Concatenate()([u_embedding, m_embedding])# (N, 2K)
y = layers.Dense(248, activation="tanh")(y)
y = layers.Dropout(0.5)(y)
y = layers.Dense(128, activation="tanh")(y)
y = layers.Dense(64, activation="sigmoid")(y)
y = layers.Dense(32, activation="sigmoid")(y)
x = layers.Multiply()([u_embedding, m_embedding])
z = layers.Concatenate()([x, y])
z = Dense(1, activation="relu")(z)
model_3 = keras.Model(inputs=(u,m), outputs=z)
# 
#  self.book_flatten = layers.Flatten()
#         self.user_flatten = layers.Flatten()
#         self.multiplied = layers.Multiply()
#         self.dense_1 = layers.Dense(248, activation="tanh")
#         self.dropout1 = layers.Dropout(0.5)
#         self.dense_2 = layers.Dense(128, activation="tanh")
# #         self.dropout2 = layers.Dropout(0.5)
#         self.dense_3 = layers.Dense(64, activation="sigmoid")
#         self.dense_4 = layers.Dense(32, activation="sigmoid")
#         self.dense_5 = layers.Dense(1, activation="relu")
#         self.concatenated1 = layers.Concatenate()
#         self.concatenated2 = layers.Concatenate()
#     def call(self, inputs):
#         user_vector = self.user_embedding(inputs[:, 1])
#         book_vector = self.book_embedding(inputs[:, 0])
#         book_flatten = self.book_flatten(book_vector)
#         user_flatten = self.user_flatten(user_vector)
#         prod = self.multiplied([user_vector, book_vector])
#         concatenated_1 = self.concatenated1([user_flatten, book_flatten])
#         dense_1 = self.dense_1(concatenated_1)
#         dropout1 = self.dropout1(dense_1)
#         dense_2 = self.dense_2(dropout1)
# #         dropout2 = self.dropout2(dense_2)
#         dense_3 = self.dense_3(dense_2)
#         dense_4 = self.dense_4(dense_3)
#         concatenated_2 = self.concatenated2([prod, dense_4])
#         dense_5 = self.dense_5(concatenated_2)
# 
# plot_model(model_1, to_file=f'model_plot_1.png', show_shapes=True, show_layer_names=True)
# plot_model(model_2, to_file=f'model_plot_2.png', show_shapes=True, show_layer_names=True)
plot_model(model_3, to_file=f'model_plot_3.png', show_shapes=True, show_layer_names=True)
# 


# In[ ]:





# In[ ]:




