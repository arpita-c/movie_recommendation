import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import keras
#from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import mean_absolute_error


dataset = pd.read_csv("ml-100k/u.data",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
n_latent_factors = 3

#print (dataset.head())

#print (len(dataset.user_id.unique()), len(dataset.item_id.unique()))

dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values

print dataset.user_id

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2)


n_latent_factors_user = 5
n_latent_factors_movie = 8

movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
movie_vec = keras.layers.Dropout(0.2)(movie_vec)


user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))
user_vec = keras.layers.Dropout(0.2)(user_vec)


concat = keras.layers.merge([movie_vec, user_vec], mode='concat',name='Concat')
concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(20,name='FullyConnected')(concat)
dropout_1 = keras.layers.Dropout(0.5,name='Dropout')(dense)
dense_2 = keras.layers.Dense(20,name='FullyConnected-1')(concat)
dropout_2 = keras.layers.Dropout(0.5,name='Dropout')(dense_2)
dense_3 = keras.layers.Dense(20,name='FullyConnected-2')(dense_2)
dropout_3 = keras.layers.Dropout(0.5,name='Dropout')(dense_3)
dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)


result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
adam = Adam(lr=0.005)
model = keras.Model([user_input, movie_input], result)
model.compile(optimizer=adam,loss= 'mse')

#print(model.summary())

#history = model.fit([train.user_id, train.item_id], train.rating, epochs=250, verbose=0,validation_data=([test.user_id, test.item_id]), test.rating))
#BASELINE = model.fit([train.user_id, train.item_id], train.rating, batch_size=64, nb_epoch=250, validation_data=([test.user_id, test.item_id]), test.rating))
#BASELINE = model.fit([train.user_id, train.item_id], train.rating, batch_size=64, nb_epoch=50, 
#                      validation_data=([test.user_id, test.item_id], test.rating))

BASELINE = model.fit([train.user_id, train.item_id], train.rating, epochs=64,verbose=1, validation_data=([test.user_id, test.item_id], test.rating))

plt.plot(BASELINE.history['loss'], label = 'loss')
plt.plot(BASELINE.history['val_loss'], label = 'val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.show()


y_hat_2 = np.round(model.predict([test.user_id, test.item_id]),0)
y_true = test.rating
print(mean_absolute_error(y_true, y_hat_2))

'''
y_hat_2 = np.round(model.predict([test.user_id, test.item_id]),0)
y_true = test.rating
print(mean_absolute_error(y_true, y_hat_2))

print(mean_absolute_error(y_true, model.predict([test.user_id, test.item_id])))


#five fold cross validation:

#fold1
dataset1_train = pd.read_csv("ml-100k/u1.base",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
dataset1_test = pd.read_csv("ml-100k/u1.test",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
history = model.fit([dataset1_train.user_id, dataset1_train.item_id], dataset1_train.rating, epochs=250, verbose=0)
y_hat_2 = np.round(model.predict([dataset1_test.user_id, dataset1_test.item_id]),0)
y_true = dataset1_test.rating
print(mean_absolute_error(y_true, y_hat_2))

#fold2
dataset2_train = pd.read_csv("ml-100k/u2.base",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
dataset2_test = pd.read_csv("ml-100k/u2.test",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
history = model.fit([dataset2_train.user_id, dataset2_train.item_id], dataset2_train.rating, epochs=250, verbose=0)
y_hat_2 = np.round(model.predict([dataset2_test.user_id, dataset2_test.item_id]),0)
y_true = dataset2_test.rating
print(mean_absolute_error(y_true, y_hat_2))



#fold3
dataset3_train = pd.read_csv("ml-100k/u3.base",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
dataset3_test = pd.read_csv("ml-100k/u3.test",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
history = model.fit([dataset3_train.user_id, dataset3_train.item_id], dataset3_train.rating, epochs=250, verbose=0)
y_hat_2 = np.round(model.predict([dataset3_test.user_id, dataset3_test.item_id]),0)
y_true = dataset3_test.rating
print(mean_absolute_error(y_true, y_hat_2))



#fold4
dataset4_train = pd.read_csv("ml-100k/u4.base",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
dataset4_test = pd.read_csv("ml-100k/u4.test",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
history = model.fit([dataset4_train.user_id, dataset4_train.item_id], dataset4_train.rating, epochs=250, verbose=0)
y_hat_2 = np.round(model.predict([dataset4_test.user_id, dataset4_test.item_id]),0)
y_true = dataset4_test.rating
print(mean_absolute_error(y_true, y_hat_2))



#fold5
dataset5_train = pd.read_csv("ml-100k/u5.base",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
dataset5_test = pd.read_csv("ml-100k/u5.test",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
history = model.fit([dataset5_train.user_id, dataset5_train.item_id], dataset5_train.rating, epochs=250, verbose=0)
y_hat_2 = np.round(model.predict([dataset5_test.user_id, dataset5_test.item_id]),0)
y_true = dataset5_test.rating
print(mean_absolute_error(y_true, y_hat_2))

'''




