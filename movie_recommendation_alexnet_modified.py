import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import imageio
from random import randint
from PIL import Image
import random
warnings.filterwarnings('ignore')
from keras.preprocessing import image

import os
np.random.seed(1000)


def generate_model(krow,kcol,usercol,moviecol):
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    model = Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    
    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    
    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    #model.add(Flatten())
    
    
    # Now let's get a tensor with the output of our vision model:
    image_input = Input(shape=(224, 224, 3))
    encoded_image = model(image_input)
    
    # Next, let's define a language model to encode the question into a vector.
    # Each question will be at most 100 word long,
    # and we will index words as integers from 1 to 9999.
    row_input = Input(shape=(krow,), dtype='float32')
    row_embeddings = Embedding(input_dim=300, output_dim=krow, input_length=300)(row_input)
    row_vector = LSTM(1)(row_embeddings)
    
    
    col_input = Input(shape=(kcol,), dtype='float32')
    col_embeddings = Embedding(input_dim=300, output_dim=kcol, input_length=300)(col_input)
    col_vector = LSTM(1)(col_embeddings)
    

    user_input = Input(shape=(1,), dtype='int64', name='user_input')
    movie_input = Input(shape = (1,), dtype = 'int64', name = 'movie_input')
   
    user_embed = Embedding(usercol, 50, input_length =1)(user_input)
    movie_embed = Embedding(moviecol, 50, input_length =1)(movie_input)   
    
    #user_embed = Embedding(1, usercol, input_length=1)(user_input)
    #movie_embed = Embedding(1, moviecol, input_length =1)(movie_input)   
    
    user_vector=LSTM(1)(user_embed)
    movie_vector=LSTM(1)(movie_embed)
    
    # Let's concatenate the question vector and the image vector:
    merged = keras.layers.concatenate([encoded_image,row_vector,col_vector,user_vector,movie_vector])
    
    # And let's train a logistic regression over 1000 words on top:
    output = Dense(1, activation='softmax')(merged)
    
    # This is our final model:
    print image_input.shape
    print row_input.shape
    final_model = Model(inputs=[image_input,row_input,col_input,user_input,movie_input], outputs=output)
    
    print(final_model.summary())

    return final_model


def main():
    
    filepath='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/ml-latest-small/dataimage_info.csv'
    dataset = pd.read_csv(filepath,sep=',',names="userId,movieId,imdbId,imagepath,rating".split(","),dtype='str')
    
    image_path='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img'
    image_new_path='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img_new1/'
    
    dataset.imagepath = dataset.imagepath.astype('str')
    dataset.rating = dataset.rating.astype('category').cat.codes.values
    
    #imagelist=list(dataset.imagepath[1:])
    #print imagelist
    imageuniquelist=list(set(list(dataset['imagepath'])))
    print len(imageuniquelist)
    
    dictvalimage={}
    
    #for idx,fileval in enumerate(imageuniquelist):    
        #filevalarr=fileval.split('/')
        #lenval=len(filevalarr)-1
        
        #filevalname=filevalarr[lenval]
        #movieid=filevalname.split(".")[0]
        ##filepath=image_path+'/'+fileval
        
        #imagepathnew=image_new_path+str(movieid)+".png"
        
        #pic = imageio.imread(fileval)
        #gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
        #gray = gray(pic)      
              
        #data = np.asarray( gray, dtype="int32" )
        ##print data.shape
        #rand_no=randint(0,10)
        #end=k+rand_no
        #data_new=data[rand_no:end,rand_no:end]
        #kplusoneCol=data[...,k+1]
        #kplus1Col_train.append(kplusoneCol)
        #kplusoneRow=data[k+1,...]
        #kplus1Row_train.append(kplusoneRow)
            ##print data_new.shape
            
        #img = Image.fromarray(data_new)
        #filename_new=image_new_path+"/"+movieid+".png"
        #img.save(filename_new)    
        
    
    
    
    
        
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dataset, test_size=0.2)   
    
    train_images=list(train['imagepath'])
    train_rating=list(train['rating'])
    train_userid=list(train['userId'])
    train_movieid=list(train['movieId'])
    #print list(train_images)
    
    k=160
    image_path='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img'
    #image_new_path='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img_new'
    
        
    test_images=list(test['imagepath'])
    test_rating= list(test['rating'])
    test_userid=list(test['userId'])
    test_movieid=list(test['movieId'])    
    
    kplus1Row_train=[]
    kplus1Col_train=[]
    train_image_list=[]
   
    print len(train_images)
    train_rating_final=[]
    test_rating_final=[]
    
    train_userid_final=[]
    test_userid_final=[]
    
    train_movieid_final=[]
    test_movieid_final=[]
    
    for idx,fileval in enumerate(train_images[:2000]):
        
        filevalarr=fileval.split('/')
        lenval=len(filevalarr)-1
        
        filevalname=filevalarr[lenval]
        movieid=filevalname.split(".")[0]
        
        filepathnew=image_new_path+'/'+str(movieid)+".png"
        
        if(os.path.isfile(fileval) and os.path.isfile(filepathnew)):  

            img = image.load_img(filepathnew, target_size=(224,224,3), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            train_image_list.append(img)
            #print idx
            #print movieid
            kplus1Row_train.append(dictvalimage[movieid]["row"])
            kplus1Col_train.append(dictvalimage[movieid]["col"])
            
            train_rating_final.append(train_rating[idx])
            train_userid_final.append(train_userid[idx])
            train_movieid_final.append(train_movieid[idx])
            
            continue
            
        elif(os.path.isfile(fileval) and os.path.isfile(filepathnew)==False):
            pic = imageio.imread(fileval)
            gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
            gray = gray(pic)      
              
            data = np.asarray( gray, dtype="int32" )
            #print data.shape
            rand_no=randint(0,10)
            end=k+rand_no
            data_new=data[rand_no:end,rand_no:end]
            kplusoneCol=data[...,k+1]
            kplus1Col_train.append(kplusoneCol)
            kplusoneRow=data[k+1,...]
            kplus1Row_train.append(kplusoneRow)
            #print data_new.shape
            
            img = Image.fromarray(data_new)
            filename_new=image_new_path+"/"+movieid+".png"
            img.save(filename_new)    
        
            img = image.load_img(filename_new, target_size=(224,224,3), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            train_image_list.append(img)
            dictvalimage[movieid]={}
            dictvalimage[movieid]["row"]=kplusoneRow
            dictvalimage[movieid]["col"]=kplusoneCol
            
            train_rating_final.append(train_rating[idx])
            train_userid_final.append(train_userid[idx])
            train_movieid_final.append(train_movieid[idx])
            
        
        else:
            continue
        
           
    kplus1Row_test=[]
    kplus1Col_test=[]
    test_image_list=[]
    test_rating_final=[]
    for fileval in test_images[:400]:
        
            
        filevalarr=fileval.split('/')
        lenval=len(filevalarr)-1
        
        filevalname=filevalarr[lenval]
        movieid=filevalname.split(".")[0]
        #filepath=image_path+'/'+fileval
        
        filepathnew=image_new_path+'/'+str(movieid)+".png"
        
        if(os.path.isfile(fileval) and os.path.isfile(filepathnew)):  

            img = image.load_img(filepathnew, target_size=(224,224,3), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            test_image_list.append(img)
                   
            kplus1Row_test.append(dictvalimage[movieid]["row"])
            kplus1Col_test.append(dictvalimage[movieid]["col"])
            
            test_rating_final.append(test_rating[idx])
            test_userid_final.append(test_userid[idx])
            test_movieid_final.append(test_movieid[idx])
            
            continue
         
        
        elif(os.path.isfile(fileval) and os.path.isfile(filepathnew)==False):
           
            pic = imageio.imread(fileval)
            gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
            gray = gray(pic)      
              
            data = np.asarray( gray, dtype="int32" )
            #print data.shape
            rand_no=randint(0,10)
            end=k+rand_no
            data_new=data[rand_no:end,rand_no:end]
            kplusoneCol=data[...,k+1]
            kplus1Col_test.append(kplusoneCol)
            kplusoneRow=data[k+1,...]
            kplus1Row_test.append(kplusoneRow)
            #print data_new.shape
            
            img = Image.fromarray(data_new)
            filename_new=image_new_path+"/"+movieid+".png"
            img.save(filename_new)    
        
            img = image.load_img(filename_new, target_size=(224,224,3), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            dictvalimage[movieid]={}
            dictvalimage[movieid]["row"]=kplusoneRow
            dictvalimage[movieid]["col"]=kplusoneCol
            
            test_rating_final.append(test_rating[idx])
            
            test_image_list.append(img)
            test_userid_final.append(test_userid[idx])
            test_movieid_final.append(test_movieid[idx])            

        else:
            continue
             
            
        
    train_image_list_numpy= np.array(train_image_list)
    test_image_list_numpy=  np.array(test_image_list)
    
    #print type(train_image_list_numpy[0])
    kplus1Col_train_numpy=  np.array(kplus1Col_train)
    kplus1Col_test_numpy=  np.array(kplus1Col_test)
    
    kplus1Row_test_numpy= np.array(kplus1Row_test)
    kplus1Row_train_numpy= np.array(kplus1Row_train)
    
    train_rating_numpy= np.array(train_rating_final)
    test_rating_numpy= np.array(test_rating_final)
    
    train_userid_numpy= np.array(train_userid_final)
    train_movieid_numpy= np.array(train_movieid_final)    
    
    test_userid_numpy= np.array(test_userid_final)
    test_movieid_numpy= np.array(test_movieid_final)    
    
    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3)))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='softmax'))
    
    #model.compile(optimizer='adam',loss= 'mse')

    #print len(train_rating_numpy)
    #model.fit(train_image_list_numpy, train_rating_numpy, epochs=10, validation_data=(test_image_list_numpy, test_rating_numpy))
    
    print train_image_list_numpy.shape
    print test_image_list_numpy.shape
    print kplus1Row_train_numpy.shape
    print kplus1Col_test_numpy.shape
    print train_rating_numpy.shape
    print test_rating_numpy.shape
    print train_movieid_numpy.shape
    print test_movieid_numpy.shape

    print train_userid_numpy.shape
    print test_userid_numpy.shape    
    
   # useridcol=len(list(set(train_userid_final)))
   # movieidcol=len(list(set(train_movieid_final)))
    
    train_userid_final= map(int,train_userid_final)
    train_movieid_final= map(int,train_movieid_final)
   
    #useridcol=len(list(set(train_userid_final)))
    #movieidcol=len(list(set(train_movieid_final)))
     
    useridcol=max(train_userid_final)+1
    movieidcol= max(train_movieid_final)+1
    
    model=generate_model(kplus1Row_train[0].shape[0], kplus1Col_train[0].shape[0],useridcol, movieidcol)
    model.compile(optimizer='adam',loss= 'mse')
    history= model.fit([train_image_list_numpy,kplus1Row_train_numpy,kplus1Col_train_numpy,train_userid_numpy,train_movieid_numpy], train_rating_numpy, epochs=1, validation_data=([test_image_list_numpy,kplus1Row_test_numpy,kplus1Col_test_numpy,test_userid_numpy,test_movieid_numpy], test_rating_numpy))
    print "hi"

    plt.clf()
    plt.figure(figsize = (10,7))
    plt.plot(history.history['loss'], label = 'alexnet_loss')
    plt.plot(history.history['val_loss'], label = 'alexnetval_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()
    
    y_hat_2 = np.round(model.predict([test_image_list_numpy,kplus1Row_test_numpy,kplus1Col_test_numpy,test_userid_numpy,test_movieid_numpy]),0)
    y_true = test_rating_numpy
    print y_hate
    print("\n")
    print ("Error")
    print(mean_absolute_error(y_true, y_hat_2))
    
    
    #y_hat_2 = np.round(LSTM_nn.predict([valid.user_id, valid.item_id]),0)
    #y_true = valid.rating
    #print("\n")
    #print ("Error")
    #print(mean_absolute_error(y_true, y_hat_2))    
    #model.fit(x=[None], y=None, batch_size=None, epochs=1, verbose=1, 
             #callbacks=None, validation_split=0., 
             #validation_data=None, shuffle=True, 
             #class_weight=None, sample_weight=None, 
             #initial_epoch=0, steps_per_epoch=None, 
             #validation_steps=None)


        


        
if __name__ == "__main__":
    main()
