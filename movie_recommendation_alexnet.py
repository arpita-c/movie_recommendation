import keras
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


np.random.seed(1000)

#Instantiate an empty model
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

# Output Layer
model.add(Dense(1))
model.add(Activation('softmax'))

model.summary()

# Compile the model
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 
#model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 
model.compile(optimizer='adam',loss= 'mse')


filepath='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/ml-latest-small/dataimage_info.csv'
dataset = pd.read_csv(filepath,sep=',',names="userId,movieId,imdbId,imagepath,rating".split(","),dtype='str')

image_path='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img'
dataset.imagepath = dataset.imagepath.astype('str')
dataset.rating = dataset.rating.astype('category').cat.codes.values

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2)   


datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
       width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15)

train_generator=datagen.flow_from_dataframe(dataframe=train,directory=image_path,x_col="imagepath",y_col="rating",
subset="training",class_mode="other",target_size=(224,224),shuffle=True)



valid_generator=datagen.flow_from_dataframe(dataframe=test,directory=image_path,x_col="imagepath",y_col="rating",
subset="training",class_mode="other",target_size=(224,224),shuffle=True)
#valid_generator=datagen.flow_from_dataframe(dataframe=train,directory=image_path,x_col="imagepath",y_col="rating",
#subset="validation",class_mode="other",target_size=(224,224),shuffle=True, batch_size=32)

#print train_generator1.n
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

#BASELINE=model.fit_generator(generator=train_generator,steps_per_epoch=2,validation_data=valid_generator,validation_steps=2,epochs=2,verbose=1)

#print "Hi"


BASELINE= model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,
                   validation_steps=STEP_SIZE_VALID,epochs=5)



plt.plot(BASELINE.history['loss'], label = 'loss')
plt.plot(BASELINE.history['val_loss'], label = 'val_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.show()
print(model.evaluate_generator(generator=train_generator,steps=2))

print (model.evaluate_generator(generator=valid_generator,steps=2))







#traindf=pd.read_csv('./ml-latest-small/dataimage_info.csv')
##datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

#datagen = ImageDataGenerator(rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #validation_split=0.2)

#filepath='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/image_test/'

#train_generator=datagen.flow_from_dataframe(dataframe=traindf,directory=filepath,x_col="imagepath",y_col="rating",
#subset="training",class_mode="other",target_size=(224,224),shuffle=True,seed=42)


#valid_generator=datagen.flow_from_dataframe(dataframe=traindf,directory=filepath,x_col="imagepath",y_col="rating",
#subset="validation",class_mode="other",target_size=(224,224),shuffle=True, seed=42)

#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,
                    #validation_steps=STEP_SIZE_VALID,epochs=10)

