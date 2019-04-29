import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def main():
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
    # 1st Fully Connected Layexr
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
    model.add(Dense(17))
    model.add(Activation('softmax'))
    
    model.summary()
    
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 



    #add images to the model
    
    #from keras.preprocessing.image import ImageDataGenerator
    #train_datagen = ImageDataGenerator( rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    #test_datagen = ImageDataGenerator(rescale = 1./255)
    #training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (224, 224,3),batch_size = 32,class_mode = 'binary')
    #test_set = test_datagen.flow_from_directory('dataset/test_set',target_size = (224, 224,3),batch_size = 32,class_mode = 'binary')
    #classifier.fit_generator(training_set,
    #steps_per_epoch = 8000,
    #epochs = 25,
    #validation_data = test_set,
    #validation_steps = 2000)    
    
    filepath='/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img/0000417.jpg'
    test_image = image.load_img(filepath, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #training_set.class_indices    


if __name__ == "__main__":
    main()