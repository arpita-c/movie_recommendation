from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd



model_vgg16 = VGG16(weights='imagenet', include_top=False)
model_vgg16.summary()

#img_path = 'train/dogs/1.jpg'
#img_path = '/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img/'
#img = image.load_img(img_path, target_size=(224, 224))
#img_data = image.img_to_array(img)
#img_data = np.expand_dims(img_data, axis=0)
#img_data = preprocess_input(img_data)

#vgg16_feature = model_vgg16.predict(img_data)

#print vgg16_feature.shape


filenames=os.listdir('./img/')
vgg16_feature_list = []

def vgg16_preprocess(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x


cn =0
img_path = '/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img/'

imagefilename=[]

for i, fname in enumerate(filenames):
        # process the files under the directory 'dogs' or 'cats'
        # ...
        img_file_path = img_path+ fname
        img = image.load_img(img_file_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = vgg16_preprocess(img_data)
        
        vgg16_feature = model_vgg16.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)

        imagefilename.append(img_file_path)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        print img_file_path
        cn += 10
        
        
        
vgg16_feature_list_np = np.array(vgg16_feature_list)
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean', n_jobs = -1).fit(vgg16_feature_list_np)

print vgg16_feature_list_np
distances, indices = nbrs.kneighbors(vgg16_feature_list_np)
print indices
print distances


test_file_name= '/home/arpita/Documents/Semester4/movie_recommendation/final_movie_recommendation/img/2474932.jpg'
test_image_index=-1

print imagefilename
for index,filename in enumerate(imagefilename):
        if(filename==test_file_name):
                test_image_index=index
                break


print test_image_index
#print recommended image

for indexval in indices[test_image_index]:
        print indexval
        print imagefilename[indexval]







#nbrs.predict(test_feature_np)[0:5]



#kmeans = KMeans(n_clusters=2, random_state=0).fit(vgg16_feature_list_np)