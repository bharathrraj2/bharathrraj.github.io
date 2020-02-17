---
title: "Fashion Class Classification : Data Analysis & Modelling"
date: 2020-02-17
tags: [data science, classification, machine learning, image processing, deep learning]
header:
  image: "/images/fashion_mnist/out.jpg"
excerpt: "data science, classification, machine learning, image processing, deep learning"
mathjax: "true"
---

# CASE STUDY: FASHION CLASS CLASSIFICATION

### PROBLEM STATEMENT AND BUSINESS CASE

Fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes.

The 10 classes are as follows:
0 => T-shirt/top 1 => Trouser 2 => Pullover 3 => Dress 4 => Coat 5 => Sandal 6 => Shirt 7 => Sneaker 8 => Bag 9 => Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.




#  Importing Data and Reshaping


```python
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns
import random
```


```python
fashion_train_df = pd.read_csv('./fashion-mnist_train.csv',sep=',')
fashion_test_df = pd.read_csv('./fashion-mnist_test.csv', sep = ',')
```


```python
fashion_train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
fashion_train_df.shape
```




    (60000, 785)




```python
fashion_test_df.shape
```




    (10000, 785)




```python
fashion_train_df['filename']=list(range(0,60000))
fashion_train_df['filename']=fashion_train_df['filename'].apply(lambda x: str(x)+'.png')
```


```python
fashion_test_df['filename']=list(range(0,10000))
fashion_test_df['filename']=fashion_test_df['filename'].apply(lambda x: str(x)+'.png')
```


```python
fashion_train_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
      <th>filename</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59995</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59995.png</td>
    </tr>
    <tr>
      <th>59996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59996.png</td>
    </tr>
    <tr>
      <th>59997</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>162</td>
      <td>163</td>
      <td>135</td>
      <td>94</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59997.png</td>
    </tr>
    <tr>
      <th>59998</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59998.png</td>
    </tr>
    <tr>
      <th>59999</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59999.png</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 786 columns</p>
</div>



### Entire set of images are in a single dataframe, to use generators later we will write the individual images to file in train/ and test/ folders


```python
import cv2
```


```python
for row in fashion_train_df.values:
    img = np.array(row[1:-1]).reshape(28,28)
    img=img.astype('uint8')
    filename = 'train/'+row[-1]
    cv2.imwrite(filename, img)

```


```python
for row in fashion_test_df.values:
    img = np.array(row[1:-1]).reshape(28,28)
    img=img.astype('uint8')
    filename = 'test/'+row[-1]
    cv2.imwrite(filename, img)


```

# Re-Import data with required columns only


```python
fashion_train_df = pd.read_csv('./fashion-mnist_train.csv',sep=',',usecols=['label'])
fashion_test_df = pd.read_csv('./fashion-mnist_test.csv', sep = ',',usecols=['label'])
```


```python
fashion_train_df['filename']=list(range(0,60000))
fashion_train_df['filename']=fashion_train_df['filename'].apply(lambda x: str(x)+'.png')
fashion_test_df['filename']=list(range(0,10000))
fashion_test_df['filename']=fashion_test_df['filename'].apply(lambda x: str(x)+'.png')
```


```python
fashion_test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>filename</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>4.png</td>
    </tr>
  </tbody>
</table>
</div>



# Create Image Generator


```python
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```


```python
#### If required we can perform shearing/zooming and increase the input images. 
'''
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
'''
```




    '\ntrain_datagen = ImageDataGenerator(\n                                    rescale=1./255,\n                                    shear_range=0.2,\n                                    zoom_range=0.2,\n                                    horizontal_flip=True)\n\ntest_datagen = ImageDataGenerator(rescale=1./255)\n'




```python
train_generator=train_datagen.flow_from_dataframe(
    dataframe=fashion_train_df, directory='train/', x_col='filename', y_col="label", color_mode = 'grayscale',class_mode="other", target_size=(28,28), batch_size=32)
```

    Found 60000 images.



```python
test_generator=test_datagen.flow_from_dataframe(
    dataframe=fashion_test_df, directory='test/', x_col='filename', y_col="label", color_mode = 'grayscale',class_mode="other", target_size=(28,28), batch_size=32)
```

    Found 10000 images.


#### Loading images from the folder and using the dataframe to map the image to its corresponding classes

# Quick EDA


```python
sns.countplot(fashion_train_df.label)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14bba79e8>




![alt]({{ site.url }}{{ site.baseurl }}/images/fashion_mnist/output_25_1.png)



```python
sns.countplot(fashion_test_df.label)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14a318588>




![alt]({{ site.url }}{{ site.baseurl }}/images/fashion_mnist/output_26_1.png)


#### Both Train and Test data are balanced

# Plot Sample Images


```python
class ImageViewer:
    def read_img(self,id, folder='train'):
        file=folder + '/' + str(id)
        im=cv2.imread(file)
        return im

    def draw_sample_images(self):
        ncols=4
        nrows = 10
        f, ax = plt.subplots(nrows=nrows,ncols=ncols, 
                             figsize=(4*ncols,5*nrows))
        i=-1
        captions=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        for label in [0,1,2,3,4,5,6,7,8,9]:
            i=i+1
            samples = fashion_train_df[fashion_train_df['label']==label]['filename'].sample(ncols).values
            for j in range(0,ncols):
                file_id=samples[j]
                im=self.read_img(file_id)
                ax[i, j].imshow(im)
                ax[i, j].set_title(captions[i], fontsize=16)  
        plt.tight_layout()
        plt.show()
```


```python
ImageViewer().draw_sample_images()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/fashion_mnist/output_30_0.png)


# Building Model


```python
import plaidml.keras
plaidml.keras.install_backend()


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam



# Initialising the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(64,3,3,input_shape = (28,28,1),activation='relu'))

# Step 2 - Pooling & Dropout
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# Step 2b - Adding a second convolutional layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 32, activation='relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer=Adam(lr=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```

    /Users/bharath/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:20: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3), input_shape=(28, 28, 1..., activation="relu")`
    /Users/bharath/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:35: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="relu", units=32)`



```python
classifier.fit_generator(train_generator,
                        steps_per_epoch=100,
                        epochs=50,
                        validation_data=test_generator,
                        validation_steps=100)
```

    Epoch 1/50
    100/100 [==============================] - 5s 46ms/step - loss: 0.3670 - acc: 0.8731 - val_loss: 0.3395 - val_acc: 0.8813
    Epoch 2/50
    100/100 [==============================] - 4s 43ms/step - loss: 0.3666 - acc: 0.8787 - val_loss: 0.3427 - val_acc: 0.8762
    Epoch 3/50
    100/100 [==============================] - 4s 38ms/step - loss: 0.3376 - acc: 0.8819 - val_loss: 0.3360 - val_acc: 0.8900
    Epoch 4/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3479 - acc: 0.8741 - val_loss: 0.3416 - val_acc: 0.8822
    Epoch 5/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.3398 - acc: 0.8787 - val_loss: 0.3626 - val_acc: 0.8719
    Epoch 6/50
    100/100 [==============================] - 4s 38ms/step - loss: 0.3366 - acc: 0.8816 - val_loss: 0.3799 - val_acc: 0.8650
    Epoch 7/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.3238 - acc: 0.8856 - val_loss: 0.3205 - val_acc: 0.8922
    Epoch 8/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.3198 - acc: 0.8797 - val_loss: 0.3491 - val_acc: 0.8813
    Epoch 9/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.3101 - acc: 0.8878 - val_loss: 0.3140 - val_acc: 0.8856
    Epoch 10/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.3342 - acc: 0.8809 - val_loss: 0.3082 - val_acc: 0.8912
    Epoch 11/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.3252 - acc: 0.8850 - val_loss: 0.3124 - val_acc: 0.8894
    Epoch 12/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3499 - acc: 0.8841 - val_loss: 0.3138 - val_acc: 0.8941
    Epoch 13/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.3002 - acc: 0.8931 - val_loss: 0.3122 - val_acc: 0.8903
    Epoch 14/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3101 - acc: 0.8884 - val_loss: 0.2944 - val_acc: 0.9033
    Epoch 15/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3230 - acc: 0.8859 - val_loss: 0.3232 - val_acc: 0.8938
    Epoch 16/50
    100/100 [==============================] - 4s 42ms/step - loss: 0.3231 - acc: 0.8875 - val_loss: 0.2839 - val_acc: 0.8953
    Epoch 17/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3194 - acc: 0.8809 - val_loss: 0.3463 - val_acc: 0.8781
    Epoch 18/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3296 - acc: 0.8787 - val_loss: 0.3004 - val_acc: 0.8969
    Epoch 19/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3182 - acc: 0.8878 - val_loss: 0.2776 - val_acc: 0.9028
    Epoch 20/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3091 - acc: 0.8863 - val_loss: 0.3118 - val_acc: 0.8932
    Epoch 21/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2963 - acc: 0.8937 - val_loss: 0.2832 - val_acc: 0.8959
    Epoch 22/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.3068 - acc: 0.8878 - val_loss: 0.2855 - val_acc: 0.9019
    Epoch 23/50
    100/100 [==============================] - 4s 43ms/step - loss: 0.3268 - acc: 0.8872 - val_loss: 0.2894 - val_acc: 0.9017
    Epoch 24/50
    100/100 [==============================] - 4s 42ms/step - loss: 0.2849 - acc: 0.8994 - val_loss: 0.2818 - val_acc: 0.8959
    Epoch 25/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.3169 - acc: 0.8894 - val_loss: 0.2976 - val_acc: 0.8909
    Epoch 26/50
    100/100 [==============================] - 4s 42ms/step - loss: 0.3085 - acc: 0.8888 - val_loss: 0.3060 - val_acc: 0.8994
    Epoch 27/50
    100/100 [==============================] - 4s 42ms/step - loss: 0.2922 - acc: 0.9003 - val_loss: 0.2937 - val_acc: 0.8926
    Epoch 28/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.3018 - acc: 0.8919 - val_loss: 0.2835 - val_acc: 0.9019
    Epoch 29/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2785 - acc: 0.8975 - val_loss: 0.2906 - val_acc: 0.8969
    Epoch 30/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.2941 - acc: 0.8944 - val_loss: 0.2857 - val_acc: 0.8986
    Epoch 31/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2891 - acc: 0.8997 - val_loss: 0.2665 - val_acc: 0.9116
    Epoch 32/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2831 - acc: 0.8981 - val_loss: 0.2810 - val_acc: 0.9019
    Epoch 33/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.2958 - acc: 0.8953 - val_loss: 0.3088 - val_acc: 0.8891
    Epoch 34/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2961 - acc: 0.8950 - val_loss: 0.2718 - val_acc: 0.8962
    Epoch 35/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2676 - acc: 0.9062 - val_loss: 0.2928 - val_acc: 0.8991
    Epoch 36/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2838 - acc: 0.8966 - val_loss: 0.2693 - val_acc: 0.9055
    Epoch 37/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.2706 - acc: 0.9022 - val_loss: 0.2755 - val_acc: 0.9075
    Epoch 38/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2794 - acc: 0.9012 - val_loss: 0.2882 - val_acc: 0.8944
    Epoch 39/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.3081 - acc: 0.8922 - val_loss: 0.2716 - val_acc: 0.9026
    Epoch 40/50
    100/100 [==============================] - 4s 42ms/step - loss: 0.2812 - acc: 0.9012 - val_loss: 0.2836 - val_acc: 0.8991
    Epoch 41/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2816 - acc: 0.8994 - val_loss: 0.2783 - val_acc: 0.9009
    Epoch 42/50
    100/100 [==============================] - 4s 42ms/step - loss: 0.2506 - acc: 0.9134 - val_loss: 0.3151 - val_acc: 0.8929
    Epoch 43/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2518 - acc: 0.9062 - val_loss: 0.2765 - val_acc: 0.9012
    Epoch 44/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2689 - acc: 0.9012 - val_loss: 0.2703 - val_acc: 0.9044
    Epoch 45/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2864 - acc: 0.9012 - val_loss: 0.2633 - val_acc: 0.9077
    Epoch 46/50
    100/100 [==============================] - 4s 41ms/step - loss: 0.2559 - acc: 0.9066 - val_loss: 0.2671 - val_acc: 0.9087
    Epoch 47/50
    100/100 [==============================] - 4s 39ms/step - loss: 0.2648 - acc: 0.9031 - val_loss: 0.2713 - val_acc: 0.9028
    Epoch 48/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2750 - acc: 0.8997 - val_loss: 0.2845 - val_acc: 0.8982
    Epoch 49/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2727 - acc: 0.9053 - val_loss: 0.2631 - val_acc: 0.9056
    Epoch 50/50
    100/100 [==============================] - 4s 40ms/step - loss: 0.2818 - acc: 0.9016 - val_loss: 0.2699 - val_acc: 0.9056





    <keras.callbacks.History at 0x14d3021d0>



#### We see a 90% accuracy. Lets evaluate and plot the confusion matrix

# Evaluate the model

#### While using generator for evaluation/prediction its important to note the order in which they import the files and our y_test data should match this order. We create a new dataframe ordered_fashion_test_df for this.

#### Also importing the data should have shuffle = off


```python
test_datagen = ImageDataGenerator(rescale=1./255)
```


```python
test_generator=test_datagen.flow_from_dataframe(dataframe=fashion_test_df, directory='test/', x_col='filename', y_col="label", color_mode = 'grayscale',class_mode="other", target_size=(28,28), shuffle=False,batch_size=100)
```

    Found 10000 images.



```python
test_generator.filenames[0:10] # Note the order here is different compared to fashion_test_df
```




    ['0.png',
     '1.png',
     '10.png',
     '100.png',
     '1000.png',
     '1001.png',
     '1002.png',
     '1003.png',
     '1004.png',
     '1005.png']




```python
fashion_test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>filename</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>4.png</td>
    </tr>
  </tbody>
</table>
</div>




```python
ordered_fashion_test_df = pd.DataFrame()
ordered_fashion_test_df['filename']=test_generator.filenames
```


```python
ordered_fashion_test_df.dtypes
```




    filename    object
    dtype: object




```python
ordered_fashion_test_df = ordered_fashion_test_df.merge(fashion_test_df,on='filename',how='left')
```


```python
ordered_fashion_test_df.head() # Correct label order which can be used for confusion matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.png</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.png</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000.png</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get the predictions for the test data
predicted_classes = classifier.predict_generator(test_generator,steps=100,workers=1)
```


```python
predicted_classes
```




    array([[2.9982531e-01, 0.0000000e+00, 2.1725893e-05, ..., 0.0000000e+00,
            2.8640032e-05, 0.0000000e+00],
           [4.7385693e-06, 4.0444899e-01, 3.8743019e-07, ..., 0.0000000e+00,
            2.0861626e-07, 0.0000000e+00],
           [1.4472008e-04, 1.7881393e-07, 2.0235777e-04, ..., 0.0000000e+00,
            1.4603138e-06, 6.2584877e-07],
           ...,
           [1.1920929e-07, 0.0000000e+00, 1.7523766e-05, ..., 6.8545341e-06,
            9.5925224e-01, 0.0000000e+00],
           [2.6696920e-04, 2.9802322e-08, 2.3841858e-06, ..., 8.9406967e-08,
            9.8979205e-01, 1.0728836e-06],
           [8.0236793e-04, 9.9356949e-02, 2.7289987e-04, ..., 2.6822090e-07,
            1.4248490e-03, 2.2649765e-06]], dtype=float32)




```python
predicted_classes = predicted_classes.argmax(axis=1)
```


```python
(predicted_classes)
```




    array([0, 1, 3, ..., 8, 8, 1])




```python
ordered_fashion_test_df.shape
```




    (10000, 2)




```python
y_test = ordered_fashion_test_df.loc[:,'label'].values
```


```python
y_test = y_test.astype(int)
```


```python
y_test
```




    array([0, 1, 3, ..., 8, 8, 1])




```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x183e52128>




![alt]({{ site.url }}{{ site.baseurl }}/images/fashion_mnist/output_54_1.png)



```python
evaluation = classifier.evaluate_generator(test_generator,steps=100)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
```

    Test Accuracy : 0.900


# Summary

#### 1. Accuracy is around 90%. Mainly the wrong predictions are with labels - {0 => T-shirt/top} ,  {4 => Coat} & {6 => Shirt}
#### 2. We can increase accuracy by training the model on all 90,000 images , we did for only 100 (steps) *32 (batch size)  = 3,200 images and 50 epochs.
#### 3. Accuracy can be improved by adding one more layer of convolution+maxpooling+dropout 
#### 4. Accuracy can be improved by performing shearing and zooming on the input images and increasing the number of input samples


```python

```
