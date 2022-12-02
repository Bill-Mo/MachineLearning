# %% [markdown]
#  # Lab:  Transfer Learning with a Pre-Trained Deep Neural Network
# 
# As we discussed earlier, state-of-the-art neural networks involve millions of parameters that are prohibitively difficult to train from scratch.  In this lab, we will illustrate a powerful technique called *fine-tuning* where we start with a large pre-trained network and then re-train only the final layers to adapt to a new task.  The method is also called *transfer learning* and can produce excellent results on very small datasets with very little computational time.  
# 
# This lab is based partially on this
# [excellent blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).  In performing the lab, you will learn to:
# * Build a custom image dataset
# * Fine tune the final layers of an existing deep neural network for a new classification task.
# * Load images with a `DataGenerator`.
# 
# The lab has two versions:
# * *CPU version*:  In this version, you use lower resolution images so that the lab can be performed on your laptop.  The resulting accuracy is lower.  The code will also take considerable time to execute.
# * *GPU version*:  This version uses higher resolution images but requires a GPU instance. See the [notes](../GCP/getting_started.md) on setting up a GPU instance on Google Cloud Platform.  The GPU training is much faster (< 1 minute).  
# 
# **MS students must complete the GPU version** of this lab.

# %% [markdown]
# ## Create a Dataset
# 
# In this example, we will try to develop a classifier that can discriminate between two classes:  `cars` and `bicycles`.  One could imagine this type of classifier would be useful in vehicle vision systems.   The first task is to build a dataset.  
# 
# TODO:  Create training and test datasets with:
# * 1000 training images of cars
# * 1000 training images of bicylces
# * 300 test images of cars
# * 300 test images of bicylces
# * The images don't need to be the same size.  But, you can reduce the resolution if you need to save disk space.
# 
# The images should be organized in the following directory structure:
# 
#     ./train
#         /car
#            car_0000.jpg
#            car_0001.jpg
#            ...
#            car_0999.jpg
#         /bicycle
#            bicycle_0000.jpg
#            bicycle_0001.jpg
#            ...
#            bicycle_0999.jpg
#     ./test
#         /car
#            car_1001.jpg
#            car_1001.jpg
#            ...
#            car_1299.jpg
#         /bicycle
#            bicycle_1000.jpg
#            bicycle_1001.jpg
#            ...
#            bicycle_1299.jpg
#            
# The naming of the files within the directories does not matter.  The `ImageDataGenerator` class below will find the filenames.  Just make sure there are the correct number of files in each directory.
#            
# A nice automated way of building such a dataset if through the [FlickrAPI](demo2_flickr_images.ipynb).  Remember that if you run the FlickrAPI twice, it may collect the same images.  So, you need to run it once and split the images into training and test directories.         
#         

# %%
import flickrapi
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import requests
from io import BytesIO
%matplotlib inline

api_key = u'07abaed9e5830f832f294cb75da267f3'
api_secret = u'36353dcc1378cdc5'
flickr = flickrapi.FlickrAPI(api_key, api_secret)


# %%
import os
def create_dir(dir_name): 
    dir_exists = os.path.isdir(dir_name)
    if not dir_exists:
        os.mkdir(dir_name)
        print("Making directory %s" % dir_name)
    else:
        print("Directory %s already exist" % dir_name)

# %%
import warnings

def create_photos(keyword, nimage, dir_name): 
    photos = flickr.walk(text=keyword, tag_mode='all', tags=keyword,extras='url_c',\
                     sort='relevance',per_page=100)
    i = 0
    nrow = 224
    ncol = 224
    for photo in photos:
        url=photo.get('url_c')
        if not (url is None):
            
            # Create a file from the URL
            # This may only work in Python3
            response = requests.get(url)
            file = BytesIO(response.content)
            
            # Read image from file
            im = skimage.io.imread(file)
            
            # Resize images
            im1 = skimage.transform.resize(im,(nrow,ncol),mode='constant')
                    
            # Convert to uint8, suppress the warning about the precision loss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im2 = skimage.img_as_ubyte(im1)
        
            # Save the image
            local_name = '{0:s}/{1:s}_{2:04d}.jpg'.format(dir_name,keyword, i)  
            skimage.io.imsave(local_name, im2)      
            print(local_name)
            i = i + 1        
        if (i >= nimage):        
            break        

# %%
# create_dir('train')
# create_dir('test')
# create_dir('train/car')
# create_dir('train/bicycle')
# create_dir('test/car')
# create_dir('test/bicycle')

# train_num = 1000
# test_num = 300
# create_photos('car', train_num, 'train/car')
# create_photos('bicycle', train_num, 'train/bicycle')
# create_photos('car', test_num, 'test/car')
# create_photos('bicycle', test_num, 'test/bicycle')



# %% [markdown]
# ## Loading a Pre-Trained Deep Network
# 
# We follow the [VGG16 demo](./demo3_vgg16.ipynb) to load a pre-trained deep VGG16 network.  First, run a command to verify your instance is connected to a GPU.

# %%
# TODO 1:
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %% [markdown]
# Now load the appropriate tensorflow packages.

# %%
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense


# %% [markdown]
# We also load some standard packages.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# Clear the Keras session.

# %%
# TODO 2:
import tensorflow.keras.backend as K
K.clear_session()

# %% [markdown]
# Set the dimensions of the input image.  The sizes below would work on a GPU machine.  But, if you have a CPU image, you can use a smaller image size, like `64 x 64`.

# %%
# TODO 3:  Set to smaller values if you are using a CPU.  
# Otherwise, do not change this code.
nrow = 64
ncol = 64

# %% [markdown]
# Now we follow the [VGG16 demo](./vgg16.ipynb) and load the deep VGG16 network.  Alternatively, you can use any other pre-trained model in keras.  When using the `applications.VGG16` method you will need to:
# * Set `include_top=False` to not include the top layer
# * Set the `image_shape` based on the above dimensions.  Remember, `image_shape` should be `height x width x 3` since the images are color.

# %%
# TODO 4:  Load the VGG16 network

input_shape = (nrow, ncol, 3)
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# %% [markdown]
# To create now new model, we create a Sequential model.  Then, loop over the layers in `base_model.layers` and add each layer to the new model.

# %%
# Create a new model
model = Sequential()

# TODO 5:  Loop over base_model.layers and add each layer to model

for layer in base_model.layers: 
    model.add(layer)

# %% [markdown]
# Next, loop through the layers in `model`, and freeze each layer by setting `layer.trainable = False`.  This way, you will not have to *re-train* any of the existing layers.

# %%
# TODO 6
for layer in model.layers: 
    layer.trainable = False

# %% [markdown]
# Now, add the following layers to `model`:
# * A `Flatten()` layer which reshapes the outputs to a single channel.
# * A fully-connected layer with 256 output units and `relu` activation
# * A `Dropout(0.5)` layer.
# * A final fully-connected layer.  Since this is a binary classification, there should be one output and `sigmoid` activation.

# %%
# TODO 7
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# %% [markdown]
# Print the model summary.  This will display the number of trainable parameters vs. the non-trainable parameters.

# %%
# TODO 8
model.summary()

# %% [markdown]
# ## Using Generators to Load Data
# 
# Up to now, the training data has been represented in a large matrix.  This is not possible for image data when the datasets are very large.  For these applications, the `keras` package provides a `ImageDataGenerator` class that can fetch images on the fly from a directory of images.  Using multi-threading, training can be performed on one mini-batch while the image reader can read files for the next mini-batch. The code below creates an `ImageDataGenerator` for the training data.  In addition to the reading the files, the `ImageDataGenerator` creates random deformations of the image to expand the total dataset size.  When the training data is limited, using data augmentation is very important.

# %%
train_data_dir = './train'
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='binary')

# %% [markdown]
# Now, create a similar `test_generator` for the test data.

# %%
# TODO 9
test_data_dir = './test'
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_generator = test_datagen.flow_from_directory(
                        test_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='binary')

# %% [markdown]
# The following function displays images that will be useful below.

# %%
# Display the image
def disp_image(im):
    if (len(im.shape) == 2):
        # Gray scale image
        plt.imshow(im, cmap='gray')    
    else:
        # Color image.  
        im1 = (im-np.min(im))/(np.max(im)-np.min(im))*255
        im1 = im1.astype(np.uint8)
        plt.imshow(im1)    
        
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

# %% [markdown]
# To see how the `train_generator` works, use the `train_generator.next()` method to get a minibatch of data `X,y`.  Display the first 8 images in this mini-batch and label the image with the class label.  You should see that bicycles have `y=0` and cars have `y=1`.

# %%
# TODO 10
plt.figure(figsize=(10,10))
nplot = 8
X_batch, y_batch = train_generator.next()
for i in range(nplot):    
    plt.subplot(1,nplot,i+1)
    disp_image(X_batch[i,:,:,:])
    plt.xlabel(y_batch[i])

# %% [markdown]
# ## Train the Model
# 
# Compile the model.  Select the correct `loss` function, `optimizer` and `metrics`.  Remember that we are performing binary classification.

# %%
# TODO 11
from tensorflow.keras.optimizers import Adam
opt = Adam(lr=1e-3)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# When using an `ImageDataGenerator`, we have to set two parameters manually:
# * `steps_per_epoch =  training data size // batch_size`
# * `validation_steps =  test data size // batch_size`
# 
# We can obtain the training and test data size from `train_generator.n` and `test_generator.n`, respectively.

# %%
# TODO 12
steps_per_epoch = train_generator.n // batch_size
validation_steps = test_generator.n // batch_size

# %% [markdown]
# Now, we run the fit.  If you are using a CPU on a regular laptop, each epoch will take about 3-4 minutes, so you should be able to finish 5 epochs or so within 20 minutes.  On a reasonable GPU, even with the larger images, it will take about 10 seconds per epoch.
# * If you use `(nrow,ncol) = (64,64)` images, you should get around 90% accuracy after 5 epochs.
# * If you use `(nrow,ncol) = (150,150)` images, you should get around 96% accuracy after 5 epochs.  But, this will need a GPU.
# 
# You will get full credit for either version.  With more epochs, you may get slightly higher, but you will have to play with the damping.
# 
# Remember to record the history of the fit, so that you can plot the training and validation accuracy curve.

# %%
import keras
keras.__version__

# %%
nepochs = 5  # Number of epochs

# Call the fit_generator function
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nepochs,
    validation_data=test_generator,
    validation_steps=validation_steps,
    use_multiprocessing = False)

# %%
# Plot the training accuracy and validation accuracy curves on the same figure.

# TODO 13
tr_accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']

plt.plot(tr_accuracy)
plt.plot(val_accuracy)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('accuarcy')
plt.legend(['training accuracy', 'validation accuracy'])

# %% [markdown]
# ## Plotting the Error Images
# 
# Now try to plot some images that were in error:
# 
# *  Generate a mini-batch `Xts,yts` from the `test_generator.next()` method
# *  Get the class probabilities using the `model.predict( )` method and compute predicted labels `yhat`.
# *  Get the images where `yts[i] ~= yhat[i]`.
# *  If you did not get any prediction error in one minibatch, run it multiple times.
# *  After you a get a few error images (say 4-8), plot the error images with the true labels and class probabilities predicted by the classifie

# %%
# TODO 14
# Xts_batch, yts_batch = test_generator.next()
# yhat_batch = model.predict(Xts_batch)
# acc = np.mean(yhat_batch == yts_batch)
# print(acc)
# error_index = np.where(yhat_batch != yts_batch)

# nplot = len(error_index)
# plt.figure(figsize=(10,10))
# for i in error_index: 
#     plt.subplot(1,nplot,i+1)
#     disp_image(Xts_batch[i,:,:,:])
#     plt.xlabel(yts_batch[i] + )

# %%



