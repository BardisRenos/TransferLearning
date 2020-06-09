# An example of Deep Learning by using Transfer Learning in Keras.


<p align="center"> 
<img src="https://github.com/BardisRenos/TransferLearning/blob/master/transferLearning.jfif" width="450" height="250">
</p>



This repository consists the parts:

1. What is transfer learning ?
2. Why to use a transfer learnign model ?
3. How can build a transfer learning model ?


## What is transfer learning ?
Transfer learning is another approach that we can use to classify images. Transfer learning models are trained on one problem and applying to another one. Therefore, I use a pre trained model with weights. In this case I use VGG16. This model have been trained how to detect generic features from pictures more than 1 million of images and trained to 1000 classes. 

<p align="center"> 
<img src="https://github.com/BardisRenos/TransferLearning/blob/master/vgg16.jpg" width="450" height="250">
</p>


## Why to use transfer learning.
Transfer learning, can be very useful because Deep convolutional neural network models may take days or even weeks to train on very large datasets. Also, produces better results than to use a non pretrained model in less time. The advantages are:
    
   1. There is no need of an extremely large training dataset.
   2. Not much computational power is required. As we are using pre-trained weights and only have to learn the weights of the last few          layers.

## How can build a transfer learning?
From the model, I can change the input layer size and the last layer, the classification layer. The input layer changes from 224 by 224 pixels to 250 by 250 pixels in RGB image. Also, not including the classification part.

## Retrieve the data set

In our case we have a dataset that consists two classes (dogs and cats). However, the dataset is not ready to use in order to feed a convolutional neural network. Therefore, is needed to reshape those images into proper structure.   

```python

def create_training_data() -> list:
    # Here we set the path of the dataset.
    DATALOCATION = 'C:\\path\\of\\the\\file'
    # The categories of the data set. Are two only Cats and dogs. Hence, Dog is category 0 and the Cat is 1
    CATEGORIES = ["Dog", "Cat"]
    training_data = []
    for categories in CATEGORIES:
        path = os.path.join(DATALOCATION, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

    return training_data
```



```python
# The size that we want to reformat all the images in the same dimensions
 IMG_SIZE = 250
 vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
```

The other part is to add how many full connected layers with the number of neurons accordingly. Finally, the number of classes of the data that need to be classify. In this case, only 2 classes. 

```python
    # Full connected layer with 1024 neurons
    x = Dense(1024, activation='relu')(x)  # dense layer 1
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(4096, activation='relu')(x)  # dense layer 3
    # We change the output leyar to 2 classes from 1000 classes before
    output = Dense(2, activation='softmax')(x)
```


