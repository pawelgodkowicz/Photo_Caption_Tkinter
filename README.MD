# Photo caption using deep learning and tkinter.

This is my mini project about Image captioning.

The goal of the project was to create a GUI using a tkinter to automatically predict the description of a given image.

### Main utilities
- python 3.7.6 
- tensorflow-gpu
- tkinter
- InceptionV3

### Dataset
Flickr8k dataset downloaded direct from [Jason Brownlee dataset GitHub repository](https://github.com/jbrownlee/Datasets):
- [Flickr8k_Dataset.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
- [Flickr8k_text.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

The dataset has a pre-defined training dataset (6000 images), validation dataset (1000 images), and test dataset (1000 images).

### Example 

> Program start
>><img src="https://github.com/pawelgodkowicz/image_caption/blob/master/img_readme/01.png?raw=true" width="332" height="190">

> Loading weights
>><img src="https://github.com/pawelgodkowicz/image_caption/blob/master/img_readme/02.png?raw=true" width="363" height="231">

> Loading image
>><img src="https://github.com/pawelgodkowicz/image_caption/blob/master/img_readme/03.png?raw=true" width="495" height="462">

> Prediction 
>><img src="https://github.com/pawelgodkowicz/image_caption/blob/master/img_readme/04.png?raw=true" width="783" height="463">


### Sources:

- Dogs and humans pictures downloaded from [www.surfcityusa.com ](https://www.surfcityusa.com/things-to-do/beaches/dog-beach/), [www.witreader.com ](https://witreader.com/articles/841395073709/) respectively.

-[Jason Brownlee - Machine Learning Mastery](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
