import numpy as np 
import pandas as pd
import string
import matplotlib.pyplot as plt

from pickle import load
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import  img_to_array

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================== jupyter ===========================

def load_document(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_descriptions(filename):
    
    description = list()
    
    for line in filename.split('\n'):
        if line != '':
            tokens = line.split('\t')
            img, desc = tokens[0], tokens[1]
            inde = img.split('#')[1]
            img = img.split('.')[0]
            description.append([img]+[inde]+[desc]) 
            
    discription_dataframe = pd.DataFrame(description,columns=["filename","index","caption"])
    
    return discription_dataframe


def random_3_images(images_dir, dataframe):
    n_pix=299
    quantity = 3
    target_size = (n_pix, n_pix, 3)
    count = 1
    uniq = np.unique(dataframe['filename'])
    IMGS =  np.random.choice(uniq, size=quantity)
    fig = plt.figure(figsize=(10,15))
    for IMG in IMGS:
        filename = images_dir + IMG + '.jpg'
        captions = list(dataframe['caption'].loc[dataframe['filename']==IMG].values)
        IMG_load = load_img(filename, target_size=target_size)
        
        ax = fig.add_subplot(quantity, 2, count, xticks=[], yticks=[])
        ax.imshow(IMG_load)
        count+=1
        
        ax = fig.add_subplot(quantity, 2, count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(-1,len(captions))
        for i, caption in enumerate(captions):
            ax.text(0,i,caption,fontsize=10)
        count += 1
    plt.show()
    

def description_cleaning(descriptions):
    
    for index, text in enumerate(descriptions):
        text = descriptions[index]
        text = text.split()
        text = [word.lower() for word in text]
        text = [word.translate(str.maketrans('','',string.punctuation)) for word in text]
        text = [word for word in text if len(word)>1]
        text = [word for word in text if word.isalpha()]
        descriptions[index] = ' '.join(text)
        

def count_words(dataframe):

    vocabulary = []
    for i in range(len(dataframe)):
        temp=dataframe.iloc[i,2]
        vocabulary.extend(temp.split())

    counter = Counter(vocabulary)
    counter_dict = pd.DataFrame({"word":list(counter.keys()), "count":list(counter.values())})
    counter_dict = counter_dict.sort_values("count", ascending=False) \
                               .reset_index()[['word', 'count']]

    return counter_dict


def plot_freq_words(words_count, title):
    
    plt.figure(figsize=(20,5))
    plt.bar(words_count.index, words_count['count'], color='#3300cc')
    plt.xticks(words_count.index, words_count['word'], rotation=60)
    plt.title(title, fontsize=15)
    plt.show()


def load_set(filename):
    
#     document = load_document(filename)
    document= open(filename).read()
    data = list()
    for line in document.split('\n'):
        if line != '':
            jpg = line.split('.')[0]
            data.append(jpg)
    return set(data)


def load_clean_descriptions(filename, dataset):
    
    doc = open(filename).read()
    description = dict()
    
    for line in doc.split('\n'):
        tokens = line.split()
        img_id, img_desc = tokens[0], tokens[1:]        
        if img_id in dataset:
            if img_id not in description:
                description[img_id] = list()
            desc = 'startseq ' + ' '.join(img_desc) + ' endseq'
            description[img_id].append(desc)
            
    return description


def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    all_descriptions = []
    for key in descriptions.keys():
        [all_descriptions.append(d) for d in descriptions[key]]
    return all_descriptions


def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def generate_description(model, tokenizer, photo_feature, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endsec':
            break
    return in_text


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def clean_summary(summary):
    idx = summary.find('startseq ')
    if idx > -1:
        summary = summary[len('startseq '):]
    idx = summary.find(' endseq')
    if idx > -1:
        summary = summary[:idx]
    return summary


def plot_pred(predictions):
    

    def make_sentence(caption_sentence):
        sentence = ''
        for c in caption_sentence:
            sentence += ' ' + c
        return sentence 
    
    
    target_size = (299,299,3)    
    count = 1
    
    fig = plt.figure(figsize=(10,20))
    npic = len(predictions)
    
    for p in predictions:
        bleu, key, _, yhat = p
        filename = image_dir + key + '.jpg'
        image = load_img(filename, target_size=target_size)
        
        ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
        ax.imshow(image)
        count += 1
        
        caption = make_sentence(yhat)
        
        ax = fig.add_subplot(npic,2,count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.text(0,0.6,"Predicted:   {}".format(caption), fontsize=20)
        ax.text(0,0.4,"BLEU:    {}".format(bleu), fontsize=20)
        count += 1
        
    plt.show()
    
# =========================== tkinter ===========================

def extract_image_feature(filename):
    
    model = InceptionV3()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    image = load_img(filename, target_size=(299,299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)

    return feature