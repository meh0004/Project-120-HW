# Text Data Preprocessing Lib
import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np
import tensorflow 
import random 
 
words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
#list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
from data_preprocessing import get_stem_words

words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))
model = tensorflow.keras.models.load_model("chatbot.h5")

def preprocessing(user_input):
   
    bag = []
    bag_of_words = []
    input_words_10 = nltk.word_tokenize(user_input)
    input_words_20 = get_stem_words(input_words_10,ignore_words)
    input_word_2 = sorted(list(set(input_words_20)))
    for i in words:
        if i in input_word_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def preprocessing(user_input):
    input_word_1 = nltk.word_tokenize(user_input)
    input_word_2 = get_stem_words(input_word_1,ignore_words)
    input_word_2 = sorted(list(set(input_word_2)))
    bag = []
    bag_of_words = []
    for i in words:
        if i in input_word_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_response(user_input):
    predict_class_label = bot_class_prediction(user_input)
    predicted_class = classes[ predict_class_label]
    for i in intents['intents']:
        if i['tag'] == predicted_class:
            bot_response = random.choice(i['responses'])
          
print("Hi I am Daisy, how can I help?")
while True:
    user_input = input("Enter the text here...")
    response = bot_response(user_input)
    print("Bot Response: ", response)
        
    
            
 

   
    