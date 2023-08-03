#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
import json 
import numpy as np
import pickle

stemmer = PorterStemmer()
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
#intents= train_data_file.intents
words = []
classes = []
pattern_word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
# function for appending stem words

def get_stem_words(words,ignore_words):
    stem_words = []
    for word in words :
        if word not in ignore_words :
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words 


#print(stem_words)
#print(pattern_word_tags_list[0]) 
#print(classes)          

         

#Create word corpus for chatbot

def create_bot_corpus(words,classes,pattern_word_tags_list,ignore_words):
    for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            pattern_word_tags_list.append((pattern_word, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    stem_words = get_stem_words(words, ignore_words)

    stem_words= sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words,classes,pattern_word_tags_list

def bag_of_words_encoding(stem_words,pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list :
        pattern_words = word_tags[0]
        bag_of_words = []
        stem_pattern_words = get_stem_words(pattern_words,ignore_words)
        for word in stem_words :
            if word in stem_pattern_words:
                bag_of_words.append(1)
            else :
                bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
     labels = []
     for word_tags in pattern_word_tags_list:
         labels_encoding = list([0]*len(classes))
         tag = word_tags[1]
         tag_index = classes.index(tag) 
         labels_encoding[tag_index] = 1 
         labels.append(labels_encoding) 
     return np.array(labels)  

def preprocess_train_data():  
    stem_words,tag_classes,word_tags_list = create_bot_corpus(words,classes,pattern_word_tags_list,ignore_words)
    pickle.dump(stem_words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))
    train_x = bag_of_words_encoding(stem_words,word_tags_list)
    train_y = class_label_encoding(tag_classes,word_tags_list)
    return train_x,train_y

train_x,train_y =preprocess_train_data()
print(train_x[0])
print(train_y[0])





