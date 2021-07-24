# importing libraries for natural language process
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# importing libraries to define and train neural network models
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sorting classes
classes = sorted(list(set(classes)))

# documents is the combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

#pickling- to convert python object hierarchy in byte stream
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# creating the training data
training = []
# creating an empty array for our output
out_empty = [0] * len(classes)
# training set and bag of words for each sentence
for doc in documents:
    # initializing the bag of words
    bag = []
    # listing the tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, an attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # creating the bag of words array with 1, if the pattern of word matches
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag
    output_row = list(out_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# creating the train - test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created!")

# Create model of - 3 layers. First layer 128 neurons, second layer 64 neurons,
# and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) # ReLU- very high-performance networks.This function takes a single number
# as an input, returning 0 if the input is negative, and 1 if the input is positive.
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # softmax - is a mathematical function that converts
# a vector of numbers into a vector of probabilities

# Compile model. Stochastic Gradient Descent with Nesterov - extension of momentum that involves calculating
# the decaying moving average of the gradients accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# categorical_crossentropy-used as a loss function for multi-class classification model where
# there are two or more output labels
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=8, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model created")
