<img src="https://github.com/sarmitamajumdar/Alternatives/blob/main/alter-png22.png" width="280"/>

# Alternatives
> Think on Zero Hunger, Focus on Nutrition

The exisisting, focused agricultural systems are busy in producing staple grains rather than producing diverse and healthier foods. The pandemic along with critical climate emergency makes us to re-think food systems. Spirulina, the blue green algae, with it's immense source of protein, essential nutrients, could fulfil the dietary needs and help protect  malnutrition  of billions of people in our over populated planet. It could be an alternative to meat or soya farming. Farming of spirulina can protect fragile ecosystems by reducing Carbon emission, treat Nuclear waste, and can produce renewable bio-fuel to produce the global energy in future.
Alternatives - an informational chatbot. Used NLP & Keras to create a 3 layer neuron, to generate output.

#pip install python3.7 

## Developing

Self taught developer on Machine Learning, deep learning. Create a conversational chatbot project.  Prior experiences in training C Data structure, Python 3. Some experience as Product Manager in a Software company for their ERP modules. Last year 2020 Call for code, submitted project "Project-Grip". It was a Dashboard on  Covid 19. Mainly on Indian scenario.

### DATA & File overview

#intents.json :  Contains predefined patterns and responses.
#train_chatbot.py : A Python file, contains script to build the model and train the Chatbot.
#Words.pkl  : A pickle file used to store the words that python object, contains a list of vocabulary.
#classes.pkl : This file contains the list of categories.
#chatbot_model.h5 – This is the trained model that contains information about the model that has Weights of the neurons.
#chatgui.py: This is the Python script in which we implemented GUI for the Chatbot.

## Features
* Provide information on future food, Spirulina
* Health benefits
* Farming

## Methodological information
##### nltk :NLP toolkit to analyze text.
##### json : To store, transmit data objects of attribute-value pairs and arrays.
##### pickle: To convert objects into character stream.
##### Keras: To Define, Compile, Fit & make Predictions.
##### Training & Testing data : To create an empty array, initialize the bag of words array with 1 if matched in pattern or 0 if false and suffle features to turn into np.array.
##### Model building: Used deep neural networks with 3 layers- first layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons which is equal to number of intents to Predict output intent with softmax( a mathematical function that converts a vector of numbers into a vector of probabilities).
##### Compile the model: Stochastic Gradient Descent(SGD), to minimize Loss & Optimize accuracy.
##### Fit and save: Fit the model and save it into chatbot_model.h5 file.
##### Tkinter : Finally used Tkinter library too develop a Graphical User Interface.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Links

- Project homepage: https://sarmitamajumdar.github.com/Alternatives/
- Repository: https://github.com/sarmitamajumdarr/Alternatives/

## Licensing

