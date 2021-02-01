# Language Classifier

This project is a machine learning classifier that classifies audio clips based on the language. The neural network is made up of 3 layers and uses a dropout rate of 0.2.

preprocess.py extracts the MFCCs from the audio clips and stores it in a json file.

classifier.py loads the json file and uses the MFCCs as the input for the classifier.

The classifier assumes the dataset consists of 12 languages. This can be changed by modifying the NUMBER_OF_LANGUAGES variable.
