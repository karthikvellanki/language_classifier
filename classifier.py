import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA = "data.json"
NUMBER_OF_LANGUAGES = 12


def get_data(json_data):

    with open(json_data, "r") as input_data:
        data = json.load(input_data)

    inputs = np.array(data["mfcc"])
    results = np.array(data["labels"])

    return inputs, results


if __name__ == "__main__":

    # get data
    X, y = get_data(DATA)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # build neural network
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.25),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.25),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.25),

        # output layer
        keras.layers.Dense(NUMBER_OF_LANGUAGES, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100)
