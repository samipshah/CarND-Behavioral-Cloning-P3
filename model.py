from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, AveragePooling2D
from keras.layers import Cropping2D
from keras.backend import tensorflow_backend as tf

import read_data
import matplotlib.pyplot as plt

def nvidia_model():
    model = Sequential()
    # cropping image 50 pixels from top 20 pixels from bottom
    model.add(Cropping2D(cropping=((22, 13), (0, 0)), input_shape=(66, 200, 3)))
    # normalization to have 0 mean values
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    # convolution layer extracting 24 features from 5x5x3 image
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    # max pooling in each feature layer to move towards most important
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # another convolution layer
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Flatten())
    # gradual reduction in dimensions in fully connected layer
    model.add(Dense(1164))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# model inspired from nvidia paper
def own_model():
    model = Sequential()
    # cropping image 50 pixels from top 20 pixels from bottom
    model.add(Cropping2D(cropping=((20, 12), (0, 0)), input_shape=(64, 64, 3)))
    # making them small
    # model.add(Lambda(lambda x: tf.resize_images(x, 64, 64, "channels_last")))
    # normalization to have 0 mean values
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    # convolution layer extracting 24 features from 5x5x3 image
    model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    # dropouts added to avoid overfitting
    model.add(Dropout(0.2))
    # max pooling in each feature layer to move towards most important
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # another convolution layer
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # due to fewer dimension avoided pooling layer
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    # gradual reduction in dimensions in fully connected layer
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def lenet_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((20, 12), (0, 0)), input_shape=(64, 64, 3)))
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def basic_model():
    model = Sequential()
    # should we crop the image before Lambda ?
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def model_train(debug=False):
    batch_size = 50
    training, validation = read_data.get_samples()
    if debug is True:
        model = load_model("model.h5")
    else:
        model = nvidia_model()
    for n in range(1):
        train_generator = read_data.generator(training, batch_size=batch_size,
                                              threshold=0.2)
        valid_generator = read_data.generator(validation, batch_size=batch_size, threshold=0.0)
        history_object = model.fit_generator(train_generator, (len(training)/batch_size), \
            epochs=3, verbose=1, validation_data=valid_generator, \
            validation_steps=(len(validation)/batch_size))

    model.save("model.h5")
    json_string = model.to_json()
    with open("model.json", "w") as f:
        f.write(json_string)
    # print(history_object.history.keys())
    # print(history_object.history['loss'])
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()

if __name__ == "__main__":
    model_train(debug=False)
