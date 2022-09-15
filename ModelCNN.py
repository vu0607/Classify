from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,\
     Dropout, Flatten, Dense, Activation,\
     BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


def define_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def call_back():
    early_stop = EarlyStopping(
         monitor='val_accuracy',
         patience=2,
         restore_best_weights=True
    )

    learning_rate_reduction = ReduceLROnPlateau(
         monitor='val_accuracy',
         patience=2,
         verbose=1,
         factor=0.5,
         min_lr=0.00001)

    callbacks = [early_stop, learning_rate_reduction]
    return callbacks

# if __name__ == '__main__':
#     define_model()