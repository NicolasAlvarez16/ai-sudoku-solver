from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class SudokuNet:
    @staticmethod # No consturctor
    def build(width, height, depth, classes):
        '''
            width: pixesl
            height: pixesl
            depht: color scale
            classes: The number of digits
        '''
        # Initialise the model
        model = Sequential()
        input_shape = (width, height, depth)

        # First se of Conv
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second set of Conv
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # First set of fullu connected layer -> Set with 50% dropout
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Second set of fullu connected layer -> Set with 50% dropout
        model.add(Dense(64))    
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model