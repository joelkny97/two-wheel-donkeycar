""""

keras.py

functions to run and train autopilots using keras

"""

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D, Convolution3D, Cropping3D,  MaxPooling3D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist


class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_linear()
        else:
            self.model = default_linear()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    # Convolution2D class name is an alias for Conv2D
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5},
                    metrics=['mse','mae'])
    model.summary()
    return model
    
def cnn_3d():
    #3DCNN base model
    img_in3D = Input(shape=(3, 120, 160, 3), name='img_in') 
    x = img_in3D 
    x = Cropping3D(cropping = ((0, 0),(60, 0),(0, 0)))(x) 
    x = Convolution3D(8, (3, 3, 3), strides=(1, 2, 2), activation='relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x) 
    x = BatchNormalization()(x) 
    x = Dropout(0.1)(x)  
    x = Flatten(name='flattened')(x) 
    x = Dense(50, activation='relu')(x) 
    x = Dropout(0.2)(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)

    model = Model(inputs=[img_in3D], outputs=[angle_out, throttle_out]) 
    model.compile(optimizer='adam', loss={'angle_out':'categorical_crossentropy',
                                              'throttle_out': 'mean_absolute_error'},    loss_weights={'angle_out': 0.9, 'throttle_out': 0.01})
    model.summary()
    return model
