import keras
import numpy as np
import cv2
import shutil
from keras.models import load_model
from scan_lib import scan_data as sc


def data_generator_declare():
    print('Все файлы должны быть в формате - img999.jpg. ' +
          'Формат может быть pdf, jpg, png. Каждому скану ' +
          'должна соответствовать фотография документа.')
    print('Программа сохранит данные в файлы с форматом .h5.')
    masks_dir = input('Введите относительный путь' +
                       ' к папке со сканами документов: ')
    images_dir = input('Введите относительный путь к папке с фотографиями документов: ')
    return sc.DataGen(images_dir, masks_dir), sc.DataGen(images_dir, masks_dir, True)


n = 0
dataGenTrain, dataGenVal = data_generator_declare()


def read_model(who):
    if who == 0:
        model = load_model('my_model.h5')
        return model
    elif who == 1:
        return load_model('scan_lib//my_model_main_1.h5')
    elif who == 2:
        return load_model('scan_lib//my_model_main_2.h5')
    elif who == 3:
        return load_model('scan_lib//my_model_main_3.h5')


def trening_model(who, eter):
    global n
    n = who
    model = read_model(who)
    best_w = keras.callbacks.ModelCheckpoint('resnet_best_' + str(n) + '.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='auto',
                                             period=1)

    last_w = keras.callbacks.ModelCheckpoint('resnet_last_' + str(n) + '.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=False,
                                             save_weights_only=True,
                                             mode='auto',
                                             period=1)

    callbacks = [best_w, last_w]

    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(adam, 'binary_crossentropy')
    batch_size = 8

    model.fit_generator(keras_generator(dataGenTrain, batch_size),
                        steps_per_epoch=eter,
                        epochs=1,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=keras_generator(dataGenVal, batch_size),
                        validation_steps=50,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0)

    return model


def delete_model():
    shutil.copy2('scan_lib//my_model2.h5', 'scan_lib//my_model.h5')


def execute_model(vely, model):
    return model.predict(vely)


def save_model(model):
    global n
    if n == 0:
        model.save('scan_lib//my_model.h5')
        model.save('scan_lib//my_model_main_1.h5')
        model.save('scan_lib//my_model_main_2.h5')
        model.save('scan_lib//my_model_main_3.h5')
    elif n == 1:
        return load_model('scan_lib//my_model_main_1.h5')
    elif n == 2:
        return load_model('scan_lib//my_model_main_2.h5')
    elif n == 3:
        return load_model('scan_lib//my_model_main_3.h5')


def keras_generator(data_gen, batch_size):
    while True:
        x_batch = []
        y_batch = []

        global n
        for i in range(batch_size):

            img, mask = data_gen.generate(n)

            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, y_batch
