"""С применением функционального подхода производится конструирование модели,
аналогичной по архитектуре модели MobileNet, из базовых слоев keras.layers.
Дефолтные значения аргументов функции создают модель на 1000 классов,
принимающую на вход цветные изображения размером 224 х 224 пикселя.
"""

import tensorflow as tf

# Проверка доступных GPU:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs available: {len(physical_devices)}')

# Если есть GPU:
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_block(input_data, n_neurons: int, strides: tuple):
    """Функция формирует блок из двух слоев convolution
    3 x 3 и 1 x 1 с последующей нормализацией и активацией.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - количество нейронов в слоях блока
        strides - сдвиг "окна" в слоях depthwise convolution
    """
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(input_data)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n_neurons, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def create_model(input_size=224, n_classes=1000):
    """Функция формирует и инициализирует модель.
    Аргументы:
        input_size - размер изображения в пикселях
        n_classes - количество классов (нейронов в финальном слое)
    """
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))

    n_neurons = 32
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (0, 1)))(inputs)
    x = tf.keras.layers.Conv2D(filters=n_neurons, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Повторяющиеся блоки модели
    for block in range(1, 13):
        # Шаг сдвига "окна" depthwise convolution в зависимости от блока
        if block in (2, 4, 6, 12, 13):
            strides = (2, 2)
        else:
            strides = (1, 1)
        # Удвоение количества нейронов
        if block in (1, 2, 4, 6, 11):
            n_neurons *= 2
        # Блок из 2 слоев convolution
        x = create_block(x, n_neurons=n_neurons, strides=strides)

    # Финальные слои:
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1024))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.Reshape((1000,))(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = create_model()
model.summary()
