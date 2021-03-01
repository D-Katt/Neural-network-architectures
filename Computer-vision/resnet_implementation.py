"""С применением функционального подхода производится конструирование моделей,
аналогичных по архитектуре моделям семейства ResNet, из базовых слоев keras.layers.
Дефолтные значения аргументов функции создают модель на 1000 классов,
принимающую на вход цветные изображения размером 224 х 224 пикселя.
Один из аргументов функции принимает кортеж значений, определяющих число блоков
внутри сегментов, что позволяет воспроизводить модели ResNet50, ResNet101 и ResNet152.
"""

import tensorflow as tf
from collections import namedtuple

# Проверка доступных GPU:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs available: {len(physical_devices)}')

# Если есть GPU:
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_identity_block(input_data, n_neurons: list, window: tuple):
    """Функция формирует блок, состоящий из нескольких последовательных
    слоев convolution и включающий механизм skip connection,
    благодаря которому исходные данные складываются
    с преобразованными данными перед финальной активацией.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - список с количеством нейронов в слоях блока
        window - размер "окна" в слоях convolution
    """
    # Количество нейронов (фильтров) для разных этапов внутри блока
    neurons_1, neurons_2, neurons_3 = n_neurons
    # Этап 1
    x = tf.keras.layers.Conv2D(filters=neurons_1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_data)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Этап 2
    x = tf.keras.layers.Conv2D(filters=neurons_2, kernel_size=window, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Этап 3
    x = tf.keras.layers.Conv2D(filters=neurons_3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Сложение преобразованного и исходного тенсоров
    x = tf.keras.layers.Add()([x, input_data])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def create_conv_block(input_data, n_neurons: list, window: tuple, strides: tuple):
    """Функция формирует блок, состоящий из нескольких последовательных
    слоев convolution и включающий механизм skip connection,
    дополненный конволюцией исходных данных. Перед финальной активацией
    складываются однократно и трехратно преобразованные исходные данные.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - список с количеством нейронов в слоях блока
        window - размер "окна" в слоях convolution
        strides - шаг при перемещении "окна"
    """
    # Количество нейронов (фильтров) для разных этапов внутри блока
    neurons_1, neurons_2, neurons_3 = n_neurons
    # Однократное преобразование исходных данных
    modified_input = tf.keras.layers.Conv2D(filters=neurons_3, kernel_size=(1, 1),
                                            strides=strides, padding='valid')(input_data)
    modified_input = tf.keras.layers.BatchNormalization(axis=3)(modified_input)
    # Трехкратное преобразование исходных данных
    # Этап 1
    x = tf.keras.layers.Conv2D(filters=neurons_1, kernel_size=(1, 1), strides=strides, padding='valid')(input_data)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Этап 2
    x = tf.keras.layers.Conv2D(filters=neurons_2, kernel_size=window, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Этап 3
    x = tf.keras.layers.Conv2D(filters=neurons_3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Сложение однократно и трехкратно преобразованного исходного тенсора
    x = tf.keras.layers.Add()([x, modified_input])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def create_model(n_id_blocks: namedtuple, input_size=224, n_classes=1000):
    """Функция формирует и инициализирует модель.
    Аргументы:
        n_id_blocks - количество identity-блоков во 2-5 частях модели
        input_size - размер изображения в пикселях
        n_classes - количество классов (нейронов в финальном слое)
        """
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))  # Входной слой
    # Сегмент 1 (отличается от остальных)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Сегмент 2
    x = create_conv_block(x, n_neurons=[64, 64, 256], window=(3, 3), strides=(1, 1))
    for _ in range(n_id_blocks.segment_2):
        x = create_identity_block(x, n_neurons=[64, 64, 256], window=(3, 3))
    # Сегмент 3
    x = create_conv_block(x, n_neurons=[128, 128, 512], window=(3, 3), strides=(2, 2))
    for _ in range(n_id_blocks.segment_3):
        x = create_identity_block(x, n_neurons=[128, 128, 512], window=(3, 3))
    # Сегмент 4
    x = create_conv_block(x, n_neurons=[256, 256, 1024], window=(3, 3), strides=(2, 2))
    for _ in range(n_id_blocks.segment_4):
        x = create_identity_block(x, n_neurons=[256, 256, 1024], window=(3, 3))
    # Сегмент 5
    x = create_conv_block(x, n_neurons=[512, 512, 2048], window=(3, 3), strides=(2, 2))
    for _ in range(n_id_blocks.segment_5):
        x = create_identity_block(x, n_neurons=[512, 512, 2048], window=(3, 3))
    # Финальный пулинг:
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Финальный слой по количеству классов:
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# Именованный кортеж для определения количества блоков по сегментам модели
Blocks = namedtuple('Segments', 'segment_2 segment_3 segment_4 segment_5')

# Количество identity-блоков по сегментам модели для ResNet50
id_blocks = Blocks(segment_2=2, segment_3=3, segment_4=5, segment_5=2)
model = create_model(n_id_blocks=id_blocks)
print('ResNet50')
model.summary()

# Количество identity-блоков по сегментам модели для ResNet101
id_blocks = Blocks(segment_2=2, segment_3=3, segment_4=22, segment_5=2)
model = create_model(n_id_blocks=id_blocks)
print('ResNet101')
model.summary()

# Количество identity-блоков по сегментам модели для ResNet152
id_blocks = Blocks(segment_2=2, segment_3=7, segment_4=35, segment_5=2)
model = create_model(n_id_blocks=id_blocks)
print('ResNet152')
model.summary()
