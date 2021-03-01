"""С применением функционального подхода производится конструирование моделей,
аналогичных по архитектуре моделям семейства VGG, из базовых слоев keras.layers.
Исходные модели была загружены через keras.applications. Через метод summary()
были получены общие сведения о наборе и последовательности слоев.
Дефолтные значения аргументов функции создают модель на 1000 классов,
принимающую на вход цветные изображения размером 224 х 224 пикселя.
Размеры "окна" для слоев convolution - 3 х 3, pooling - 2 х 2.
Функция принимает аргумент, определяющий максимальное количество слоев convolution
в блоках модели, что позволяет воспроизводить архитектуру VGG16 и VGG19.
"""

import tensorflow as tf

# Проверка доступных GPU:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs available: {len(physical_devices)}')

# Если есть GPU:
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_block(input_data, n_neurons: int, n_conv=2):
    """Функция формирует блок, состоящий из нескольких слоев
    convolution и финального слоя pooling.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - количество нейронов в слоях блока
        n_conv - количество слоев convolution внутри блока
    """
    x = tf.keras.layers.Conv2D(n_neurons, 3, padding='same', activation='relu')(input_data)
    for conv_layer in range(n_conv - 1):
        x = tf.keras.layers.Conv2D(n_neurons, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    return x


def create_model(max_conv: int, input_size=224, n_classes=1000):
    """Функция формирует и инициализирует модель.
    Аргументы:
        max_conv - максимальное количество слоев convolution в блоках
        input_size - размер изображения в пикселях
        n_classes - количество классов (нейронов в финальном слое)
        """
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))  # Входной слой
    # Блоки внутри модели:
    x = create_block(inputs, 64)  # Блок 1
    x = create_block(x, 128)  # Блок 2
    x = create_block(x, 256, n_conv=max_conv)  # Блок 3
    x = create_block(x, 512, n_conv=max_conv)  # Блок 4
    x = create_block(x, 512, n_conv=max_conv)  # Блок 5
    # Преобразование данных в одномерный массив:
    x = tf.keras.layers.Flatten()(x)
    # Полносвязные слои:
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    # Финальный слой по количеству классов:
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# Модель, аналогичная VGG16
model = create_model(max_conv=3)
print('VGG16')
model.summary()

# Модель, аналогичная VGG19
model = create_model(max_conv=4)
print('VGG19')
model.summary()
