# ANN algorithms
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

# 20 layers ann model:
def ann_model(x_train):
    import random
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    random.seed(42)

    model = Sequential([
      Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(1), ])

    return model

# 30 layers ann model:
def ann_model1(x_train):
    import random
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    random.seed(42)

    model = Sequential([
      Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(256, activation='relu'),
      Dense(1), ])

    return model


# 30 layers ann model:
def ann_model2(x_train):
    import random
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    random.seed(42)

    nn = 25
    activation_function = 'selu'

    model = Sequential([
      Dense(nn, activation=activation_function, input_shape=(x_train.shape[1],)),
      # Dense(nn, activation='relu', input_shape=(1,)),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(nn, activation=activation_function),
      Dense(1), ])

    return model