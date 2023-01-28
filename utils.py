import math

import pandas as pd
from keras import Sequential


def cross_validation(x_data: pd.DataFrame, y_data: pd.DataFrame,  model: Sequential, k: int, epoch: int):
    size_sample = math.ceil(x_data.shape[0] / k)
    history = list()

    for i in range(k):
        x_test = x_data[i*size_sample:(i+1)*size_sample]
        y_test = y_data[i*size_sample:(i+1)*size_sample]

        x_train = pd.concat([x_data[0:i*size_sample], x_data[(i+1)*size_sample:]])
        y_train = pd.concat([y_data[0:i * size_sample], y_data[(i + 1) * size_sample:]])

        model.fit(
            x_train,
            y_train,
            epochs=epoch,
            batch_size=64,
            validation_data=(x_test, y_test),
            verbose=0
        )

        history.append(model.evaluate(x_test, y_test)) # [[loss, metrics]]

    return history
