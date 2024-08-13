import numpy as np
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(LSTM(hp.Int(f'units_in_lstm_layer_{i}',min_value=32,max_value=128,step=32),return_sequences=True))
        # model.add(Dropout(hp.Float(f'rate_in_dropout_layer_{i}',min_value=0.1,max_value=0.5,step=0.1)))
    model.add(LSTM(hp.Int('units_int_last_lstm_layer',min_value=32,max_value=128,step=32)))
    model.add(Dropout(hp.Float('rate_in_last_dropout_layer',min_value=0.1,max_value=0.5,step=0.1)))
    model.add(Dense(y_train.shape[1], activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

list_id_forecasting_moment = ['2020-04-25 00:00:00_7', '2020-07-18 00:00:00_14', '2020-10-10 00:00:00_21', '2021-01-02 00:00:00_84', '2021-03-27 00:00:00_84', '2021-06-19 00:00:00_84', '2021-09-12 00:00:00_84', '2021-12-05 00:00:00_84', '2022-02-27 00:00:00_84']

for id_forecasting_moment in list_id_forecasting_moment[:]:
    print(id_forecasting_moment)

    X_train = np.loadtxt('data/X_train_' + id_forecasting_moment + '.csv', delimiter=',')
    y_train = np.loadtxt('data/y_train_' + id_forecasting_moment + '.csv', delimiter=',')
    X_validation = np.loadtxt('data/X_validation_' + id_forecasting_moment + '.csv', delimiter=',')
    y_validation = np.loadtxt('data/y_validation_' + id_forecasting_moment + '.csv', delimiter=',')

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

    tuner = RandomSearch(
        build_model,
        objective='mse',
        max_trials=64,
        executions_per_trial=3,
        directory='tunning',
        project_name=id_forecasting_moment
    )

    tuner.search(
        x=X_train,
        y=y_train,
        validation_data=(X_validation, y_validation),
        epochs=128,
        shuffle=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
    )

    list_best_models = tuner.get_best_models(num_models=20)
    for i, best_model in enumerate(list_best_models):
        best_model.save('model/best_model_' + id_forecasting_moment + '_' + str(i) + '.keras')