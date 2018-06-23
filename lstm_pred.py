from utils import *
import pandas as pd
import matplotlib.pylab as plt
from indexes import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':
    TDOWN, TUP, FUP, FDOWN = [], [], [], []
    F1 = []
    windows = range(20,70,10)
    balance = []
    for window in windows:
        print ("---- window size ---- " + str(window))
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        Y_pred = []
        STEP = 1
        WINDOW = window
        EMB_SIZE = 2
        FORECAST = 10
        close_price = []
        exp_movavg = []
        for stock in  ['ITSA4.csv', 'ITUB3.csv', 'BBDC3.csv', 'BBDC4.csv', 'BBSE3.csv']:
            X, Y = [], []
            data_original = pd.read_csv('./dataset/' + stock)
            # real prices
            closep = data_original.ix[:, 'close'].tolist()
            volume = data_original.ix[:, 'volume'].tolist()
            y = np.array(np.array(closep[FORECAST:]) > np.array(closep[:-FORECAST]), dtype=np.int)[1:]
            # exponential moving average calcs
            ema = exp_moving_average(closep, window=9)
            rsi = relative_strength_index(closep)
            obv = on_balance_volume(closep, volume)
            for i in range(0, len(data_original), STEP):
                try:
                    avg = ema[i:i+WINDOW]
                    r = rsi[i:i+WINDOW]
                    o   = obv[i:i+WINDOW]
                    if np.std(r) == 0:
                        r = (np.array(r) - np.mean(r)) / 0.001
                    else:
                        r = (np.array(r) - np.mean(r)) / np.std(r)
                    if np.std(avg) == 0:
                        avg = (np.array(avg) - np.mean(avg)) / 0.001
                    else:
                        avg = (np.array(avg) - np.mean(avg)) / np.std(avg)
                    if np.std(o) == 0:
                        o = (np.array(o) - np.mean(o)) / 0.001
                    else:
                        o = (np.array(o) - np.mean(o)) / np.std(o)
                    if y[i + WINDOW - 1]:
                        y_i = [1, 0]
                    else:
                        y_i = [0, 1]
                    x_i = np.column_stack((avg, o)) # r,
                except Exception as e:
                    break
                X.append(x_i)
                Y.append(y_i)
            # append results
            # extract features and create outrue_downut
            X, Y = np.array(X), np.array(Y)
            X_train_st, X_test_st, Y_train_st, Y_test_st = create_Xt_Yt(X, Y)
            X_train = np.append(X_train, X_train_st)
            Y_train = np.append(Y_train, Y_train_st)
            Y_test = np.append(Y_test, Y_test_st)
            X_test = np.append(X_test, X_test_st)
        # reshape the estationary data
        X_train = np.reshape(X_train, (int(X_train.shape[0]/(WINDOW*EMB_SIZE)), WINDOW, EMB_SIZE))
        Y_train = np.reshape(Y_train, (int(Y_train.shape[0]/2), 2))
        X_test = np.reshape(X_test, (int(X_test.shape[0]/(WINDOW*EMB_SIZE)), WINDOW, EMB_SIZE))
        Y_test = np.reshape(Y_test, (int(Y_test.shape[0]/2), 2))
        model = Sequential()

        model.add(LSTM(units=30, return_sequences=True, input_shape = (WINDOW, EMB_SIZE)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())

        model.add(LeakyReLU())
        model.add(Dense(2))
        model.add(Activation('softmax'))
        opt = RMSprop(lr=0.002)

        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
        checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)

        model.compile(optimizer=opt,
                    loss='categorical_hinge',
                    metrics=['accuracy'])
        history = model.fit(X_train, Y_train,
                nb_epoch = 50,
                batch_size = 16,
                verbose=1,
                validation_data=(X_test, Y_test),
                callbacks=[reduce_lr, checkpointer],
                shuffle=False)

        model.load_weights("lolkek.hdf5")
        pred = model.predict(np.array(X_test))

        Y_pred = model.predict_proba(X_test)
        y_pred = Y_pred > 0.5

        true_up, false_down, false_up, true_down = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in Y_pred]).ravel()
        f1 = f1_score(Y_test, y_pred, average='weighted')

        TUP.append(true_up)
        TDOWN.append(true_down)
        FDOWN.append(false_down)
        FUP.append(false_up)
        F1.append(f1)

        print ("Class Test Balance ^     : " + str(sum(Y_test)/len(Y_test)))
        print ("Class Training Balance   : " + str(sum(Y_train)/len(Y_train)))
        print ("Class Prediction Balance : " + str(sum(y_pred)/len(Y_test)))
        print ("Precision up %           : " + str(float(true_up) / (true_up + false_up)))
        print ("Precision down %         : " + str(float(true_down) / (true_down + false_down)))
        print ("\n")

        balance.append(sum(y_pred)/len(Y_test))
        fpr, tpr, _ = roc_curve(Y_test[:, 0], Y_pred[:, 0])
        auc = metrics.roc_auc_score(Y_test[:, 0], Y_pred[:, 0])
        plt.plot(fpr,tpr,label="ROC Up window=" + str(window) +  ", auc="+str(auc))
        fpr, tpr, _ = roc_curve(Y_test[:, 1], Y_pred[:, 1])
        auc = metrics.roc_auc_score(Y_test[:, 1], Y_pred[:, 1])
        plt.plot(fpr,tpr,label="ROC Down window=" + str(window) +  ", auc="+str(auc))
        plt.legend(loc=4)
        plt.title('ROC')
        plt.ylabel('True Rate')
        plt.xlabel('False Rate')

    plt.grid()
    plt.show()

    precision_down = [TUP[i] / (TUP[i] + FUP[i]) for i in range(len(TUP))]
    precision_up = [TDOWN[i] / (TDOWN[i] + FDOWN[i]) for i in range(len(TDOWN))]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(windows, precision_down)
    plt.plot(windows, precision_up)
    plt.plot(windows, F1)
    plt.title('LSTM Precision')
    plt.ylabel('Precision (%)')
    plt.legend(['Precision Down', 'Precision Up', 'F1 score'], loc='best')
    plt.grid()

    plt.subplot(212)
    plt.bar(windows, np.array(balance)[:, 0], 5, color="blue")

    plt.ylabel('Up/Down distribution')
    plt.xlabel('Window size')
    plt.legend(['Distribution of up predictions (%)'], loc='best')
    plt.grid()
    plt.show()

