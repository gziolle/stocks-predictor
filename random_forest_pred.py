from utils import *
import pandas as pd
import matplotlib.pylab as plt
from indexes import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

if __name__ == '__main__':
    TP, TN, FN, FP = [], [], [], []
    windows = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    balance = []
    F1 = []
    for window in windows:
        print ("---- window size ---- " + str(window))
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        Y_pred = []
        STEP = 1
        WINDOW = window
        EMB_SIZE = 1
        FORECAST = 10
        N_TREES = 100
        close_price = []
        exp_movavg = []
        for stock in  ['ITSA4.csv', 'ITUB3.csv', 'BBDC3.csv', 'BBDC4.csv', 'BBSE3.csv']:
            X, Y = [], []
            data_original = pd.read_csv('./dataset/' + stock)
            # real prices
            closep = data_original.ix[:, 'close'].tolist()
            y = np.array(np.array(closep[FORECAST:]) > np.array(closep[:-FORECAST]), dtype=np.int)[1:]
            # exponential moving average calcs
            ema = exp_moving_average(closep, window=9)
            for i in range(0, len(data_original), STEP):
                try:
                    avg = ema[i:i+WINDOW]
                    if np.std(avg) == 0:
                        avg = (np.array(avg) - np.mean(avg)) / 0.001
                    else:
                        avg = (np.array(avg) - np.mean(avg)) / np.std(avg)
                    if y[i + WINDOW - 1]:
                        y_i = 1
                    else:
                        y_i = 0
                    x_i = np.column_stack((avg))
                except Exception as e:
                    break
                X.append(x_i)
                Y.append(y_i)
            # append results
            # extract features and create output
            X, Y = np.array(X), np.array(Y)
            X_train_st, X_test_st, Y_train_st, Y_test_st = create_Xt_Yt(X, Y)
            X_train = np.append(X_train, X_train_st)
            Y_train = np.append(Y_train, Y_train_st)
            Y_test = np.append(Y_test, Y_test_st)
            X_test = np.append(X_test, X_test_st)

        # reshape the estationary data
        X_train = np.reshape(X_train, (int(X_train.shape[0]/(WINDOW*EMB_SIZE)), WINDOW))
        X_test = np.reshape(X_test, (int(X_test.shape[0]/(WINDOW*EMB_SIZE)), WINDOW))

        # train
        clf = RandomForestClassifier(n_estimators = N_TREES, max_depth=int(WINDOW / 2), random_state=0, n_jobs=-1)
        clf.fit(X_train, Y_train)
        # predict
        Y_pred = clf.predict_proba(X_test)
        y_pred = Y_pred[:, 1] > Y_pred[:, 0]
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        TN.append(tn)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        f1 = f1_score(Y_test, y_pred, average='weighted')
        F1.append(f1)
        # print ("true neg  : " + str(tn) + " false positive : " + str(fp))
        # print ("false neg : " + str(fn) + " true positive  : " + str(tp))
        print ("Class Test Balance ^     : " + str(sum(Y_test)/len(Y_test)))
        print ("Class Training Balance   : " + str(sum(Y_train)/len(Y_train)))
        print ("Class Prediction Balance : " + str(sum(y_pred)/len(Y_test)))
        print ("Precision down %         : " + str(float(tn) / (tn + fn)))
        print ("Precision up %           : " + str(float(tp) / (tp + fp)))
        print ("F1 score                 : " + str(f1))
        print ("\n")
        balance.append(sum(y_pred)/len(Y_test))

    precision_down = [TN[i] / (TN[i] + FN[i]) for i in range(len(TN))]
    precision_up = [TP[i] / (TP[i] + FP[i]) for i in range(len(TP))]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(windows, precision_down)
    plt.plot(windows, precision_up)
    plt.plot(windows, F1)
    plt.title('Random Forest Precision')
    plt.ylabel('Precision (%)')
    plt.legend(['Precision Down', 'Precision Up', 'F1 score'], loc='best')
    plt.grid()

    plt.subplot(212)
    plt.bar(windows, balance, 5, color="blue")
    plt.ylabel('Up/Down distribution')
    plt.xlabel('Window size')
    plt.legend(['Distribution ones on predictions (%)'], loc='best')
    plt.grid()
    plt.show()
