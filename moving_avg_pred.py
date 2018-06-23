from utils import *
import pandas as pd
import matplotlib.pylab as plt
from indexes import *
import numpy as np
from sklearn.metrics import f1_score

if __name__ == '__main__':
    Y = []
    Y_pred = []
    STEP = 1
    close_price = []
    exp_movavg = []
    for stock in  ['ITSA4.csv', 'ITUB3.csv', 'BBDC3.csv', 'BBDC4.csv', 'BBSE3.csv']:
        data_original = pd.read_csv('./dataset/' + stock)
        # real data
        closep = data_original.ix[:, 'close'].tolist()
        y = np.array(np.array(closep[STEP:]) > np.array(closep[:-STEP]), dtype=np.int)
        # prediction
        ema = exp_moving_average(closep, window=9)
        y_pred = np.array(np.array(ema[STEP:]) > np.array(ema[:-STEP]), dtype=np.int)[:-1]
        # append results
        Y = np.append(Y, y[1:])
        Y_pred = np.append(Y_pred, y_pred)
        close_price = np.append(close_price, closep)
        exp_movavg = np.append(exp_movavg, ema)

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()

    print (str(tn) + ", " + str(fp))
    print (str(fn) + ", " + str(tp))
    print("Precision down %: " + str(float(tn) / (tn + fn)))
    print("Precision up %: " + str(float(tp) / (tp + fp)))

    plt.plot(close_price[200:1000])
    plt.plot(exp_movavg[200:1000])
    plt.title('Moving average aproximation')
    plt.ylabel('price')
    plt.xlabel('time')
    plt.legend(['close price', 'exp moving average'], loc='best')
    plt.grid()
    plt.show()

    true_up, false_down, false_up, true_down = confusion_matrix(Y, Y_pred).ravel()
    f1 = f1_score(Y, Y_pred, average='weighted')
    print ("Class Balance ^     : " + str(sum(Y)/len(Y)))
    print ("Class Prediction Balance : " + str(sum(Y_pred)/len(Y)))
    print ("Precision up %           : " + str(float(true_up) / (true_up + false_up)))
    print ("Precision down %         : " + str(float(true_down) / (true_down + false_down)))
    print ("F1 score                 : " + str(f1))
    print ("\n")
