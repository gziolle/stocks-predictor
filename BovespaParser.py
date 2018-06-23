from os import listdir
from datetime import datetime
from preprocessing import Preprocessing
import numpy as np

class BovespaParser:

    def __init__(self, path, years = 'ALL', rsi=False, ma=None, macd = False):
        self.path = path
        self.years = years
        self.data = dict()
        self.rsi = rsi
        self.ma = ma
        self.macd = macd

    def parse_data(self, years='ALL'):
        if years == 'ALL':
            cotahists = listdir(self.path)
            cotahists.sort()
            for filename in cotahists:
                with open("stocks/" + filename, encoding = "ISO-8859-1") as file:
                    for line in file.readlines():
                        try:
                            self.parse_line(line)
                        except:
                            pass

    def parse_line (self, text):
        # remove mercado a prazo
        if text[49:52].replace(" ", '') != "":
            return
        # get just stocks
        if text[39:45].replace(" ", '') != "ON" and text[39:45].replace(" ", '') != "PN":
            return
        date = text[2:10]
        code = text[12:24].replace(' ', '')
        open_value = int(text[56:69]) / 100
        max_value = int(text[69:82]) / 100
        min_value = int(text[82:95]) / 100
        close = int(text[108:121]) / 100
        volume = int(text[170:188])
        if code not in self.data:
            self.data[code] = []
        self.data[code].append([date, open_value, max_value, min_value,close, volume])

    def get_stocks_names(self):
        return [stock for stock in self.data]

    def get_stock(self, stock):
        return self.data[stock]

    def save_stock(self, stock):
        with open("dataset/" + stock + ".csv", 'w+') as file:
            file.write("date,open,max,min,close,volume,y\n")
            prep = Preprocessing()
            close = np.array(self.data[stock])[:, 4]
            y = prep.create_train_result(close)
            for index in range(len(y)):
                x = ",".join([str(i) for i in self.data[stock][index]])
                file.write(x + "," + str(y[index]) + "\n")
