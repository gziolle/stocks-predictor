from BovespaParser import BovespaParser
import numpy as np
from preprocessing import Preprocessing
from os import mkdir

if __name__ == '__main__':
    try:
        mkdir("dataset")
    except:
        pass
    bvp = BovespaParser("stocks")
    bvp.parse_data()
    stock_names = bvp.get_stocks_names()

    for stock in stock_names:
        # Get each stock data ['data', 'close', 'volume']
        X = bvp.get_stock(stock)
        X = np.array(X)

        # ignore Stocks with less than 1k entries
        if len(X) < 1000:
            continue
        print (stock + ": " + str(len(X)))

        # save each stock 
        bvp.save_stock(stock)
