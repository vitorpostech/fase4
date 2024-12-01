import yfinance as yf
import pickle

def data_get(symbol, start_date, end_date):

    dataframe = yf.download(symbol, start=start_date, end=end_date)
    dataframe.columns = ['Adj Close'] + [col[0] for col in dataframe.columns[1:]]

    return dataframe

def save_raw_data(symbol, start_date, end_date):
    raw_path= f'data/raw/{symbol}_{start_date}_{end_date}.pkl'
    dataframe= data_get(symbol, start_date, end_date)
    with open(raw_path, 'wb') as file_:
        pickle.dump(dataframe, file_)
    file_.close()
    print(raw_path)

def load_raw_data(path):
    with open(path, 'rb') as file_:
        dataframe= pickle.load(file_)
    file_.close()

    return dataframe