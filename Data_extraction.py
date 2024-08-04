import yfinance as yf

end_date = '2024-03-31'
start_date = '2004-03-31'


ticker_dict = { 'SP500': '^GSPC', 'NIFTY50': '^NSEI', 'FTSE100': '^FTSE'}

for ticker in ticker_dict:
    print('Downloading data for:',ticker)
    data = yf.download(ticker_dict[ticker], start=start_date, end=end_date)
    print('Data Downloaded for:',ticker)
    name = str(ticker)
    data.to_pickle(f'data/{name}.pkl')
    print('Data Saved for:',ticker)
    print('-----------------------------------')
