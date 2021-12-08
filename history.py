import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime
import calendar
import csv

# This program downloads minute level ticker barset data for a given range of time from Alpaca.

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

ACCESS_KEY = "AKMPZZHOCXIIMX604N7T"
SECRET_KEY = "z4O9LNGbQNTpmRVAgGZX644pClj4jiKehUQiq0oD"

TICKER = 'AAPL'

NY = 'America/New_York'

api = tradeapi.REST(ACCESS_KEY, SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

start=pd.Timestamp('2015-01-01 8:00', tz=NY)
end=pd.Timestamp('2016-01-01 4:00', tz=NY)

with open('{}_data_file.csv'.format(TICKER.lower()), mode='w') as data_file:
	barset = api.get_barset(TICKER, '1D', start=start.isoformat(), end=end.isoformat(), limit=1000)
	tickerwriter = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for item in barset[TICKER]:
		tickerwriter.writerow([item.o, item.c, item.h, item.l, item.v])
		#print(counter)	
		
