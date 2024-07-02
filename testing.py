from trading_bot import initialize_lc, fetch_data
from datetime import datetime
from advanced_ta import LorentzianClassification

df = fetch_data('TSLA', start_date='2020-01-01', end_date=datetime.now().strftime('%Y-%m-%d'))
lc = LorentzianClassification(df)
lc.dump('output/result.csv')
lc.plot('output/result.jpg')
df['classification'] = lc.classify()
print(df['classification'])