import datetime

numdays = 366

#base = datetime.date.today()
base = datetime.date(2015, 1, 1)
date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]

for date in date_list:
    print(date)
