from openpyxl import load_workbook
import pandas as pd
wb = load_workbook(filename='/home/wilson/Work/Scripts/Data/XLSDataExtractor/Strategy_for_TOPIX.xlsx')
sheet = wb['data']


dates = sheet['A2:A3510']
values = sheet['D2:D3510']


result = []

for date, value in zip(dates, values):
    date = pd.to_datetime(date[0].value)
    result.append((date, value[0].value))


df = pd.DataFrame(result, columns=['Date', 'value'])
df = df.set_index('Date')
df.to_csv('test.csv')
