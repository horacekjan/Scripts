import pandas as pd


df_tom = pd.read_csv(r'D:\Work\Tmp\Esteban\0\Tom.csv', index_col=0)
df_esteban = pd.read_csv(r'D:\Work\Tmp\Esteban\0\Esteban.csv', index_col=0)
df_roman = pd.read_csv(r'D:\Work\Tmp\Esteban\0\Roman.csv', index_col=0)


print(df_tom.equals(df_esteban))
print(df_tom.equals(df_roman))
print(df_esteban.equals(df_roman))