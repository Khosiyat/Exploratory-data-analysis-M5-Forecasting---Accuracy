import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU



root_dir = Path('/kaggle/')
dir_ = root_dir / 'input/competitive-data-science-predict-future-sales'
list(dir_.glob('*'))
#read files
sales_train = pd.read_csv(dir_ / 'sales_train.csv')
items_dataFrame = pd.read_csv(dir_ / 'items.csv')
item_categories_dataFrame = pd.read_csv(dir_ / 'item_categories.csv')
shop_dataFrame = pd.read_csv(dir_ / 'shops.csv')

##################___Preprocessing Data

#joining all the available csv files
training_dataFRame = sales_train.join(items_dataFrame, on='item_id', how='outer', lsuffix='left_side', rsuffix='right_side')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id']]
training_dataFRame = training_dataFRame.join(item_categories_dataFrame, on='item_category_id', how='outer', lsuffix='left_side', rsuffix='right_side')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id', 'item_category_name']]
training_dataFRame = training_dataFRame.join(shop_dataFrame, on='shop_id', how='outer', lsuffix='left_side', rsuffix='right_side')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id', 'item_category_name', 'shop_name']]
training_dataFRame.reset_index(drop=True, inplace=True)


#Cleaning the data
training_dataFRame.dropna(inplace=True)
training_dataFRame.drop_duplicates(inplace=True)



##################___ features are engeenered as modifying their fields and columns. 
training_dataFRame[['day', 'month', 'year']] = training_dataFRame.date.str.split('.', expand=True)#date columns are split into day/month/year.
training_dataFRame.day = training_dataFRame.day.apply(lambda x: int(x))
training_dataFRame.month = training_dataFRame.month.apply(lambda x: int(x))
training_dataFRame.year = training_dataFRame.year.apply(lambda x: int(x))
#statistical description o the columns



training_dataFRame.describe()
#groupby the dates
year_group_count = training_dataFRame.groupby('year').count().item_id.reset_index()
year_group_count.columns = ['year', 'total_bill']
month_group_count = training_dataFRame.groupby('month').count().item_id.reset_index()
month_group_count.columns = ['month', 'total_bill']
day_group_and_count = training_dataFRame.groupby('day').count().item_id.reset_index()
day_group_and_count.columns = ['day', 'total_bill']
#Visualize the bills according to data
fig, axes = plt.subplots(1, 3, figsize=(20, 4))
sb.barplot(x='year', y='total_bill', data=year_group_count, ax=axes[0])
sb.barplot(x='month', y='total_bill', data=month_group_count, ax=axes[1])
sb.barplot(x='day', y='total_bill', data=day_group_and_count, ax=axes[2])
plt.style.use('seaborn')
plt.xkcd()



#groupby the date block num
date_block_count = training_dataFRame.groupby('date_block_num').count().item_id.reset_index()
#Visualize the blocked date and total bill
date_block_count.columns = ['date_block', 'total_bill']
fig = plt.figure(figsize=(12, 4))
ax = fig.add_axes([0, 0, 1, 1])
sb.barplot(x='date_block', y='total_bill', data=date_block_count, ax=ax)
plt.gcf().autofmt_xdate()
plt.legend()
plt.xlabel('Amount of Date block')
plt.ylabel("Total amount of Bill")
plt.title("Total Bill")

plt.style.use('seaborn')
plt.xkcd()
