# import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('car data.csv')
#print(df.head())

# handling outlier_cols

outlier_cols = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven']

def remove_outliers_iqr(data, column):
  q1,q2,q3 = np.percentile(data[column],[25,50,75])
  #print("q1,q2,q3 is :",q1,q2,q3)
  IQR = q3-q1
  #print("IQR is :" ,IQR)
  lower_limit = q1-(1.5*IQR)
  upper_limit = q3+(1.5*IQR)
  data[column]=np.where(data[column]>upper_limit,upper_limit,data[column]) # Capping the upper limit
  data[column]=np.where(data[column]<lower_limit,lower_limit,data[column]) # Flooring the lower limit

for column in outlier_cols:
  remove_outliers_iqr(df,column)


# Feature Engineering
df.drop(columns=['Car_Name'], inplace=True)
current_year = datetime.now().year

le_fuel = LabelEncoder()
le_trans = LabelEncoder()
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])
df = pd.get_dummies(df, columns=['Seller_Type'], drop_first=True)
with open('Fuel_Type.pkl', 'wb') as f:
    pickle.dump(le_fuel, f)

with open('Transmission.pkl', 'wb') as f:
    pickle.dump(le_trans, f)

#splitting data into dependent and independent columns
x=df.drop('Selling_Price',axis=1)
y=df['Selling_Price']

#Scaling
normalisation = StandardScaler()
x_scaled = normalisation.fit_transform(x)
# Coverting to Dataframe
x=pd.DataFrame(x_scaled)
with open('scaling.pkl', 'wb') as f:
    pickle.dump(normalisation, f)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.33)

# Train a Linear Regression model
#model = LinearRegression()
model = RandomForestRegressor()
model.fit(x_train, y_train)

# save the model 
pickle.dump(model,open('model.pkl','wb'))


