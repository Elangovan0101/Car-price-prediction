import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('CarPrice_Assignment.csv')


df.drop_duplicates(inplace=True)
df.drop(columns=['car_ID', 'symboling', 'CarName'], inplace=True)


categorical_columns = df.select_dtypes(include=object).columns.tolist()
numerical_columns = df.select_dtypes(exclude=object).columns.tolist()
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Split data into features and target
x = df.drop('price', axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test, y_pred)
print(f"Linear Regression - R-squared: {r2_square}")
print(f'Linear Regression - Mean Squared Error: {mse}')

forest = RandomForestRegressor()
forest.fit(x_train, y_train)
fors_pred = forest.predict(x_test)
mse = mean_squared_error(y_test, fors_pred)
r2_square = r2_score(y_test, fors_pred)
print(f"Random Forest - R-squared: {r2_square}")
print(f'Random Forest - Mean Squared Error: {mse}')
