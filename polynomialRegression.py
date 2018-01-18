#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset= pd.read_csv('Position_Salaries.csv');
X= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#No feature scaling and division into train and test set is needed
#building the models: Linear and Polynomial
from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
regressor= PolynomialFeatures(degree=2)
poly_X= regressor.fit_transform(X)
lin_reg=LinearRegression()
lin_reg.fit(poly_X,y)
#plotting points
plt.scatter(X,y,color='red')
plt.plot(X,reg1.predict(X),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#improving the curve
X_grid=np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape((len(X_grid)),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg.predict(regressor.fit_transform(X_grid)),color='blue')
plt.title('Salary vs Experience(poly)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

reg1.predict(6.5)
lin_reg.predict(regressor.fit_transform(6.5))
