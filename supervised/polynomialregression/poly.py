import numpy as np
import random
import matplotlib as plt
import matplotlib.pyplot as mlt
from sklearn.linear_model import LinearRegression
 
m=100
x=6*np.random.rand(m,1)-3
y=0.5*x**2+2+np.random.rand(m,1)
from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly_features.fit_transform(x)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)



mlt.scatter(x,y)
mlt.xlabel("X values")
mlt.ylabel("Y values")
mlt.show()


