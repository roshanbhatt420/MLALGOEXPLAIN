from sklearn.linear_model import Ridge
risge_reg=Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(x,y)

