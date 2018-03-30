import numpy as np
np.random.seed(5)

b0 = 2
b1 = 1
N = 100
step = 0.2
mu = 0 # pas de biais
sigma = 10

x = np.random.randn(int(N/step))*5
# x = np.arange(0, N, step)
e = np.random.normal(mu, sigma, int(N/step))
y = b0 + b1*x + e

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, alpha=0.5, color='orchid')
fig.suptitle('Example OSL')
fig.tight_layout(pad=2);
ax.grid(True)
fig.savefig('data_osl.png', dpi=125)


import statsmodels.api as sm

# converti en matrice des features
x = sm.add_constant(x) # constant intercept term
# Model: y ~ x + c

model = sm.OLS(y, x)
fitted = model.fit()

x_pred = np.linspace(x.min(), x.max(), 50)
x_pred2 = sm.add_constant(x_pred)

y_pred = fitted.predict(x_pred2)
ax.plot(x_pred, y_pred, '-', color='darkorchid', linewidth=2)
fig.savefig('data_osl_droite.png', dpi=125)

print(fitted.params)     # the estimated parameters for the regression line
print(fitted.summary())  # summary statistics for the regression


# Calcul de l intervalle de confiance
y_hat = fitted.predict(x) # x is an array from line 12 above
y_err = y - y_hat
mean_x = x.T[1].mean()
n = len(x)
dof = n - fitted.df_model - 1

from scipy import stats
# cet IC est corrige en fonction de la distribution des donnee
# le plus petit IC se trouve ou la plus forte concentration de donnee
t = stats.t.ppf(1-0.025, df=dof)
s_err = np.sum(np.power(y_err, 2))
conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x_pred-mean_x),2) /
                  ((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))

upper = y_pred + abs(conf)
lower = y_pred - abs(conf)
ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4)

fig.savefig('data_osl_droite_ICt.png', dpi=125)

from statsmodels.sandbox.regression.predstd import wls_prediction_std

sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.05)
ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1)
fig.savefig('filename4.png', dpi=125)
