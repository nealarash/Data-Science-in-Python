
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv("Ecommerce Customers")
df


df.describe()
df.info()


# Exploratory Data Analysis

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x="Time on Website",y="Yearly Amount Spent", data=df)


sns.jointplot(x="Time on Website",y="Yearly Amount Spent", data=df, kind="hex")


sns.pairplot(df)


# The length of membership is the most correlated feature

sns.lmplot(x="Length of Membership",y="Yearly Amount Spent", data=df)

# Training and Testing Data

df.columns
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_


predictions=lm.predict(X_test)


plt.scatter(predictions,y_test)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# Evaluating the Model

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# Residuals

sns.distplot((y_test-predictions),bins=50)


# Conclusion

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
