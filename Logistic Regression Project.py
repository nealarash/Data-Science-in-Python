

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


ad_data= pd.read_csv('advertising.csv')


ad_data.head()
ad_data.info
ad_data.describe()


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


sns.jointplot(data=ad_data, x="Age", y="Area Income")


sns.jointplot(x = "Age", y = "Daily Time Spent on Site",
              kind = "kde", data = ad_data, color='red');



sns.jointplot(data=ad_data, x="Daily Time Spent on Site", y="Daily Internet Usage", color='green')



sns.pairplot(data=ad_data,hue='Clicked on Ad',palette='bwr')



from sklearn.model_selection import train_test_split


ad_data.head()


x = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]

y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, 
                                                    random_state=101)


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


