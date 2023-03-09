# # Performing a simple logictic regression model to predict house prices

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import the data
df = pd.read_excel('innercity.xlsx')

# Ignore the rows which has $ as value in any column
df = df[~df.isin(['$']).any(axis=1)]

# Output data description
print("Summary of the data: ")
print(df.describe())

# Output data information
print("Information of the data: ")
print(df.info())

# calculate the skewness of the data
print("Skewness of the data: ")
print(df.skew())

# Plot Data Distribution
df.hist(figsize=(10,10))

# Pairplot
import seaborn as sns
sns.pairplot(df)

# heat map of correlation
sns.heatmap(df.corr())

#outlier detection
# Plot a boxplot for each column except the forst two columns
#outlier detection
for i in range(2, len(df.columns)):
    column = df.columns[i]
    sns.boxplot(df[column])

# outlier treatment
df = df[df['room_bed'] < 8]
df = df[df['room_bath'] < 5]
df = df[df['living_measure'] < 6000]
df = df[df['lot_measure'] < 100000]
df = df[df['ceil'] < 4]
df = df[df['coast'] < 2]
df = df[df['sight'] < 5]
df = df[df['condition'] < 5]
df = df[df['quality'] < 12]
df = df[df['ceil_measure'] < 6000]
df = df[df['basement'] < 4000]
df = df[df['yr_built'] > 1900]
df = df[df['yr_renovated'] < 2015]
df = df[df['zipcode'] < 98080]
df = df[df['lat'] > 47]
df = df[df['long'] < -120]
df = df[df['living_measure15'] < 6000]
df = df[df['lot_measure15'] < 100000]
df = df[df['total_area'] < 100000]

# drop the cid and dayhours column
df = df.drop(['cid'], axis=1)
df = df.drop(['dayhours'], axis=1)
# Ignore redundant columns
df = df.drop(['basement'], axis=1)
df = df.drop(['yr_renovated'], axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
var = ['living_measure','lot_measure','ceil_measure','living_measure15','lot_measure15','total_area']
df[var] = scaler.fit_transform(df[var])

# Encode using dummy variables
df = pd.get_dummies(df, columns=['room_bed','room_bath','ceil','coast','sight','condition','quality','zipcode'])

# split the data into train and test
from sklearn.model_selection import train_test_split
X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Load Linear Regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

# print the intercept and coefficients
print('Intercept:')
print(lm.intercept_)
print('Coefficients:')
print(lm.coef_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print('Coefficients of the model (dataframe):')
print(coeff_df)

# predict the values
y_pred = lm.predict(X_test)

# Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

# Plotting the predicted values against the actual values and the line of best fit
plt.figure(figsize=(12,6))
plt.scatter(y_test,lm.predict(X_test),c='blue',marker='o',edgecolors='white')
plt.title('Linear Regression')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.plot([0, 4000000], [0, 4000000], 'k-', color = 'r')
plt.show()








