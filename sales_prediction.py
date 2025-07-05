import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
data = pd.read_csv('data/advertising.csv')

# Step 2: Data Cleaning
print("Missing values:\n", data.isnull().sum())
data.dropna(inplace=True)

# Step 3: Exploratory Data Analysis
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
plt.tight_layout()
plt.show()

# Step 4: Feature Selection
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Step 9: Coefficients
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Impact:\n", coef_df)

# Step 10: Predict Future Scenario (Example)
new_data = pd.DataFrame({
    'TV': [200],
    'Radio': [25],
    'Newspaper': [20]
})
predicted_sales = model.predict(new_data)
print("\nPredicted Future Sales:", predicted_sales[0])
