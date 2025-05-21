import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Excel file directly from your path
data_file = "/Users/aryaanashraf/Desktop/celebal/projects/online_retail_II.xlsx"
df = pd.read_excel(data_file)

# Step 2: Clean and prepare the data
df = df.dropna(subset=['Customer ID'])  # Correct column name
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Filter only valid (positive) quantity and price
df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]  # Dataset uses 'Price', not 'UnitPrice'

# Calculate Total transaction amount
df['TotalPrice'] = df['Quantity'] * df['Price']

# Set snapshot date for Recency calculation
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Step 3: Calculate RFM values
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# Step 4: Model training
X = rfm[['Recency', 'Frequency']]
y = rfm['Monetary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Step 5: Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")
