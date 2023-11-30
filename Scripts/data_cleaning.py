# Import necessary libraries
import pandas as pd

# Load the Excel file
excel_path = '../data/combined_Red_Irish_Potatoes.xlsx'
df = pd.read_excel(excel_path)

# Drop irrelevant columns
df.drop(['Commodity', 'Classification', 'Grade', 'Sex'], axis=1, inplace=True)

# Convert 'Wholesale' and 'Retail' to numerical values
df['Wholesale_num'] = df['Wholesale'].str.extract('(\d+.\d+)').astype(float)
df['Retail_num'] = df['Retail'].str.extract('(\d+.\d+)').astype(float)
df.drop(['Wholesale', 'Retail'], axis=1, inplace=True)  # Drop original columns

# Convert 'Date' to datetime and sort data
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Market', 'Date'], inplace=True)

# Apply 7-day rolling average to fill missing values in 'Wholesale_num' and 'Retail_num'
df['Wholesale_num'] = df.groupby('Market')['Wholesale_num'].transform(lambda x: x.fillna(x.rolling(14, min_periods=1).mean()))
df['Retail_num'] = df.groupby('Market')['Retail_num'].transform(lambda x: x.fillna(x.rolling(14, min_periods=1).mean()))

# Drop rows with missing values
df.dropna(subset=['Wholesale_num', 'Retail_num', 'Supply Volume'], inplace=True)

# Calculate IQR and determine outliers for 'Wholesale_num' and 'Retail_num'
Q1_wholesale, Q3_wholesale = df['Wholesale_num'].quantile([0.25, 0.75])
IQR_wholesale = Q3_wholesale - Q1_wholesale
outliers_wholesale = df[(df['Wholesale_num'] < (Q1_wholesale - 1.5 * IQR_wholesale)) | (df['Wholesale_num'] > (Q3_wholesale + 1.5 * IQR_wholesale))]

Q1_retail, Q3_retail = df['Retail_num'].quantile([0.25, 0.75])
IQR_retail = Q3_retail - Q1_retail
outliers_retail = df[(df['Retail_num'] < (Q1_retail - 1.5 * IQR_retail)) | (df['Retail_num'] > (Q3_retail + 1.5 * IQR_retail))]

# Combine and drop outliers
outliers_combined = pd.concat([outliers_wholesale, outliers_retail]).drop_duplicates().sort_values(by='Date')
df_cleaned_without_outliers = df.drop(outliers_combined.index)

# Save the cleaned DataFrame
pickle_path = '../data/cleaned_data.pkl'
df_cleaned_without_outliers.to_pickle(pickle_path)
