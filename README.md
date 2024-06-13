

# 🎬 Movie Data Analysis Project 🎥

This project showcases my skills in data analysis, visualization, and handling missing data using a movie dataset.

## 📦 Importing necessary libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

# Setting plot styles
plt.style.use('ggplot')
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None
```

## 📂 Reading the data

```python
# Reading the data
df = pd.read_csv(r'C:\Users\mikea\Downloads\movies.csv')

# Displaying the first few rows of the dataframe
df.head()
```

## 🔍 Checking for missing data

```python
# Checking for missing data
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100, 2)))
```

## 🛠 Displaying data types of columns

```python
# Displaying data types of columns
print(df.dtypes)
```

## 📊 Checking for outliers using boxplot

```python
# Checking for outliers using boxplot
df.boxplot(column=['gross'])
plt.show()
```

## 🗑 Dropping duplicates

```python
# Dropping duplicates
df = df.drop_duplicates()
```

## 🔄 Sorting data by gross revenue

```python
# Sorting data by gross revenue
df.sort_values(by=['gross'], inplace=False, ascending=False)
```

## 💰 Plotting budget vs gross earnings

```python
# Plotting budget vs gross earnings
plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.show()
```

## ⭐ Plotting score vs gross earnings

```python
# Plotting score vs gross earnings
sns.regplot(x="score", y="gross", data=df)
plt.show()
```

## 📈 Correlation matrix between all numeric columns using Pearson, Kendall, and Spearman methods

```python
# Correlation matrix between all numeric columns using Pearson, Kendall, and Spearman methods
print("Pearson correlation matrix:")
print(df.corr(method='pearson'))

print("Kendall correlation matrix:")
print(df.corr(method='kendall'))

print("Spearman correlation matrix:")
print(df.corr(method='spearman'))
```

## 🔥 Heatmap of Pearson correlation matrix

```python
# Heatmap of Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation matrix for Numeric Features")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()
```

## 🔢 Assigning numeric values to categorical columns

```python
# Assigning numeric values to categorical columns
df_numerized = df.copy()
for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        df_numerized[col_name] = df_numerized[col_name].astype('category').cat.codes
```

## 🔍 Displaying correlation matrix for numerized dataframe

```python
# Displaying correlation matrix for numerized dataframe
correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation matrix for Movies")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()
```

## 🏢 Grouping by company and year to get the gross sum

```python
# Grouping by company and year to get the gross sum
CompanyGrossSum = df.groupby(['company', 'year'])[['gross']].sum()
CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross', 'company', 'year'], ascending=False)[:15]
CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64')
print(CompanyGrossSumSorted)
```

## 📉 Scatter plot of budget vs gross earnings

```python
# Scatter plot of budget vs gross earnings
plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.show()
```

## 🐝 Swarm plot and strip plot for rating vs gross

```python
# Swarm plot and strip plot for rating vs gross
sns.swarmplot(x="rating", y="gross", data=df)
plt.show()

sns.stripplot(x="rating", y="gross", data=df)
plt.show()
```
```

