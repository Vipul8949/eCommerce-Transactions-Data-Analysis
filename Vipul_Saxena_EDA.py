import os

print("Current Directory:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading and preprocessing dataset
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')


print(customers.head())
print(products.head())
print(transactions.head())

print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())


print(customers.duplicated().sum())
print(products.duplicated().sum())
print(transactions.duplicated().sum())

print(customers.describe())
print(products.describe())
print(transactions.describe())

# Merging all the datasets together
merged = pd.merge(transactions, customers, on='CustomerID', how='left')


merged = pd.merge(merged, products, on='ProductID', how='left')

print(merged.head())


missing_values = merged.isnull().sum()
print("Missing Values:\n", missing_values)


missing_percentage = (missing_values / len(merged)) * 100
print("\nMissing Values Percentage:\n", missing_percentage)
print("                                               ")



# Customer Analysis

print("Unique Customers:", merged['CustomerID'].nunique())

region_counts = merged['Region'].value_counts()


region_percentage = (region_counts / region_counts.sum()) * 100

region_summary = pd.DataFrame({
    'Customer Count': region_counts,
    'Percentage (%)': region_percentage
})


print(region_summary)

# Plotting result
region_counts.plot(kind='bar', title='Customers by Region', color='blue')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.show()

#Product Analysis


print("Unique Products:", merged['ProductID'].nunique())
print("                                               ")
print("Unique Categories:", merged['Category'].nunique())
print("                                               ")

import pandas as pd
import matplotlib.pyplot as plt

def product_summary(data, top_n=5, lowest_n=5):
 
    total_quantity_sold = data['Quantity'].sum()

  
    popular_products = data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(top_n)
    print(f"Top {top_n} Most Popular Products:")
    print(popular_products)

   
    product_percentage_popular = (popular_products / total_quantity_sold) * 100

    popular_product_summary = pd.DataFrame({
        'Quantity Sold': popular_products,
        'Percentage (%)': product_percentage_popular
    })


    lowest_selling_products = data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=True).head(lowest_n)
    print(f"\n{lowest_n} Lowest Selling Products:")
    print(lowest_selling_products)


    product_percentage_lowest = (lowest_selling_products / total_quantity_sold) * 100


    lowest_selling_summary = pd.DataFrame({
        'Quantity Sold': lowest_selling_products,
        'Percentage (%)': product_percentage_lowest
    })


    print("\nProduct Summary (Top and Lowest Selling Products with Percentages):")
    print("\nPopular Products Summary:")
    print(popular_product_summary)
    print("\nLowest Selling Products Summary:")
    print(lowest_selling_summary)

    # Plot Top Products
    popular_products.plot(kind='bar', title=f'Top {top_n} Popular Products', color='red', label='Popular Products')
    plt.xlabel('Product Name')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.show()

    # Plot lowest selling  Products
    lowest_selling_products.plot(kind='bar', title=f'{lowest_n} Lowest Selling Products', color='blue', label='Lowest Selling Products')
    plt.xlabel('Product Name')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.show()


product_summary(merged)

# Transaction Analysis


total_revenue = merged['TotalValue'].sum()
print("Total Revenue: $", total_revenue)

revenue_by_region = merged.groupby('Region')['TotalValue'].sum()
print("\nRevenue by Region:")
print(revenue_by_region)

revenue_percentage_by_region = (revenue_by_region / total_revenue) * 100

region_revenue_summary = pd.DataFrame({
    'Total Revenue (USD)': revenue_by_region,
    'Revenue Percentage (%)': revenue_percentage_by_region
})


print("\n       Region Revenue Summary :     \n")
print(region_revenue_summary)

#plotting
revenue_by_region.plot(kind='bar', title='Revenue by Region', color='green')
plt.xlabel('Region')
plt.ylabel('Revenue')
plt.show()


# Category wise revenue


revenue_by_category = merged.groupby('Category')['TotalValue'].sum()


category_revenue_percentage = (revenue_by_category / total_revenue) * 100


category_revenue_summary = pd.DataFrame({
    'Total Revenue (USD)': revenue_by_category,
    'Revenue Percentage (%)': category_revenue_percentage
})


print("\n       Product Category Revenue Summary       :\n")
print(category_revenue_summary)

#plotting
revenue_by_category.plot(kind='bar', title='Revenue by Product Category', color='purple')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue')
plt.show()


# Region-wise highest transaction

highest_transaction_by_region = merged.groupby('Region')['TotalValue'].max()


highest_transaction_summary = pd.DataFrame({
    'Highest Transaction Value (USD)': highest_transaction_by_region
})


print("\nHighest Transaction Values by Region:")
print(highest_transaction_summary)


highest_transaction_by_region.plot(kind='bar', title='Highest Transaction Values by Region', color='pink')
plt.xlabel('Region')
plt.ylabel('Highest Transaction Value')
plt.show()


merged['TransactionDate'] = pd.to_datetime(merged['TransactionDate'])

# montnly sales
monthly_sales = merged.groupby(merged['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()

monthly_sales.index = monthly_sales.index.strftime('%B')


total_sales = merged['TotalValue'].sum()


monthly_sales_percentage = (monthly_sales / total_sales) * 100


monthly_sales_summary = pd.DataFrame({
    'Total Sales (USD)': monthly_sales,
    'Sales Percentage (%)': monthly_sales_percentage
})

print("\n    Monthly Sales Summary  :    \n")
print(monthly_sales_summary)

 #plotting
monthly_sales.plot(kind='line', title='Monthly Sales Trends', marker='o', color='purple')
plt.xlabel('Month')
plt.ylabel('Revenue (USD)')
plt.grid(True)
plt.show()
