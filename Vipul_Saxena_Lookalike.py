import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Loading and merging the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')


merged = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
print(merged.columns)

# main code 
customer_features = merged.groupby('CustomerID').agg({
    'Region': 'first',             
    'TotalValue': 'sum',          
    'Quantity': 'sum',             
    'Price_x': 'mean',             
    'ProductID': 'nunique',        
}).reset_index()

le_region = LabelEncoder()
customer_features['Region'] = le_region.fit_transform(customer_features['Region'])


scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features[['TotalValue', 'Quantity', 'Price_x', 'ProductID']])


similarity_matrix = cosine_similarity(scaled_features)

# finding looalike for first 20 customers
lookalike_map = {}


for idx, customer_id in enumerate(customer_features['CustomerID'][:20]):
    
    similarity_scores = list(enumerate(similarity_matrix[idx]))
  
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_3 = similarity_scores[1:4]  
    lookalike_map[customer_id] = [
        (customer_features['CustomerID'].iloc[i], round(score, 4)) for i, score in top_3
    ]

# .csv file 
lookalike_df = pd.DataFrame([
    {'CustomerID': cust_id, 'Lookalikes': lookalikes}
    for cust_id, lookalikes in lookalike_map.items()
])

# Save file
lookalike_df.to_csv('Vipul_Saxena_Lookalike.csv', index=False)


print(lookalike_df)