import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset

customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')


merged = pd.merge(transactions, customers, on='CustomerID')

print("Merged Dataset:")
print(merged.head())


le = LabelEncoder()
merged['Region'] = le.fit_transform(merged['Region'])


customer_data = merged.groupby('CustomerID').agg({
    'Region': 'first',         
    'TotalValue': 'sum',        
    'Quantity': 'sum'           
}).reset_index()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['TotalValue', 'Quantity']])


customer_data[['TotalValue_Scaled', 'Quantity_Scaled']] = scaled_data


print("Preprocessed Data:")
print(customer_data.head())

db_scores = []
kmeans_models = {}


customer_data['Cluster'] = None

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_data[f'Cluster_{n_clusters}'] = kmeans.fit_predict(scaled_data)
    
 
    kmeans_models[n_clusters] = kmeans
    
    
    db_index = davies_bouldin_score(scaled_data, kmeans.labels_)
    db_scores.append((n_clusters, db_index))
    print(f"Number of Clusters: {n_clusters}, DB Index: {db_index:.4f}")


db_scores_df = pd.DataFrame(db_scores, columns=['Number of Clusters', 'DB Index'])


print("\nDB Scores for Different Clusters:")
print(db_scores_df)


optimal_clusters = db_scores_df.loc[db_scores_df['DB Index'].idxmin(), 'Number of Clusters']
print(f"\nOptimal Number of Clusters: {optimal_clusters}")


kmeans_optimal = kmeans_models[int(optimal_clusters)]
customer_data['Cluster'] = kmeans_optimal.labels_

# Scatter plot 
plt.figure(figsize=(10, 6))
plt.scatter(
    customer_data['TotalValue_Scaled'],
    customer_data['Quantity_Scaled'],
    c=customer_data['Cluster'],
    cmap='viridis',
    s=50
)
plt.title(f'Customer Segmentation (KMeans, {optimal_clusters} Clusters)')
plt.xlabel('TotalValue (Scaled)')
plt.ylabel('Quantity (Scaled)')
plt.colorbar(label='Cluster')
plt.show()

# Correlation Matrix
correlation_matrix = customer_data[['TotalValue', 'Quantity', 'Region']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Boxplot
sns.boxplot(x='Cluster', y='TotalValue', data=customer_data)
plt.title('Distribution of TotalValue by Cluster')
plt.show()

# Save file
output_file = 'Vipul_Saxena_Clustering.csv'
customer_data.to_csv(output_file, index=False)
print(f"Clustering results saved to {output_file}")
