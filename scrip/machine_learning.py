#!/usr/bin/env python
# coding: utf-8

# In[187]:


import pandas as pd
import numpy as np
import seaborn as sns
from  matplotlib import pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os 
import importlib


# In[188]:


# Loading the datasets

customer = pd.read_csv("data.csv")
purchase = pd.read_csv("purchase.csv")


# In[189]:


#Top 5  records of the customer dataset

customer.head()


# In[191]:


# Top 5  records of the purchase  dataset

purchase.head()


# In[192]:


# print the concise information of the customer dataset

customer.info()


# In[193]:


# Droping the address & avatar columns of the customer dataset

customer_cleaned = customer.drop(columns=["Address","Avatar"])


# In[194]:


# Clean up column names by stripping whitespaces

customer_cleaned.columns = customer_cleaned.columns.str.strip()


# In[195]:


# Handling duplicate records in the email column

customer_cleaned[customer_cleaned.duplicated(subset='Email')].shape[0]


# In[196]:


customer_cleaned


# In[197]:


# Calculate Covariance and Correlation between Time on App & Yearly Amount SPent

covariance = customer_cleaned['Time on App'].cov(customer_cleaned['Yearly Amount Spent'])
correlation = customer_cleaned['Time on App'].corr(customer_cleaned['Yearly Amount Spent'])

covariance


# In[198]:


# Scatter plot showing the relationship between Time on App against Yearly Amount Spent

sns.scatterplot(x='Time on App', y='Yearly Amount Spent', data=customer_cleaned)
plt.title('Time on App vs Yearly Amount Spent')
plt.xlabel('Time on App')
plt.ylabel('Yearly Amount Spent')
plt.show()


# In[199]:


#Calculate Covariance and Correlation between Time on Website  and Yearly Amount SPent

covariance = customer_cleaned['Time on Website'].cov(customer_cleaned['Yearly Amount Spent'])
correlation = customer_cleaned['Time on Website'].corr(customer_cleaned['Yearly Amount Spent'])

covariance


# In[52]:


#Scatter plot showing the relationship between Time on Website  against Yearly Amount Spent
sns.scatterplot(x='Time on Website', y='Yearly Amount Spent', data=customer_cleaned)
plt.title('Time on Website vs Yearly Amount Spent')
plt.xlabel('Time on Website')
plt.ylabel('Yearly Amount Spent')
plt.show()


# In[53]:


#Calculate covariance and correlation between Lenth of Membership   and Yearly Amount SPent

covariance = customer_cleaned['Length of Membership'].cov(customer_cleaned['Yearly Amount Spent'])
correlation = customer_cleaned['Length of Membership'].corr(customer_cleaned['Yearly Amount Spent'])
covariance


# In[201]:


#Scatter plot showing the relationship between Length of Membership  against Yearly Amount Spent

sns.scatterplot(x='Length of Membership', y='Yearly Amount Spent', data=customer_cleaned)
plt.title('Length of Membership vs Yearly Amount Spent')
plt.xlabel('Length of Membership')
plt.ylabel('Yearly Amount Spent')
plt.show()


# In[203]:


# Correlation heatmap for all the numerical features 

correlation_matrix = customer_cleaned[['Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation Matrix')
plt.show()


# In[204]:


# Determing High and Low spenders using summary statistics

yearly_spent = customer_cleaned['Yearly Amount Spent']
mean_spent = yearly_spent.mean()
median_spent = yearly_spent.median()
mode_spent = yearly_spent.mode()[0]  
max_spent = yearly_spent.max()
min_spent = yearly_spent.min()
percentile_90 = np.percentile(yearly_spent, 90)
percentile_10 = np.percentile(yearly_spent, 10)

# Print the results
print(f"Mean Yearly Amount Spent: {mean_spent}")
print(f"Median Yearly Amount Spent: {median_spent}")
print(f"Mode Yearly Amount Spent: {mode_spent}")
print(f"Maximum Yearly Amount Spent: {max_spent}")
print(f"Minimum Yearly Amount Spent: {min_spent}")
print(f"90th Percentile Yearly Amount Spent: {percentile_90}")
print(f"10th Percentile Yearly Amount Spent: {percentile_10}")


# Identify highest spenders ( above 90th percentile)

highest_spenders = customer_cleaned[customer_cleaned['Yearly Amount Spent'] > percentile_90]


# Display the highest spenders

print(f"Highest Spenders (Above 90th Percentile):")
print(highest_spenders)



# Identify lower spenders (Users  below 10th percentile)
lower_spenders = customer_cleaned[customer_cleaned['Yearly Amount Spent'] < percentile_10]

# Display the lower spenders

print("Lower Spenders (Below 10th Percentile):")
print(lower_spenders)


# In[205]:


# Merge the two dataset using the  'Email' column

customer_purchase= pd.merge(customer_cleaned , purchase, on='Email')
customer_purchase


# In[206]:


# Extract features for k-NN (ignore the 'Email')

X = customer_purchase[['Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]
y = customer_purchase['Item Purchased']


# In[207]:


# Normalize the numerical features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[208]:


# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[209]:


# Determing the optimal value of k using range of values( 1 - 21)

k_values = range(1, 21)
cross_val_accuracies = []

# Perform cross-validation for each k

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Using cross-validation with 5 folds
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cross_val_accuracies.append(scores.mean())





# In[210]:


knn


# In[213]:


# Plotting the values of each k against accuracy

plt.figure(figsize=(10, 6))
plt.plot(k_values, cross_val_accuracies, marker='o')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# Finding the best k value

best_k = k_values[cross_val_accuracies.index(max(cross_val_accuracies))]

print(f'Best k value: {best_k}')


# In[214]:


# Recommendation function using KNN with k=5

def recommend_knn(Email, customer_purchase, X_scaled, k=5, n_recommendations=5):
    # Normalize email input
    email = Email.strip().lower()
    
    # Check if the user exists in the dataset
    if email not in customer_purchase['Email'].values:
        return "User not found"
    
    
    # Get the user's data
    user_data = customer_purchase[customer_purchase['Email'] == Email]
    user_features = user_data[['Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]

    
    # Scale the user's features using the same scaler used for the model
    scaler = StandardScaler()
    user_scaled = scaler.fit_transform(user_features)
    
    
    # Fit the KNN model with k=5
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_scaled)
    
    
    # Find the k nearest neighbors
    distances, indices = knn_model.kneighbors(user_scaled)
    
    
    # Get the emails of the nearest neighbors
    neighbor_emails = customer_purchase.iloc[indices[0]]['Email'].values
    
    # Get items purchased by the nearest neighbors
    neighbor_purchases = purchase[purchase['Email'].isin(neighbor_emails)]

    # Recommend items purchased by neighbors, excluding items purchased by the current user
    user_purchases = set(purchase[purchase['Email'] == email]['Item Purchased'])
    recommendations = neighbor_purchases[~neighbor_purchases['Item Purchased'].isin(user_purchases)]
    
    # Return the top N recommendations
    recommended_items = recommendations['Item Purchased'].value_counts().index.tolist()
    return recommended_items[:n_recommendations]


# In[215]:


# Recommendation engine to test a random user in the dataset

email_to_recommend = 'lisagutierrez@johnson-herring.net'
recommendations = recommend_knn(email_to_recommend, customer_purchase, X_scaled, k=5)
print("Recommendations for", email_to_recommend, ":", recommendations)


# In[216]:


#Evaluation of the model using Mean Average Precision

def mean_average_precision(email, customer_purchase, purchase, k=5, n_recommendations=5):
    recommended_items = recommend_knn(email, customer_purchase, X_scaled, k, n_recommendations)
    actual_items = set(purchase[purchase['Email'] == email]['Item Purchased'])
    relevant_items = set(purchase[purchase['Email'] != email]['Item Purchased'])
    

    # Calculate precision at each rank

    precisions = []
    for i, item in enumerate(recommended_items, 1):
        if item in relevant_items and item not in actual_items:
            precision_at_i = len(set(recommended_items[:i]).intersection(relevant_items)) / i
            precisions.append(precision_at_i)

    # Calculate mean of all precisions
    map_score = sum(precisions) / len(precisions) if precisions else 0
    return map_score


# Example usage
map_score = mean_average_precision(email_to_evaluate, customer_purchase, purchase, k=5)
print(f'Mean Average Precision (MAP): {map_score:.2f}')
    


# In[218]:


# Applying the Mean Average Precision to a random user in the dataset

email_to_evaluate = 'lisagutierrez@johnson-herring.net'
map_score = mean_average_precision(email_to_evaluate, customer_purchase, purchase, k=5)
print(f'Mean Average Precision (MAP): {map_score:.2f}')

# Retrieve actual_items and relevant_items for manual verification
def get_actual_and_relevant(email, purchase):
    actual_items = set(purchase[purchase['Email'] == email]['Item Purchased'])
    relevant_items = set(purchase[purchase['Email'] != email]['Item Purchased'])
    return actual_items, relevant_items

# Example usage for retrieval
email_to_evaluate = 'lisagutierrez@johnson-herring.net'
actual_items, relevant_items = get_actual_and_relevant(email_to_evaluate, purchase)

print("Actual Items Purchased by User:", actual_items)
print("Relevant Items (Purchased by Others):", relevant_items)

recommended_items


# In[ ]:





# In[ ]:




