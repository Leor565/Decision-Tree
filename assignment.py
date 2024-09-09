# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:43:41 2024

@author: EliteBook 840
"""

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

data_leor = pd.read_csv("E:/Assignmnet4_supervised/student-por.csv" , delimiter=';')

# a. Check the names and types of columns
print("Column names and types:")
print(data_leor.dtypes)

# b. Check the missing values
print("\nMissing values:")
print(data_leor.isnull().sum())

# c. Check the statistics of the numeric fields
print("\nStatistics of numeric fields:")
print(data_leor.describe())

# d. Check the categorical values
print("\nCategorical values:")
for column in data_leor.select_dtypes(include=['object']).columns:
    print(column + ":")
    print(data_leor[column].value_counts())
    print("\n")
    
# Calculate the total of G1, G2, G3
data_leor['total_score'] = data_leor['G1'] + data_leor['G2'] + data_leor['G3']

# Create the pass_leor column based on the total score
data_leor['pass_leor'] = data_leor['total_score'].apply(lambda x: 1 if x >= 35 else 0)

# Drop the total_score column if you no longer need it
data_leor.drop('total_score', axis=1, inplace=True)

# Print the first few rows to verify the new column
print(data_leor.head())


# Drop the columns G1, G2, G3 permanently
data_leor.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)

# Print the first few rows to verify the changes
print(data_leor.head())


# Separate features and target variable
features_leor = data_leor.drop('pass_leor', axis=1)  # Excluding the target variable
target_variable_leor = data_leor['pass_leor']  # Target variable

# Print the first few rows of features and target variable to verify
print("Features:")
print(features_leor.head())

print("\nTarget Variable:")
print(target_variable_leor.head())

print("Total number of instances in each class:")
print(target_variable_leor.value_counts())

# Create lists for numeric and categorical features
numeric_features_leor = list(features_leor.select_dtypes(include=['int64', 'float64']).columns)
cat_features_leor = list(features_leor.select_dtypes(include=['object']).columns)

# Print the lists to verify
print("Numeric Features:", numeric_features_leor)
print("\nCategorical Features:", cat_features_leor)


# Define transformers for numeric and categorical columns
numeric_transformer = 'passthrough'  # To preserve numeric columns
categorical_transformer = OneHotEncoder(drop='first')  # One-hot encode categorical columns

# Define which columns are categorical and numeric
numeric_features = numeric_features_leor
categorical_features = cat_features_leor

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
transformer_leor = preprocessor.fit_transform(features_leor)

# Print the transformed data shape to verify
print("Transformed Data Shape:", transformer_leor.shape)


from sklearn.tree import DecisionTreeClassifier

# Define the decision tree classifier
clf_leor = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Print the classifier to see the default parameters
print(clf_leor)

pipeline_leor = Pipeline([
    ('preprocessor', preprocessor),  # Column transformer prepared in step 8
    ('classifier', clf_leor)     # Model prepared in step 9
])

# Print the pipeline to verify
#print(pipeline_leor)

seed = 36; 

X_train_leor, X_test_leor, y_train_leor, y_test_leor = train_test_split(
    features_leor,  # Features DataFrame
    target_variable_leor,  # Target variable Series
    test_size=0.2,  # 80% train, 20% test
    random_state=seed  # Seed for reproducibility
)

# Fit the pipeline to the training data
pipeline_leor.fit(X_train_leor, y_train_leor)

# Define StratifiedKFold with shuffling
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(pipeline_leor, X_train_leor, y_train_leor, cv=cv)

# Print the ten cross-validation scores
print("Ten cross-validation scores:", cv_scores)

# Print the mean of the ten cross-validation scores
print("Mean of the ten cross-validation scores:", np.mean(cv_scores))

from sklearn.tree import export_graphviz
import pydotplus
#from IPython.display import Image

# Export the decision tree to a DOT file
dot_data = export_graphviz(clf_leor, out_file=None, 
                           feature_names=preprocessor.get_feature_names_out(),  
                           class_names=['0', '1'],  
                           filled=True, rounded=True,  
                           special_characters=True)

# Create the graph from DOT data
try:
    graph = pydotplus.graph_from_dot_data(dot_data)
except Exception as e:
    print("Error generating graph:", e)
else:
    # Save the decision tree visualization as a PNG file
    try:
        graph.write_png("decision_tree_leor.png")
    except Exception as e:
        print("Error saving PNG file:", e)
    else:
        print("Decision tree visualization saved as 'decision_tree_leor.png' in the current working directory.")
        
# Predict on training set
train_predictions = pipeline_leor.predict(X_train_leor)

# Calculate accuracy on training set
train_accuracy = accuracy_score(y_train_leor, train_predictions)

# Print accuracy on training set
print("Accuracy on training set:", train_accuracy)

# Predict on testing set
test_predictions = pipeline_leor.predict(X_test_leor)

# Calculate accuracy on testing set
test_accuracy = accuracy_score(y_test_leor, test_predictions)

# Print accuracy on testing set
print("Accuracy on testing set:", test_accuracy)

from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
test_predictions = pipeline_leor.predict(X_test_leor)

# Calculate accuracy score
accuracy = accuracy_score(y_test_leor, test_predictions)

# Print accuracy score
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test_leor, test_predictions))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_leor, test_predictions)

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

from sklearn.model_selection import RandomizedSearchCV

# Define parameters for randomized grid search
parameters = {
    'classifier__min_samples_split': range(10, 300, 20),
    'classifier__max_depth': range(1, 30, 2),
    'classifier__min_samples_leaf': range(1, 15, 3)
}

# Set up randomized grid search object
randomized_search = RandomizedSearchCV(
    estimator=pipeline_leor,  # Pipeline object
    param_distributions=parameters,  # Parameter grid
    scoring='accuracy',  # Scoring metric
    cv=5,  # Number of folds for cross-validation
    n_iter=7,  # Number of parameter settings to sample
    refit=True,  # Refit the best estimator
    verbose=3  # Verbosity level
)

# Fit randomized grid search to the data
randomized_search.fit(X_train_leor, y_train_leor)

# Print best parameters and best score
print("Best Parameters:", randomized_search.best_params_)
print("Best Score:", randomized_search.best_score_)

randomized_search.fit(X_train_leor, y_train_leor)

# Print the best parameters
print("Best Parameters:", randomized_search.best_params_)

# Print the score of the model
print("Best Score:", randomized_search.best_score_)

# Print the best estimator
print("Best Estimator:", randomized_search.best_estimator_)

# Use the best estimator to make predictions on the test data
test_predictions = randomized_search.best_estimator_.predict(X_test_leor)

from sklearn.metrics import precision_score, recall_score, accuracy_score

# Calculate precision score
precision = precision_score(y_test_leor, test_predictions)

# Calculate recall score
recall = recall_score(y_test_leor, test_predictions)

# Calculate accuracy score
accuracy = accuracy_score(y_test_leor, test_predictions)

# Print precision, recall, and accuracy scores
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

from joblib import dump

# Save the fine-tuned model (best estimator) as a .pkl file
dump(randomized_search.best_estimator_, 'fine_tuned_model.pkl')

# Save the full pipeline as a .pkl file
dump(pipeline_leor, 'full_pipeline.pkl')