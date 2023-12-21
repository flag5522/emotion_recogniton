import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=pd.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='weighted', labels=pd.unique(y_pred))
    f1 = f1_score(y_true, y_pred, average='weighted', labels=pd.unique(y_pred))
    
    # Check if binary or multiclass classification
    if len(pd.unique(y_true)) > 2:
        auc_roc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average='weighted', multi_class='ovr')
    else:
        auc_roc = roc_auc_score(y_true_encoded, y_pred_encoded, average='weighted', multi_class='ovr')

    return accuracy, precision, recall, f1, auc_roc

# Step 1: Load and Combine Data
happiness_data = pd.read_csv('h_combined_data.csv')
sadness_data = pd.read_csv('s_combined_data.csv')
anger_data = pd.read_csv('a_combined_data.csv')

# Add the "Emotion" column to each dataset
happiness_data["Emotion"] = "Happiness"
sadness_data["Emotion"] = "Sadness"
anger_data["Emotion"] = "Anger"

# Combine the data from different emotions
all_data = pd.concat([happiness_data, sadness_data, anger_data], ignore_index=True)

# Mask 'F1' and 'F2' values
columns_to_mask = ['F1 Values', 'F2 Values']
all_data[columns_to_mask] = 'Masked'

# Extract features and target variable
X = all_data.drop(["Emotion", 'F1 Values', 'F2 Values'], axis=1)  # Exclude 'F1 Values' and 'F2 Values'
y = all_data["Emotion"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Train SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train_imputed, y_train)

# Save the trained SVM model
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the imputer
with open('imputer.pkl', 'wb') as imputer_file:
    pickle.dump(imputer, imputer_file)

# Save the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
with open('label_encoder.pkl', 'wb') as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_imputed)

# Evaluate the model
accuracy, precision, recall, f1, auc_roc = evaluate_model(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC Score: {auc_roc:.2f}")
