import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Example pre-trained RF model (Support Vector Machine)
svm_model = SVC(probability=True, random_state=42)
# Note: This is a placeholder. In a real-world scenario, the model should be trained beforehand.

# Define hybrid authentication function
def hybrid_authentication(rf_model, puf_responses, rf_features, puf_challenges):
    rf_result = rf_model.predict(rf_features)
    puf_result = np.all(puf_responses == puf_challenges, axis=1)  # Binary match
    hybrid_result = rf_result & puf_result
    return hybrid_result

# Simulate RF features and PUF challenges
n_samples = 1000
rf_features_test = np.random.random((n_samples, 10))  # Example RF features
puf_challenges_test = np.random.randint(0, 2, (n_samples, 64))
puf_responses_test = puf_challenges_test  # Assuming perfect matching for this simulation
rf_labels = np.random.randint(0, 2, n_samples)  # Simulated ground truth labels for RF model

# Train the RF model
svm_model.fit(rf_features_test, rf_labels)

# Evaluate hybrid model
hybrid_results = hybrid_authentication(svm_model, puf_responses_test, rf_features_test, puf_challenges_test)
hybrid_accuracy = np.mean(hybrid_results) * 100
print(f"Hybrid Model Authentication Accuracy: {hybrid_accuracy:.2f}%")

# Analyze SVM Model
rf_predictions = svm_model.predict(rf_features_test)
rf_probs = svm_model.predict_proba(rf_features_test)[:, 1]

# Classification Report
print("\nClassification Report (RF Model):")
print(classification_report(rf_labels, rf_predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(rf_labels, rf_predictions)
print("\nConfusion Matrix (RF Model):")
print(conf_matrix)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(rf_labels, rf_probs)
roc_auc = auc(fpr, tpr)

# Visualizations
# 1. ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 2. Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (RF Model)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
plt.yticks(tick_marks, ['Class 0', 'Class 1'])

# Add numbers in matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center")

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# 3. Hybrid Model Accuracy Analysis
plt.figure(figsize=(8, 6))
plt.hist(hybrid_results, bins=3, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Hybrid Model Results Distribution")
plt.xlabel("Hybrid Model Output (0=Fail, 1=Pass)")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.show()
