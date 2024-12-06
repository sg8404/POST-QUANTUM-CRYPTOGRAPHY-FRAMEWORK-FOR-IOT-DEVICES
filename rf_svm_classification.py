from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic RF dataset
np.random.seed(42)
num_samples = 10000
num_features = 3  # amplitude, phase, noise

# Simulate features
amplitude = np.random.uniform(0, 1, num_samples)
phase = np.random.uniform(0, 2 * np.pi, num_samples)
noise = np.random.normal(0, 0.1, num_samples)

# Simulate labels (binary classification: 0 or 1)
labels = (amplitude + np.sin(phase) + noise > 1.0).astype(int)

# Combine into dataset
X = np.column_stack((amplitude, phase, noise))
y = labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='rbf', gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
y_probs = svm_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"RF Fingerprinting Classification Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization 1: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix Heatmap")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
plt.yticks(tick_marks, ['Class 0', 'Class 1'])

# Annotate matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center")

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Visualization 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Visualization 3: Feature Distributions
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(amplitude, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title("Amplitude Distribution")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.hist(phase, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title("Phase Distribution")
plt.xlabel("Phase (radians)")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.hist(noise, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.title("Noise Distribution")
plt.xlabel("Noise")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Visualization 4: Decision Boundary (Optional for 2D data)
if num_features == 2:
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
