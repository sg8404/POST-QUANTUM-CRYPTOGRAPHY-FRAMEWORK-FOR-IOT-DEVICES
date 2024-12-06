import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Simulated RF fingerprinting dataset
X_train = np.random.rand(10000, 5)  # Features: amplitude, phase, etc.
y_train = np.random.randint(0, 2, 10000)  # Labels: legitimate (1), unauthorized (0)
X_test = np.random.rand(2000, 5)
y_test = np.random.randint(0, 2, 2000)

# Analyze the dataset
def analyze_dataset(X, y, name):
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    df['Label'] = y
    print(f"{name} Dataset Description:")
    print(df.describe())
    print(f"{name} Label Distribution:\n", df['Label'].value_counts())

    # Correlation heatmap
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"{name} Correlation Heatmap")
    plt.show()

    # Feature distribution histograms
    df.iloc[:, :-1].hist(bins=15, figsize=(15, 10), color='teal')
    plt.suptitle(f"{name} Feature Distributions")
    plt.show()

    # Boxplots for label-wise feature distributions
    for column in df.columns[:-1]:
        sns.boxplot(x='Label', y=column, data=df)
        plt.title(f"{name} - {column} by Label")
        plt.show()

analyze_dataset(X_train, y_train, "Training")

# List of models to evaluate
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Function to train and evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Calculate FAR
        false_acceptances = cm[0, 1]
        total_unauthorized = cm[0, 0] + cm[0, 1]
        far = false_acceptances / total_unauthorized if total_unauthorized > 0 else 0

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "FAR": far
        })

        print(f"\nModel: {name}")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1:.2f}, FAR: {far:.4f}")

    # Convert results to DataFrame and visualize
    results_df = pd.DataFrame(results)
    return results_df

# Evaluate models and display results
results_df = evaluate_models(models, X_train, y_train, X_test, y_test)

# Visualize model performance
results_df.set_index("Model").plot(kind="bar", figsize=(12, 8))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()
