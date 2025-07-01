import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load CSV
df = pd.read_csv("har_features_pca.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Choose a model: 'rf', 'svm', or 'mlp'
model_type = 'rf'

if model_type == 'rf':
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
elif model_type == 'svm':
    clf = SVC(kernel='rbf', C=1, class_weight='balanced', probability=True, random_state=42)
elif model_type == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)

# Train and evaluate
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
