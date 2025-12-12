import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ---------------------------------------
# 1. LOAD TRAIN DATA
# ---------------------------------------
train = pd.read_csv("/kaggle/input/clean-kkbox/clean_kkbox.csv")

print("Train loaded:", train.shape)

features = [
    'bd', 'city', 'song_length', 'language', 
    'registered_via', 'source_system_tab', 
    'source_type', 'source_screen_name'
]

X = train[features]
y = train['target']

# Fix missing values
X = X.fillna(-1)

# ---------------------------------------
# 2. SPLIT + TRAIN RANDOM FOREST
# ---------------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Dummies only on train
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)

# Align columns  
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
proba_valid = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, proba_valid)

print("Validation AUC:", auc)
print(classification_report(y_valid, model.predict(X_valid)))


# ---------------------------------------
# 3. LOAD TEST DATA FOR SUBMISSION
# ---------------------------------------
test = pd.read_csv("/kaggle/input/test-data/test_clean.csv")

print("Test loaded:", test.shape)

test_ids = test['id']  # Required for submission

# Only select features
X_test = test[features]

# Fix missing values
X_test = X_test.fillna(-1)

# Apply dummies
X_test = pd.get_dummies(X_test)

# Align with training columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ---------------------------------------
# 4. PREDICT PROBABILITIES
# ---------------------------------------
test_pred = model.predict_proba(X_test)[:, 1]

# ---------------------------------------
# 5. CREATE SUBMISSION FILE
# ---------------------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "target": test_pred
})

submission.to_csv("submission.csv", index=False)

print("submission.csv generated!")