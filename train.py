import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import FlashFloodClassifier

def train_and_evaluate(df, features, target_col='streamflow_cfs', model_path='flash_flood_model.pth', scaler_path='scaler.pkl'):
    # Define labels (flood event = >95th percentile)
    flood_threshold = df[target_col].quantile(0.95)
    y = (df[target_col] > flood_threshold).astype(int)
    X = df[features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Model setup
    model = FlashFloodClassifier(input_dim=X_train_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

    # Save model and scaler
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = (model(X_test_tensor) > 0.5).int().numpy()

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds)
    }
    print("Evaluation:", metrics)
    return model, scaler, metrics