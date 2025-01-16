import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

dataset = pd.read_csv("league_of_legends.csv")
X = dataset.drop("target_column", axis=1)
y = dataset["target_column"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

input_size = X_train.shape[1]
model = LogisticRegressionModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    y_pred_train = (model(X_train_tensor) >= 0.5).float()
    y_pred_test = (model(X_test_tensor) >= 0.5).float()
    train_accuracy = (y_pred_train.eq(y_train_tensor).sum() / len(y_train_tensor)).item()
    test_accuracy = (y_pred_test.eq(y_test_tensor).sum() / len(y_test_tensor)).item()
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
for epoch in range(num_epochs):
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_train = (model(X_train_tensor) >= 0.5).float()
    y_pred_test = (model(X_test_tensor) >= 0.5).float()
    train_accuracy = (y_pred_train.eq(y_train_tensor).sum() / len(y_train_tensor)).item()
    test_accuracy = (y_pred_test.eq(y_test_tensor).sum() / len(y_test_tensor)).item()
print(f"Training Accuracy after L2 Regularization: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy after L2 Regularization: {test_accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred_test.numpy())
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test.numpy()))
fpr, tpr, thresholds = roc_curve(y_test, model(X_test_tensor).numpy())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

torch.save(model.state_dict(), "logistic_regression_model.pth")
loaded_model = LogisticRegressionModel(input_size)
loaded_model.load_state_dict(torch.load("logistic_regression_model.pth"))
loaded_model.eval()
with torch.no_grad():
    loaded_test_predictions = (loaded_model(X_test_tensor) >= 0.5).float()
    loaded_test_accuracy = (loaded_test_predictions.eq(y_test_tensor).sum() / len(y_test_tensor)).item()
print(f"Loaded Model Testing Accuracy: {loaded_test_accuracy * 100:.2f}%")

learning_rates = [0.001, 0.01, 0.1, 0.5]
best_lr = None
best_accuracy = 0
for lr in learning_rates:
    model = LogisticRegressionModel(input_size)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred_test = (model(X_test_tensor) >= 0.5).float()
        test_accuracy = (y_pred_test.eq(y_test_tensor).sum() / len(y_test_tensor)).item()
    print(f"Learning Rate: {lr}, Test Accuracy: {test_accuracy * 100:.2f}%")
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_lr = lr
print(f"Best Learning Rate: {best_lr}, Test Accuracy: {best_accuracy * 100:.2f}%")

feature_importance = model.linear.weight.detach().numpy().flatten()
feature_names = X.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()