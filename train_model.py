import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных (предполагается, что данные уже подготовлены)
extracted_features, filenames = torch.load("C:/Users/User/Desktop/IVF/AI/Extracted_Features/extracted_features.pt")
labels_df = pd.read_csv("C:/Users/User/PycharmProjects/pythonProject/KAN/Blastocyst/image_labels.csv")
labels = torch.tensor(labels_df['label'].values)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(extracted_features, labels, test_size=0.2, random_state=42)

# Преобразование данных в тензоры и создание DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Определение модели KAN
class KAN(nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(width[0], width[1])
        self.fc2 = nn.Linear(width[1], width[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

kan_model = KAN(width=[X_train.shape[1], 10, 1], grid=5, k=3)

# Определение функции потерь и оптимизатора
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(kan_model.parameters(), lr=0.001)

# Функция для обучения модели с сохранением истории потерь и точности
def train_model_with_accuracy(model, criterion, optimizer, train_loader, test_loader, epochs=10):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        model.train()

        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Вычисление точности на обучающем наборе
            probs = torch.sigmoid(outputs)
            preds = probs.round()
            correct_train += (preds == labels.unsqueeze(1)).sum().item()
            total_train += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # Оценка на тестовом наборе
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        model.eval()
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = model(data)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                test_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = probs.round()
                correct_test += (preds == labels.unsqueeze(1)).sum().item()
                total_test += labels.size(0)

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(correct_test / total_test)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies

# Обучение модели
train_losses, test_losses, train_accuracies, test_accuracies = train_model_with_accuracy(kan_model, loss_fn, optimizer, train_loader, test_loader, epochs=30)

# Функция для построения ROC-AUC и PRC кривых
def plot_roc_prc_curves(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    prc_auc = auc(recall, precision)

    # Построение графиков
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # ROC-AUC кривая
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axs[0].plot(fpr, tpr, label=f"ROC-AUC: {roc_auc:.4f}")
    axs[0].set_title("ROC-AUC Curve")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend()

    # PRC кривая
    axs[1].plot(recall, precision, label=f"PRC AUC: {prc_auc:.4f}")
    axs[1].set_title("Precision-Recall Curve")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend()

    plt.show()

# Построение графиков потерь и точности на одном рисунке
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# График потерь
axs[0].plot(train_losses, label='Train Loss')
axs[0].plot(test_losses, label='Test Loss')
axs[0].set_title('Loss Over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# График точности
axs[1].plot(train_accuracies, label='Train Accuracy')
axs[1].plot(test_accuracies, label='Test Accuracy')
axs[1].set_title('Accuracy Over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.show()

# Расчёт метрик
all_labels = []
all_probs = []
all_preds = []

# Получаем прогнозы на тестовом наборе
with torch.no_grad():
    for data, labels in test_loader:
        outputs = kan_model(data)
        probs = torch.sigmoid(outputs)
        all_probs.extend(probs.cpu().numpy())
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Преобразуем списки в массивы
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)

# Расчёт метрик
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)
mcc = matthews_corrcoef(all_labels, all_preds)

# Вывод метрик
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"MCC: {mcc:.4f}")

# Построение ROC-AUC и PRC кривых
plot_roc_prc_curves(kan_model, test_loader)

# Сохранение модели
torch.save(kan_model.state_dict(), 'kan_image_model.pth')
