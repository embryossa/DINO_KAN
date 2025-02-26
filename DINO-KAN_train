from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from kan import KAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                             f1_score, accuracy_score)
import re

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clinical_data_path = "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Clincial_annotations.xlsx"
dino_features_path = "dino_features.pt"  # Файл с сохраненными фичами DINO
batch_size = 32
input_size = 1024 + 7  # Размер DINO-фич + 7 клинических признаков

# Загрузка данных
clinical_data = pd.read_excel(clinical_data_path)
clinical_features = [
    'EXP_silver', 'ICM_silver', 'TE_silver',
    'COC', 'MII', 'Age', 'Endo'
]

def load_and_process_data(clinical_data_path, dino_features_path, clinical_features):
    # Загрузка клинических данных
    clinical_data = pd.read_excel(clinical_data_path)

    # Загрузка DINO-фич
    dino_data = torch.load(dino_features_path)

    # Проверка структуры DINO-фич
    if isinstance(dino_data, dict):
        assert 'filenames' in dino_data and 'features' in dino_data, "Некорректная структура DINO-фич!"
        filenames = dino_data['filenames']
        features = dino_data['features']
    else:
        raise ValueError("Неподдерживаемый формат DINO-фич!")

    # Создание DataFrame для DINO-фич
    dino_df = pd.DataFrame({
        'Image': filenames,
        'dino_features': [f.numpy() if isinstance(f, torch.Tensor) else f for f in features]
    })

    # Предобработка имен файлов
    def clean_image_name(name):
        name = str(name).lower().strip()
        name = re.sub(r'\.png$', '', name)
        return re.split(r'[/\\]', name)[-1]  # Извлекаем только имя файла

    clinical_data['Image'] = clinical_data['Image'].apply(clean_image_name)
    dino_df['Image'] = dino_df['Image'].apply(clean_image_name)

    # Визуализация для отладки
    print("\nПример клинических имен после обработки:", clinical_data['Image'].head().values)
    print("Пример DINO-имен после обработки:", dino_df['Image'].head().values)

    # Объединение данных
    full_data = clinical_data.merge(
        dino_df,
        on='Image',
        how='inner'
    )

    # Проверка результатов
    if full_data.empty:
        sample_clin = clinical_data['Image'].head(3).tolist()
        sample_dino = dino_df['Image'].head(3).tolist()
        raise ValueError(
            f"Нет совпадений после объединения!\n"
            f"Пример клинических имен: {sample_clin}\n"
            f"Пример DINO-имен: {sample_dino}"
        )

    # Нормализация клинических данных
    scaler = StandardScaler()
    full_data.loc[:, clinical_features] = scaler.fit_transform(full_data[clinical_features])

    # Фильтрация некорректных значений
    full_data = full_data.dropna(subset=clinical_features + ['HA'])

    print("\nУспешно объединенные данные:")
    print(f"Всего записей: {len(full_data)}")
    print(f"Пример данных:\n{full_data[['Image', 'HA'] + clinical_features].head(3)}")

    return full_data, scaler

class CombinedDataset(Dataset):
    def __init__(self, data):
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            # Преобразование DINO-фич
            dino_feats = row['dino_features']
            if not isinstance(dino_feats, np.ndarray):
                dino_feats = np.array(dino_feats)

            dino_tensor = torch.tensor(dino_feats, dtype=torch.float32)

            # Клинические фичи
            clinical_tensor = torch.tensor(
                row[clinical_features].values.astype(np.float32),
                dtype=torch.float32
            )

            # Проверка размерностей
            if dino_tensor.ndim == 0:
                dino_tensor = dino_tensor.unsqueeze(0)

            combined = torch.cat([dino_tensor.flatten(), clinical_tensor])

            target = torch.tensor(row['HA'], dtype=torch.float32)

            return combined, target

        except Exception as e:
            print(f"Ошибка в обработке изображения {row['Image']}: {str(e)}")
            raise

# Использование
try:
    full_data, scaler = load_and_process_data(
        clinical_data_path="C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Clincial_annotations.xlsx",
        dino_features_path="dino_features.pt",
        clinical_features=['EXP_silver', 'ICM_silver', 'TE_silver', 'COC', 'MII', 'Age', 'Endo']
    )

    # Разделение данных
    train_df, test_df = train_test_split(
        full_data,
        test_size=0.2,
        random_state=42,
        stratify=full_data['HA']
    )

    # Создание DataLoader
    train_loader = DataLoader(
        CombinedDataset(train_df),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        CombinedDataset(test_df),
        batch_size=32,
        num_workers=2,
        pin_memory=True
    )

except Exception as e:
    print(f"Критическая ошибка: {str(e)}")

# Разделение данных с стратификацией
train_df, test_df = train_test_split(
    full_data,
    test_size=0.2,
    random_state=42,
    stratify=full_data['HA']
)

# Создание DataLoader с проверкой батчей
train_loader = DataLoader(
    CombinedDataset(train_df),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    CombinedDataset(test_df),
    batch_size=batch_size,
    drop_last=False
)

# Проверка первого батча
for batch in train_loader:
    features, targets = batch
    print("\nПроверка размерностей:")
    print("Форма фичей:", features.shape)
    print("Форма целей:", targets.shape)
    break



# Модель KAN с дополнительной логикой отслеживания
class PregnancyPredictor(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.kan = KAN(width=[input_size, 10, 5, 1], grid=5, k=3)
        self.train_metrics = {'loss': []}
        self.val_metrics = {'roc_auc': [], 'mcc': [], 'f1': [], 'accuracy': [], 'loss': []}

    def forward(self, x):
        return self.kan(x).squeeze()

model = PregnancyPredictor(input_size).to(device)

# Обучение с прогресс-баром
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_metrics = {'roc_auc': 0}
epochs = 100

with tqdm(total=epochs, desc="Обучение модели", unit="epoch",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar_epoch:

    for epoch in range(epochs):
        # Тренировка
        model.train()
        train_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                  leave=False, unit="batch") as pbar_batch:

            for features, targets in pbar_batch:
                features, targets = features.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar_batch.set_postfix(loss=f"{loss.item():.4f}")

        # Валидация
        model.eval()
        all_probs = []
        all_targets = []
        val_loss = 0

        with torch.no_grad(), tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]",
                                   leave=False, unit="batch") as pbar_val:

            for features, targets in pbar_val:
                features = features.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets.to(device))

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(targets.cpu().numpy())
                val_loss += loss.item()

                pbar_val.set_postfix(val_loss=f"{loss.item():.4f}")

        # Расчет метрик
        roc_auc = roc_auc_score(all_targets, all_probs)
        predictions = (np.array(all_probs) > 0.5).astype(int)

        metrics = {
            'roc_auc': roc_auc,
            'mcc': matthews_corrcoef(all_targets, predictions),
            'f1': f1_score(all_targets, predictions),
            'accuracy': accuracy_score(all_targets, predictions),
            'loss': val_loss / len(test_loader)
        }

        # Сохранение лучшей модели
        if metrics['roc_auc'] > best_metrics['roc_auc']:
            best_metrics = metrics.copy()
            torch.save(model.state_dict(), 'best_model.pth')
            pbar_epoch.write(f"🚀 Новый рекорд ROC AUC: {metrics['roc_auc']:.4f}")

        # Обновление прогресс-бара
        pbar_epoch.set_postfix({
            'Train Loss': f"{train_loss/len(train_loader):.4f}",
            'Val Loss': f"{metrics['loss']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}",
            'F1': f"{metrics['f1']:.4f}"
        })

        # Визуализация метрик
        if (epoch+1) % 5 == 0:
            pbar_epoch.write("\n📊 Метрики эпохи {}:".format(epoch+1))
            pbar_epoch.write("│ Train Loss: {:.4f}".format(train_loss/len(train_loader)))
            pbar_epoch.write("│ Val Loss:    {:.4f}".format(metrics['loss']))
            pbar_epoch.write("├───────────────")
            pbar_epoch.write("│ ROC AUC:     {:.4f}".format(metrics['roc_auc']))
            pbar_epoch.write("│ MCC:         {:.4f}".format(metrics['mcc']))
            pbar_epoch.write("│ F1:          {:.4f}".format(metrics['f1']))
            pbar_epoch.write("└─Accuracy:    {:.4f}\n".format(metrics['accuracy']))

        pbar_epoch.update(1)

# Финальный вывод
print("\n🏆 Лучшие метрики:")
print("├ ROC AUC:     {:.4f}".format(best_metrics['roc_auc']))
print("├ MCC:         {:.4f}".format(best_metrics['mcc']))
print("├ F1:          {:.4f}".format(best_metrics['f1']))
print("└ Accuracy:    {:.4f}".format(best_metrics['accuracy']))
