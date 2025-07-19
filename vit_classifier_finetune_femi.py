import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Пути к данным
PATHS = {
    "images_dir": "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Images",
    "metadata": "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Clincial_annotations.xlsx"
}

# Классы для работы с ViT
class CustomViTEncoder(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self.hidden_dim = vit_model.hidden_dim
        self.patch_size = vit_model.patch_size
        self.image_size = vit_model.image_size
        self.class_token = vit_model.class_token
        self.conv_proj = vit_model.conv_proj
        self.encoder = vit_model.encoder
        self.pos_embedding = vit_model.encoder.pos_embedding

    def forward(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        if self.pos_embedding is not None:
            x = x + self.pos_embedding

        x = self.encoder(x)
        return x

class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder

        # Замораживаем энкодер на начальных этапах
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Получаем выход энкодера [batch_size, 197, 768]
        features = self.encoder(x)

        # Берем class token (первый токен) для классификации
        cls_token = features[:, 0, :]

        return self.classifier(cls_token)

    def unfreeze_encoder(self, last_n_layers=3):
        """Размораживаем последние слои энкодера"""
        total_layers = len(self.encoder.encoder.layers)

        for i, layer in enumerate(self.encoder.encoder.layers):
            if i >= total_layers - last_n_layers:
                for param in layer.parameters():
                    param.requires_grad = True

# Класс датасета с сохранением меток в самом датасете
class EmbryoExcelDataset(Dataset):
    def __init__(self, images_dir, metadata_path, transform=None):
        """
        images_dir: папка с изображениями
        metadata_path: путь к Excel-файлу с метаданными
        """
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.samples = []  # Будет хранить кортежи (путь_к_изображению, метка)

        # Загружаем метаданные из Excel
        metadata = pd.read_excel(metadata_path)
        print(f"Loaded metadata with {len(metadata)} rows")

        # Проверяем наличие необходимых столбцов
        if 'HA' not in metadata.columns:
            raise ValueError("Metadata does not contain 'HA' column")

        # Определяем столбец с именами файлов
        image_col = None
        for col in ['Image', 'Filename', 'File', 'Image Name']:
            if col in metadata.columns:
                image_col = col
                break

        if image_col is None:
            raise ValueError("Metadata does not contain image filename column")

        print(f"Using '{image_col}' column for image filenames")

        # Собираем существующие изображения
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_images.extend(glob.glob(os.path.join(images_dir, ext)))
        print(f"Found {len(all_images)} images in directory")

        # Создаем список образцов
        for _, row in metadata.iterrows():
            filename = row[image_col]
            label = row['HA']

            # Пропускаем невалидные метки
            if pd.isna(label) or label not in [0, 1]:
                continue

            # Пробуем найти соответствующее изображение
            found = False
            for img_path in all_images:
                img_filename = os.path.basename(img_path)

                # Проверяем совпадение с именем файла или именем без расширения
                if filename == img_filename or filename == os.path.splitext(img_filename)[0]:
                    self.samples.append((img_path, int(label)))
                    found = True
                    break

            if not found:
                print(f"Warning: Image not found for {filename}")

        print(f"Created dataset with {len(self.samples)} samples")

        # Собираем метки для анализа распределения
        self.labels = [label for _, label in self.samples]
        if self.labels:
            class_counts = np.bincount(self.labels)
            print(f"Class distribution: {class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def train_classifier(model, train_loader, val_loader, device, train_labels, epochs=30, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Рассчитываем веса классов для дисбаланса
    if train_labels:
        class_counts = np.bincount(train_labels)
        class_weights = torch.tensor([
            len(train_labels) / (2 * class_counts[0]),
            len(train_labels) / (2 * class_counts[1])
        ], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0

    for epoch in range(epochs):
        # Тренировка
        model.train()
        train_loss = 0
        train_total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)

        # Валидация - собираем предсказания для расчета AUC
        model.eval()
        val_loss = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Получаем вероятности для класса 1
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Рассчитываем AUC
        if len(np.unique(all_labels)) >= 2:
            val_auc = roc_auc_score(all_labels, all_probs)
        else:
            val_auc = 0.5  # Если только один класс в валидации

        # Рассчитываем точность для информации
        predictions = [1 if p > 0.5 else 0 for p in all_probs]
        val_acc = np.mean(np.array(predictions) == np.array(all_labels))

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        # Сохраняем лучшую модель по AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "MAE_vit_classifier.pth")
            print(f"  🏆 New best model saved with val AUC: {val_auc:.4f}")

        # После 5 эпох размораживаем часть энкодера
        if epoch == 5:
            model.unfreeze_encoder(last_n_layers=3)
            print("  🔓 Unfreezing last 3 encoder layers")

    print("✅ Обучение завершено!")

def main():
    # Конфигурация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем датасет
    dataset = EmbryoExcelDataset(
        images_dir=PATHS["images_dir"],
        metadata_path=PATHS["metadata"],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    )

    # Собираем все метки для стратификации
    all_labels = [label for _, label in dataset]

    # Разделение на train/val (80/20)
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    # Создаем подмножества
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Собираем метки для тренировочного набора
    train_labels = [all_labels[i] for i in train_indices]

    # Создаем сэмплер для балансировки классов
    if train_labels:
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices) * 2,  # Увеличиваем выборку
            replacement=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Создаем и загружаем модель
    base_vit = vit_b_16(pretrained=False)
    base_vit.heads = nn.Identity()

    encoder = CustomViTEncoder(base_vit)

    # Загружаем предобученные веса из MAE
    checkpoint = torch.load("mae_femi_best.pth", map_location=device)

    # Удаляем ключи, связанные с головой классификации
    encoder_state_dict = {}
    for key, value in checkpoint["encoder"].items():
        if 'vit.heads.head' not in key:
            encoder_state_dict[key] = value

    # Загружаем отфильтрованный state_dict
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    model = ViTClassifier(encoder=encoder, num_classes=2)

    # Обучаем классификатор, передавая метки тренировочного набора
    train_classifier(
        model,
        train_loader,
        val_loader,
        device,
        train_labels=train_labels,
        epochs=30,
        lr=1e-4
    )

if __name__ == "__main__":
    main()
