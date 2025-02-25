import torch
import torch.nn as nn
import pytorch_lightning as pl
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# Конфигурация путей
PATHS = {
    "images_dir": "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Images",
    "metadata": "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Clincial_annotations.xlsx",
    "save_dir": "C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst"
}

# Параметры эксперимента
CFG = {
    "img_size": (512, 384),  # Соответствует вашим изображениям 512x384
    "batch_size": 32,
    "num_workers": 4,
    "lr": 3e-4,
    "max_epochs": 50,
    "test_size": 0.2,
    "random_state": 42
}

class EmbryoDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform or self.default_transform()

        # Проверка существования файлов
        self._validate_files()

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize(CFG['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _validate_files(self):
        missing = []
        for img_name in self.df['Image']:
            path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(path):
                missing.append(path)
        if missing:
            raise FileNotFoundError(f"Отсутствует {len(missing)} изображений. Пример: {missing[0]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['Image'])
        img = Image.open(img_path).convert('RGB')

        # Основные таргеты
        morphology = torch.tensor(row['Fond'] - 1, dtype=torch.long)  # Приводим к 0-4
        live_birth = torch.tensor(row['LB'], dtype=torch.float)

        # Дополнительные клинические признаки
        clinical_features = torch.tensor([
            row['EXP_silver'],
            row['ICM_silver'],
            row['TE_silver'],
            row['COC'],
            row['MII'],
            row['Age'],
            row['Endo']
        ], dtype=torch.float32)

        return {
            'image': self.transform(img),
            'morphology': morphology,
            'live_birth': live_birth,
            'clinical': clinical_features
        }

class EmbryoModel(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = SwinTransformer(
            img_size=CFG['img_size'],
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            num_classes=0  # Без классификатора
        )

        # Мультизадачные головы
        self.morphology_head = nn.Linear(1024, num_classes)
        self.live_birth_head = nn.Sequential(
            nn.Linear(1024 + 7, 256),  # 7 клинических признаков
            nn.ReLU(),
            nn.Linear(256, 1))

        self.loss_morph = nn.CrossEntropyLoss()
        self.loss_birth = nn.BCEWithLogitsLoss()

    def forward(self, x, clinical=None):
        features = self.backbone(x)
        morph_logits = self.morphology_head(features)

        if clinical is not None:
            birth_input = torch.cat([features, clinical], dim=1)
            birth_pred = self.live_birth_head(birth_input)
            return morph_logits, birth_pred

        return morph_logits

    def training_step(self, batch, batch_idx):
        images = batch['image']
        morph_targets = batch['morphology']
        birth_targets = batch['live_birth']
        clinical = batch['clinical']

        morph_preds, birth_preds = self(images, clinical)

        loss_morph = self.loss_morph(morph_preds, morph_targets)
        loss_birth = self.loss_birth(birth_preds.squeeze(), birth_targets)
        total_loss = 0.6*loss_morph + 0.4*loss_birth

        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=CFG['lr'])

# Загрузка и подготовка данных
def prepare_datasets():
    # Загрузка метаданных
    df = pd.read_excel(PATHS['metadata'])

    # Фильтрация только размеченных данных
    labeled_df = df.dropna(subset=['LB', 'Fond'])

    # Разделение на train/test
    train_df, test_df = train_test_split(
        labeled_df,
        test_size=CFG['test_size'],
        random_state=CFG['random_state'],
        stratify=labeled_df['LB']  # Стратификация по live birth
    )

    # Создание датасетов
    train_ds = EmbryoDataset(train_df, PATHS['images_dir'])
    test_ds = EmbryoDataset(test_df, PATHS['images_dir'])

    return train_ds, test_ds

# Инициализация и обучение
def main():
    train_ds, test_ds = prepare_datasets()

    model = EmbryoModel()

    trainer = pl.Trainer(
        max_epochs=CFG['max_epochs'],
        accelerator='auto',
        devices=1,
        default_root_dir=PATHS['save_dir'],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='train_loss',
                dirpath=PATHS['save_dir'],
                filename='best_model_{epoch}_{train_loss:.2f}',
                save_top_k=3
            )
        ]
    )

    trainer.fit(
        model,
        DataLoader(train_ds,
                   batch_size=CFG['batch_size'],
                   shuffle=True,
                   num_workers=CFG['num_workers']),
        DataLoader(test_ds,
                   batch_size=CFG['batch_size'],
                   num_workers=CFG['num_workers'])
    )

if __name__ == "__main__":
    main()
