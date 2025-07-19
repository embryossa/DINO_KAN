import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
import math


class EmbryoDataset(Dataset):
    def __init__(self, image_folder, transform=None, image_size=(224, 224)):
        self.image_paths = sorted(
            sum([glob.glob(os.path.join(image_folder, ext)) for ext in ['*.png', '*.jpg', '*.jpeg']], [])
        )
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)


def patchify(img, patch_size=16):
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h, w = H // patch_size, W // patch_size
    patches = img.reshape(B, C, h, patch_size, w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, -1)
    return patches


def unpatchify(patches, patch_size=16, img_size=224):
    B, N, D = patches.shape
    h = w = int(math.sqrt(N))
    patches = patches.reshape(B, h, w, 3, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, img_size, img_size)
    return patches


def random_masking(x, mask_ratio=0.75):
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked, ids_keep, ids_restore, ids_mask


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=196, encoder_dim=768, decoder_dim=512, patch_dim=768):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8, dim_feedforward=2048)
            for _ in range(8)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x_encoded, ids_restore):
        B, N_encoded, C = x_encoded.shape
        N = ids_restore.shape[1]  # Полное количество патчей (196)

        # Проецируем в пространство декодера
        x_encoded = self.decoder_embed(x_encoded)  # [B, N_encoded, decoder_dim]

        # Создаем маскированные токены
        mask_tokens = self.mask_token.expand(B, N - N_encoded, -1)  # [B, N_masked, decoder_dim]

        # Объединяем видимые и маскированные токены
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)  # [B, N, decoder_dim]

        # Восстанавливаем исходный порядок патчей
        x_unshuffled = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2])
        )

        # Пропускаем через блоки декодера
        x_unshuffled = self.decoder_blocks(x_unshuffled)
        x_unshuffled = self.decoder_norm(x_unshuffled)
        pred = self.decoder_pred(x_unshuffled)
        return pred


# Кастомный энкодер для возврата полной последовательности
class CustomViTEncoder(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self.hidden_dim = vit_model.hidden_dim
        self.patch_size = vit_model.patch_size
        self.image_size = vit_model.image_size
        self.class_token = vit_model.class_token

        # Для torchvision 0.21.0
        self.conv_proj = vit_model.conv_proj
        self.encoder = vit_model.encoder

        # Позиционное кодирование теперь внутри encoder
        self.pos_embedding = vit_model.encoder.pos_embedding

    def forward(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        # Разбиение на патчи
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        # Добавление class token
        class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        # Позиционные эмбеддинги
        if self.pos_embedding is not None:
            x = x + self.pos_embedding

        # Кодирование
        x = self.encoder(x)
        return x


class MAETrainer:
    def __init__(self, encoder, decoder, dataloader, device):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.device = device

        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=800)

    def train(self, epochs=80, patience=5):
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0
            for imgs in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                imgs = imgs.to(self.device)
                patches = patchify(imgs)  # [B, 196, 768]

                # Применяем маскирование
                tokens, ids_keep, ids_restore, ids_mask = random_masking(patches)

                # Пропускаем через энкодер
                x_encoded = self.encoder(imgs)  # [B, 197, 768]

                # Убираем class token и выбираем видимые патчи
                x_encoded_patches = x_encoded[:, 1:, :]  # [B, 196, 768]
                x_visible = torch.gather(
                    x_encoded_patches,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).expand(-1, -1, x_encoded_patches.shape[-1])
                )  # [B, len_keep, 768]

                # Декодируем
                pred = self.decoder(x_visible, ids_restore)  # [B, 196, 768]

                # Вычисляем потерю только на маскированных участках
                mask = torch.zeros(patches.shape[:2], device=patches.device)
                mask.scatter_(1, ids_mask, 1)
                loss = ((pred - patches) ** 2)[mask.bool()].mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

            self.scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save({
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict()
                }, "mae_femi_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break


def main():
    image_folder = "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Images"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = EmbryoDataset(image_folder=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Инициализация кастомного энкодера
    base_vit = vit_b_16(pretrained=True)
    encoder = CustomViTEncoder(base_vit)
    decoder = MAEDecoder(encoder_dim=768, decoder_dim=512, patch_dim=768)

    trainer = MAETrainer(encoder, decoder, dataloader, device)
    trainer.train(epochs=80, patience=5)


if __name__ == "__main__":
    main()
