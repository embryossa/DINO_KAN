import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchcam.methods import ScoreCAM
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from torchvision.transforms import InterpolationMode
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau
from skimage.metrics import structural_similarity as ssim
import random
import warnings
warnings.filterwarnings('ignore')

# ============================= МОДЕЛИ =============================

class ClinicalMLP(nn.Module):
    def __init__(self, input_dim=4, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.model(x)

class ImageCNN(nn.Module):
    def __init__(self, fine_tune=True):
        super().__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        for param in self.resnet.parameters():
            param.requires_grad = fine_tune

        original_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.feature_extractor = nn.Sequential(
            nn.Linear(original_in_features, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.feature_extractor(self.resnet(x))

class FusionModel(nn.Module):
    def __init__(self, num_classes=2, fine_tune_cnn=False):
        super(FusionModel, self).__init__()
        self.image_branch = ImageCNN(fine_tune=fine_tune_cnn)
        self.clinical_branch = ClinicalMLP()
        self.classifier = nn.Linear(512 + 1024, num_classes)

    def forward(self, img, clin):
        return self.classifier(
            torch.cat([
                self.image_branch(img),
                self.clinical_branch(clin)
            ], dim=1)
        )

class ViTEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ViTEmbryoClassifier, self).__init__()
        self.vit = models.vit_b_16(pretrained=pretrained)
        self.hidden_dim = self.vit.heads[0].in_features

        self.vit.heads = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

    def forward_with_attention(self, x):
        """Прямой проход с извлечением attention weights"""
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        attention_weights = []

        for layer in self.vit.encoder.layers:
            x_norm = layer.ln_1(x)

            # Получаем attention через MultiheadAttention
            attn_output, attn_weights = layer.self_attention(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )

            attention_weights.append(attn_weights.detach())
            x = x + layer.dropout(attn_output)
            x = x + layer.mlp(layer.ln_2(x))

        x = self.vit.encoder.ln(x)
        logits = self.vit.heads(x[:, 0])

        return logits, attention_weights

    def get_features(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]

# ============================= ВИЗУАЛИЗАЦИЯ =============================

class ModelComparator:
    def __init__(self, cnn_model_path, vit_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        # Загрузка моделей
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        self.vit_model = self._load_vit_model(vit_model_path)

        # Базовые трансформации
        self.base_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Список для сохранения аугментированных изображений
        self.augmented_samples = []

    def _load_cnn_model(self, model_path):
        """Загрузка CNN модели"""
        model = FusionModel().to(self.device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print("✅ CNN модель загружена успешно")
            return model
        except Exception as e:
            print(f"❌ Ошибка загрузки CNN модели: {e}")
            return None

    def _load_vit_model(self, model_path):
        """Загрузка ViT модели"""
        model = ViTEmbryoClassifier(num_classes=2).to(self.device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print("✅ ViT модель загружена успешно")
            return model
        except Exception as e:
            print(f"❌ Ошибка загрузки ViT модели: {e}")
            return None

    def _apply_random_transforms(self, image):
        """Применение случайных трансформаций к изображению"""
        # Случайное вращение
        rotation = random.uniform(-20, 20)
        image = image.rotate(rotation, resample=Image.BICUBIC)

        # Случайное масштабирование
        scale = random.uniform(0.8, 1.2)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, resample=Image.BICUBIC)

        # Случайный сдвиг
        max_offset_x = int(image.width * 0.1)
        max_offset_y = int(image.height * 0.1)
        offset_x = random.randint(-max_offset_x, max_offset_x)
        offset_y = random.randint(-max_offset_y, max_offset_y)
        image = image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, offset_x, 0, 1, offset_y),
            resample=Image.BICUBIC
        )

        # Случайное изменение яркости и контраста
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Случайное размытие
        if random.random() < 0.4:
            blur_radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        # Случайное добавление шума
        if random.random() < 0.3:
            img_array = np.array(image) / 255.0
            noise = np.random.normal(0, 0.05, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 1) * 255
            image = Image.fromarray(noisy_img.astype(np.uint8))

        # Случайное отражение
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Применение базовых трансформаций
        return self.base_transform(image)

    def _denormalize(self, tensor):
        """Денормализация изображения для визуализации"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    def get_cnn_heatmap(self, image_tensor):
        """Получение тепловой карты для CNN"""
        if self.cnn_model is None:
            return None

        # Фиктивные клинические данные для CNN
        clinical_params = torch.tensor([1.0, 1.0, 30.0, 10.0], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cnn_model(image_tensor, clinical_params)
            pred_class = torch.argmax(logits).item()

        # Создание ScoreCAM
        cam_extractor = ScoreCAM(self.cnn_model.image_branch.resnet, 'layer4')

        with torch.no_grad():
            _ = self.cnn_model.image_branch(image_tensor)

        activation = cam_extractor(pred_class, logits)[0].cpu().squeeze().numpy()
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)

        return activation, pred_class

    def get_vit_heatmap(self, image_tensor):
        """Получение карты внимания для ViT"""
        if self.vit_model is None:
            return None

        with torch.no_grad():
            try:
                logits, attention_weights = self.vit_model.forward_with_attention(image_tensor)
                pred_class = torch.argmax(logits).item()

                # Берем последний слой внимания
                last_attention = attention_weights[-1]  # [B, num_heads, N, N]

                # Усредняем по головам внимания
                attention_map = last_attention[0].mean(dim=0)  # [N, N]

                # CLS token attention к патчам
                cls_attention = attention_map[0, 1:].cpu().numpy()

                # Преобразуем в 2D карту
                num_patches = len(cls_attention)
                patch_size = int(np.sqrt(num_patches))

                if patch_size * patch_size == num_patches:
                    attention_2d = cls_attention.reshape(patch_size, patch_size)
                    # Нормализация
                    attention_2d = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min() + 1e-8)
                    return attention_2d, pred_class
                else:
                    print(f"Предупреждение: неквадратное количество патчей ({num_patches})")
                    return None

            except Exception as e:
                print(f"Ошибка получения ViT карты внимания: {e}")
                return None

    def compare_single_image(self, image_path, num_runs=10):
        """Сравнение моделей на одном изображении с трансформациями"""
        print(f"\n🔍 Анализ изображения: {image_path}")

        # Загрузка оригинального изображения
        original_image = Image.open(image_path).convert('RGB')

        # Сохранение карт для анализа стабильности
        cnn_maps = []
        vit_maps = []
        cnn_predictions = []
        vit_predictions = []
        self.augmented_samples = []  # Сброс списка аугментированных изображений

        print(f"Выполняется {num_runs} прогонов со случайными трансформациями...")

        for run in range(num_runs):
            # Применение случайных трансформаций
            augmented_image = self._apply_random_transforms(original_image)

            # Сохранение для визуализации (денормализованная версия)
            denorm_img = self._denormalize(augmented_image).permute(1, 2, 0).numpy()
            self.augmented_samples.append(denorm_img)

            image_tensor = augmented_image.unsqueeze(0).to(self.device)

            # CNN
            cnn_result = self.get_cnn_heatmap(image_tensor)
            if cnn_result:
                cnn_heatmap, cnn_pred = cnn_result
                cnn_maps.append(cnn_heatmap)
                cnn_predictions.append(cnn_pred)

            # ViT
            vit_result = self.get_vit_heatmap(image_tensor)
            if vit_result:
                vit_heatmap, vit_pred = vit_result
                vit_maps.append(vit_heatmap)
                vit_predictions.append(vit_pred)

            print(f"Прогон {run+1}/{num_runs} завершен")

        # Анализ результатов
        results = self._analyze_stability(cnn_maps, vit_maps, cnn_predictions, vit_predictions)

        # Визуализация
        self._visualize_comparison(original_image, cnn_maps, vit_maps, results)

        return results

    def _calculate_intra_ssim(self, maps):
        """Вычисление среднего SSIM между картами внутри модели"""
        ssim_values = []
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                ssim_val = ssim(maps[i], maps[j], data_range=1.0)
                ssim_values.append(ssim_val)
        return np.mean(ssim_values) if ssim_values else 0

    def _calculate_intra_spearman(self, maps):
        """Вычисление средней ранговой корреляции Спирмена"""
        corr_values = []
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                corr, _ = spearmanr(maps[i].flatten(), maps[j].flatten())
                corr_values.append(corr)
        return np.mean(corr_values) if corr_values else 0

    def _calculate_intra_kendall(self, maps):
        """Вычисление средней ранговой корреляции Кендалла"""
        tau_values = []
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                tau, _ = kendalltau(maps[i].flatten(), maps[j].flatten())
                tau_values.append(tau)
        return np.mean(tau_values) if tau_values else 0

    def _calculate_iou(self, map1, map2, threshold):
        """Вычисление IoU для бинаризованных карт"""
        binary1 = (map1 > threshold).astype(np.uint8)
        binary2 = (map2 > threshold).astype(np.uint8)

        intersection = np.logical_and(binary1, binary2).sum()
        union = np.logical_or(binary1, binary2).sum()

        return intersection / (union + 1e-8)

    def _analyze_stability(self, cnn_maps, vit_maps, cnn_preds, vit_preds):
        """Анализ стабильности моделей с новыми метриками"""
        results = {}
        cnn_mean = vit_mean = None

        if cnn_maps:
            # Анализ CNN
            cnn_maps_array = np.array(cnn_maps)
            cnn_mean = np.mean(cnn_maps_array, axis=0)
            cnn_std = np.std(cnn_maps_array, axis=0)

            # Метрики стабильности CNN
            results['cnn_mean_std'] = np.mean(cnn_std)
            results['cnn_max_std'] = np.max(cnn_std)
            results['cnn_pred_stability'] = len(set(cnn_preds)) / len(cnn_preds) if cnn_preds else 0

            # Попарные корреляции между прогонами CNN
            cnn_correlations = []
            for i in range(len(cnn_maps)):
                for j in range(i+1, len(cnn_maps)):
                    corr, _ = pearsonr(cnn_maps[i].flatten(), cnn_maps[j].flatten())
                    cnn_correlations.append(corr)
            results['cnn_mean_correlation'] = np.mean(cnn_correlations) if cnn_correlations else 0
            results['cnn_std_correlation'] = np.std(cnn_correlations) if cnn_correlations else 0

            # MSE между прогонами CNN
            cnn_mse_values = []
            for i in range(len(cnn_maps)):
                for j in range(i+1, len(cnn_maps)):
                    mse = mean_squared_error(cnn_maps[i].flatten(), cnn_maps[j].flatten())
                    cnn_mse_values.append(mse)
            results['cnn_mean_mse'] = np.mean(cnn_mse_values) if cnn_mse_values else 0
            results['cnn_std_mse'] = np.std(cnn_mse_values) if cnn_mse_values else 0

            # Новые метрики для CNN
            results['cnn_mean_ssim'] = self._calculate_intra_ssim(cnn_maps)
            results['cnn_spearman'] = self._calculate_intra_spearman(cnn_maps)
            results['cnn_kendall'] = self._calculate_intra_kendall(cnn_maps)

        if vit_maps:
            # Анализ ViT
            vit_maps_array = np.array(vit_maps)
            vit_mean = np.mean(vit_maps_array, axis=0)
            vit_std = np.std(vit_maps_array, axis=0)

            # Метрики стабильности ViT
            results['vit_mean_std'] = np.mean(vit_std)
            results['vit_max_std'] = np.max(vit_std)
            results['vit_pred_stability'] = len(set(vit_preds)) / len(vit_preds) if vit_preds else 0

            # Попарные корреляции между прогонами ViT
            vit_correlations = []
            for i in range(len(vit_maps)):
                for j in range(i+1, len(vit_maps)):
                    corr, _ = pearsonr(vit_maps[i].flatten(), vit_maps[j].flatten())
                    vit_correlations.append(corr)
            results['vit_mean_correlation'] = np.mean(vit_correlations) if vit_correlations else 0
            results['vit_std_correlation'] = np.std(vit_correlations) if vit_correlations else 0

            # MSE между прогонами ViT
            vit_mse_values = []
            for i in range(len(vit_maps)):
                for j in range(i+1, len(vit_maps)):
                    mse = mean_squared_error(vit_maps[i].flatten(), vit_maps[j].flatten())
                    vit_mse_values.append(mse)
            results['vit_mean_mse'] = np.mean(vit_mse_values) if vit_mse_values else 0
            results['vit_std_mse'] = np.std(vit_mse_values) if vit_mse_values else 0

            # Новые метрики для ViT
            results['vit_mean_ssim'] = self._calculate_intra_ssim(vit_maps)
            results['vit_spearman'] = self._calculate_intra_spearman(vit_maps)
            results['vit_kendall'] = self._calculate_intra_kendall(vit_maps)

        # Межмодельное сравнение
        if cnn_mean is not None and vit_mean is not None:
            # Приведение к единому размеру
            target_size = (224, 224)
            cnn_norm = cv2.resize(cnn_mean, target_size)
            vit_norm = cv2.resize(vit_mean, target_size)

            # Нормализация
            cnn_norm = (cnn_norm - cnn_norm.min()) / (cnn_norm.max() - cnn_norm.min() + 1e-8)
            vit_norm = (vit_norm - vit_norm.min()) / (vit_norm.max() - vit_norm.min() + 1e-8)

            # Ранговые корреляции
            results['cross_spearman'] = spearmanr(cnn_norm.flatten(), vit_norm.flatten())[0]
            results['cross_kendall'] = kendalltau(cnn_norm.flatten(), vit_norm.flatten())[0]

            # IoU с адаптивным порогом
            threshold = np.percentile(np.concatenate([cnn_norm.flatten(), vit_norm.flatten()]), 75)
            results['cross_iou'] = self._calculate_iou(cnn_norm, vit_norm, threshold)

            # SSIM между моделями
            results['cross_ssim'] = ssim(cnn_norm, vit_norm, data_range=1.0)

        return results

    def _visualize_comparison(self, original_image, cnn_maps, vit_maps, results):
        """Визуализация сравнения с аугментациями"""
        fig = plt.figure(figsize=(24, 18))
        plt.suptitle("Анализ устойчивости моделей к вариациям изображений", fontsize=18, fontweight='bold')

        # Оригинальное изображение
        plt.subplot(3, 4, 1)
        plt.imshow(original_image)
        plt.title('Оригинальное изображение', fontsize=12)
        plt.axis('off')

        # Примеры аугментированных изображений
        if self.augmented_samples:
            for i in range(min(3, len(self.augmented_samples))):
                plt.subplot(3, 4, i+2)
                plt.imshow(self.augmented_samples[i])
                plt.title(f'Аугментация {i+1}', fontsize=12)
                plt.axis('off')

        if cnn_maps:
            cnn_mean = np.mean(cnn_maps, axis=0)
            cnn_std = np.std(cnn_maps, axis=0)

            # Масштабирование карт CNN до 224x224
            cnn_mean_resized = cv2.resize(cnn_mean, (224, 224))
            cnn_std_resized = cv2.resize(cnn_std, (224, 224))

            # CNN средняя карта
            plt.subplot(3, 4, 5)
            plt.imshow(cnn_mean_resized, cmap='hot', interpolation='bilinear')
            plt.title('CNN: Средняя тепловая карта', fontsize=12)
            plt.colorbar()

            # CNN стандартное отклонение
            plt.subplot(3, 4, 6)
            plt.imshow(cnn_std_resized, cmap='viridis', interpolation='bilinear')
            plt.title('CNN: Стандартное отклонение', fontsize=12)
            plt.colorbar()

            # CNN наложение на изображение
            plt.subplot(3, 4, 7)
            img_resized = np.array(original_image.resize((224, 224)))
            heatmap_cnn = plt.cm.hot(cnn_mean_resized)[:, :, :3]
            overlay_cnn = 0.6 * img_resized/255.0 + 0.4 * heatmap_cnn
            plt.imshow(overlay_cnn)
            plt.title('CNN: Наложение', fontsize=12)
            plt.axis('off')

        if vit_maps:
            vit_mean = np.mean(vit_maps, axis=0)
            vit_std = np.std(vit_maps, axis=0)

            # Масштабирование карт ViT до 224x224
            vit_mean_resized = cv2.resize(vit_mean, (224, 224))
            vit_std_resized = cv2.resize(vit_std, (224, 224))

            # ViT средняя карта
            plt.subplot(3, 4, 9)
            plt.imshow(vit_mean_resized, cmap='hot', interpolation='bilinear')
            plt.title('ViT: Средняя карта внимания', fontsize=12)
            plt.colorbar()

            # ViT стандартное отклонение
            plt.subplot(3, 4, 10)
            plt.imshow(vit_std_resized, cmap='viridis', interpolation='bilinear')
            plt.title('ViT: Стандартное отклонение', fontsize=12)
            plt.colorbar()

            # ViT наложение на изображение
            plt.subplot(3, 4, 11)
            img_resized = np.array(original_image.resize((224, 224)))
            heatmap_vit = plt.cm.hot(vit_mean_resized)[:, :, :3]
            overlay_vit = 0.6 * img_resized/255.0 + 0.4 * heatmap_vit
            plt.imshow(overlay_vit)
            plt.title('ViT: Наложение', fontsize=12)
            plt.axis('off')

        # Метрики стабильности
        plt.subplot(3, 4, (4, 8))
        metrics_text = self._format_metrics(results)
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        plt.title('Метрики стабильности и согласованности', fontsize=14, fontweight='bold')

        # Межмодельное сравнение
        if 'cross_iou' in results:
            plt.subplot(3, 4, 12)
            # Приведение карт к единому размеру
            target_size = (224, 224)
            img_resized = np.array(original_image.resize(target_size))

            # Средние карты
            cnn_mean = cv2.resize(np.mean(cnn_maps, axis=0), target_size)
            vit_mean = cv2.resize(np.mean(vit_maps, axis=0), target_size)

            # Нормализация
            cnn_norm = (cnn_mean - cnn_mean.min()) / (cnn_mean.max() - cnn_mean.min() + 1e-8)
            vit_norm = (vit_mean - vit_mean.min()) / (vit_mean.max() - vit_mean.min() + 1e-8)

            # Бинаризация с адаптивным порогом
            threshold = np.percentile(np.concatenate([cnn_norm.flatten(), vit_norm.flatten()]), 75)
            cnn_binary = (cnn_norm > threshold).astype(np.uint8)
            vit_binary = (vit_norm > threshold).astype(np.uint8)
            intersection = np.logical_and(cnn_binary, vit_binary).astype(np.uint8)

            plt.imshow(intersection, cmap='Reds')
            plt.title(f'Пересечение регионов (IoU = {results["cross_iou"]:.3f})', fontsize=12)
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Дополнительная визуализация корреляций
        if cnn_maps and vit_maps and len(cnn_maps) > 1 and len(vit_maps) > 1:
            try:
                self._plot_correlation_analysis(cnn_maps, vit_maps)
            except Exception as e:
                print(f"Предупреждение: Не удалось построить корреляционный анализ: {e}")

        # Визуализация аугментаций
        self._visualize_augmentations()

    def _visualize_augmentations(self):
        """Визуализация примеров аугментированных изображений"""
        if not self.augmented_samples:
            return

        plt.figure(figsize=(15, 5))
        plt.suptitle("Примеры аугментированных изображений", fontsize=16)

        num_samples = min(5, len(self.augmented_samples))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(self.augmented_samples[i])
            plt.title(f'Вариант {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def _plot_correlation_analysis(self, cnn_maps, vit_maps):
        """Дополнительная визуализация корреляционного анализа"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle("Корреляционный анализ стабильности", fontsize=16)

        # CNN корреляционная матрица
        cnn_corr_matrix = np.corrcoef([m.flatten() for m in cnn_maps])
        sns.heatmap(cnn_corr_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[0,0], cbar_kws={'label': 'Корреляция'})
        axes[0,0].set_title('CNN: Корреляции между прогонами')

        # ViT корреляционная матрица
        vit_corr_matrix = np.corrcoef([m.flatten() for m in vit_maps])
        sns.heatmap(vit_corr_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[0,1], cbar_kws={'label': 'Корреляция'})
        axes[0,1].set_title('ViT: Корреляции между прогонами')

        # Распределение значений карт
        cnn_values = np.concatenate([m.flatten() for m in cnn_maps])
        vit_values = np.concatenate([m.flatten() for m in vit_maps])

        axes[1,0].hist(cnn_values, bins=50, alpha=0.7, color='red', label='CNN')
        axes[1,0].hist(vit_values, bins=50, alpha=0.7, color='blue', label='ViT')
        axes[1,0].set_xlabel('Значения активации')
        axes[1,0].set_ylabel('Частота')
        axes[1,0].set_title('Распределение значений активации')
        axes[1,0].legend()

        # Стабильность по позициям
        cnn_std_flat = np.std(cnn_maps, axis=0).flatten()
        vit_std_flat = np.std(vit_maps, axis=0).flatten()

        # Если размеры разные, приводим к одному размеру
        if len(cnn_std_flat) != len(vit_std_flat):
            target_size = min(len(cnn_std_flat), len(vit_std_flat))
            cnn_std_flat = cnn_std_flat[:target_size]
            vit_std_flat = vit_std_flat[:target_size]

        axes[1,1].scatter(cnn_std_flat, vit_std_flat, alpha=0.6)
        axes[1,1].plot([0, max(cnn_std_flat.max(), vit_std_flat.max())],
                       [0, max(cnn_std_flat.max(), vit_std_flat.max())], 'r--', alpha=0.5)
        axes[1,1].set_xlabel('CNN Стандартное отклонение')
        axes[1,1].set_ylabel('ViT Стандартное отклонение')
        axes[1,1].set_title('Сравнение стабильности по позициям')

        plt.tight_layout()
        plt.show()

    def _format_metrics(self, results):
        """Форматирование метрик для отображения"""
        text = "АНАЛИЗ СТАБИЛЬНОСТИ И СОГЛАСОВАННОСТИ МОДЕЛЕЙ\n"
        text += "="*60 + "\n\n"

        if 'cnn_mean_std' in results:
            text += "🔴 CNN (ResNet + ScoreCAM):\n"
            text += f"• Средн. станд. откл.:     {results['cnn_mean_std']:.4f}\n"
            text += f"• Макс. станд. откл.:      {results['cnn_max_std']:.4f}\n"
            text += f"• Стабильность предск.:    {results['cnn_pred_stability']:.4f}\n"
            text += f"• Средн. корреляция:       {results['cnn_mean_correlation']:.4f}\n"
            text += f"• Станд. откл. корр.:      {results['cnn_std_correlation']:.4f}\n"
            text += f"• Средн. MSE:              {results['cnn_mean_mse']:.4f}\n"
            text += f"• Станд. откл. MSE:        {results['cnn_std_mse']:.4f}\n"
            text += f"• SSIM (сходство):         {results['cnn_mean_ssim']:.4f}\n"
            text += f"• Корр. Спирмена:          {results['cnn_spearman']:.4f}\n"
            text += f"• Корр. Кендалла:          {results['cnn_kendall']:.4f}\n\n"

        if 'vit_mean_std' in results:
            text += "🟢 ViT (Vision Transformer):\n"
            text += f"• Средн. станд. откл.:     {results['vit_mean_std']:.4f}\n"
            text += f"• Макс. станд. откл.:      {results['vit_max_std']:.4f}\n"
            text += f"• Стабильность предск.:    {results['vit_pred_stability']:.4f}\n"
            text += f"• Средн. корреляция:       {results['vit_mean_correlation']:.4f}\n"
            text += f"• Станд. откл. корр.:      {results['vit_std_correlation']:.4f}\n"
            text += f"• Средн. MSE:              {results['vit_mean_mse']:.4f}\n"
            text += f"• Станд. откл. MSE:        {results['vit_std_mse']:.4f}\n"
            text += f"• SSIM (сходство):         {results['vit_mean_ssim']:.4f}\n"
            text += f"• Корр. Спирмена:          {results['vit_spearman']:.4f}\n"
            text += f"• Корр. Кендалла:          {results['vit_kendall']:.4f}\n\n"

        # Межмодельные метрики
        if 'cross_spearman' in results:
            text += "🌐 МЕЖМОДЕЛЬНОЕ СРАВНЕНИЕ:\n"
            text += f"• SSIM (сходство):         {results['cross_ssim']:.4f}\n"
            text += f"• Корр. Спирмена:          {results['cross_spearman']:.4f}\n"
            text += f"• Корр. Кендалла:          {results['cross_kendall']:.4f}\n"
            text += f"• IoU (пересечение):        {results['cross_iou']:.4f}\n\n"

        # Сравнение стабильности
        if 'cnn_mean_std' in results and 'vit_mean_std' in results:
            text += "📊 СРАВНЕНИЕ СТАБИЛЬНОСТИ:\n"
            stability_diff = results['vit_mean_std'] - results['cnn_mean_std']
            corr_diff = results['vit_mean_correlation'] - results['cnn_mean_correlation']

            if stability_diff < 0:
                text += f"• ViT более стабилен на {abs(stability_diff):.4f}\n"
            else:
                text += f"• CNN более стабилен на {abs(stability_diff):.4f}\n"

            if corr_diff > 0:
                text += f"• ViT более консистентен на {corr_diff:.4f}\n"
            else:
                text += f"• CNN более консистентен на {abs(corr_diff):.4f}\n"

        return text

# ============================= ОСНОВНОЙ КОД =============================

def main():
    # Пути к моделям (ИЗМЕНИТЕ НА СВОИ ПУТИ)
    cnn_model_path = "C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst/Fusion_model/best_fusion_model.pth"
    vit_model_path = "vit_classifier.pth"

    # Путь к тестовому изображению (ИЗМЕНИТЕ НА СВОЙ ПУТЬ)
    test_image_path = "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Human embryo image datasets/ed3/alldata/2/image_10.jpg"

    # Создание компаратора
    comparator = ModelComparator(cnn_model_path, vit_model_path)

    # Сравнение моделей с аугментациями
    results = comparator.compare_single_image(test_image_path, num_runs=10)

    # Выводы
    print("\n" + "="*60)
    print("ЗАКЛЮЧЕНИЕ ПО АНАЛИЗУ УСТОЙЧИВОСТИ МОДЕЛЕЙ К ВАРИАЦИЯМ")
    print("="*60)

    if 'cnn_mean_std' in results and 'vit_mean_std' in results:
        print(f"\n📈 Стабильность карт активации:")
        print(f"   CNN средн. станд. откл.: {results['cnn_mean_std']:.4f}")
        print(f"   ViT средн. станд. откл.: {results['vit_mean_std']:.4f}")

        if results['cnn_mean_std'] > results['vit_mean_std']:
            print(f"   ✅ ViT показал БОЛЬШУЮ устойчивость к вариациям")
        else:
            print(f"   ✅ CNN показал БОЛЬШУЮ устойчивость к вариациям")

        print(f"\n🎯 Консистентность между прогонами:")
        print(f"   CNN средн. корреляция: {results['cnn_mean_correlation']:.4f}")
        print(f"   ViT средн. корреляция: {results['vit_mean_correlation']:.4f}")

        if results['vit_mean_correlation'] > results['cnn_mean_correlation']:
            print(f"   ✅ ViT показал БОЛЬШУЮ консистентность при вариациях")
        else:
            print(f"   ✅ CNN показал БОЛЬШУЮ консистентность при вариациях")

        print(f"\n🔄 Стабильность предсказаний:")
        print(f"   CNN: {results['cnn_pred_stability']:.4f} (чем ближе к 0, тем стабильнее)")
        print(f"   ViT: {results['vit_pred_stability']:.4f} (чем ближе к 0, тем стабильнее)")

        print(f"\n🔍 Внутримодельное сходство:")
        print(f"   CNN SSIM: {results.get('cnn_mean_ssim', 0):.4f}")
        print(f"   ViT SSIM: {results.get('vit_mean_ssim', 0):.4f}")

    # Анализ согласованности объяснений
    if 'cross_iou' in results:
        print("\n🤝 Согласованность объяснений моделей:")
        print(f"   SSIM:               {results.get('cross_ssim', 0):.4f}")
        print(f"   Корр. Спирмена:    {results.get('cross_spearman', 0):.4f}")
        print(f"   Корр. Кендалла:    {results.get('cross_kendall', 0):.4f}")
        print(f"   IoU:               {results.get('cross_iou', 0):.4f}")

        if results['cross_iou'] > 0.3:
            print("   ✅ Высокая согласованность важных регионов")
        else:
            print("   ❌ Низкая согласованность важных регионов")

        print(f"• Средн. MSE CNN:              {results['cnn_mean_mse']:.4f}\n")
        print(f"• Станд. откл. MSE CNN:        {results['cnn_std_mse']:.4f}\n")
        print(f"• Средн. MSE VIT:              {results['vit_mean_mse']:.4f}\n")
        print(f"• Станд. откл. MSE VIT:        {results['vit_std_mse']:.4f}\n")

if __name__ == "__main__":
    main()