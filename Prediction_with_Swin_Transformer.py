import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms

# Конфигурация
CFG = {
    "img_size": (512, 384)  # Размер изображения должен соответствовать обученной модели
}

class ClinicalDataInput:
    @staticmethod
    def get_float_input(prompt, min_val=0.0, max_val=20.0):
        while True:
            try:
                value = float(input(prompt))
                if min_val <= value <= max_val:
                    return value
                print(f"Значение должно быть между {min_val} и {max_val}")
            except ValueError:
                print("Пожалуйста, введите числовое значение")

    @staticmethod
    def get_int_input(prompt, min_val=0, max_val=None):
        while True:
            try:
                value = int(input(prompt))
                if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                    print(f"Допустимый диапазон: {min_val}-{max_val}" if max_val else f"Минимум: {min_val}")
                    continue
                return value
            except ValueError:
                print("Пожалуйста, введите целое число")

    @classmethod
    def collect_data(cls):
        print("\nВведите клинические данные:")
        data = [
            cls.get_int_input("EXP_silver (оценка расширения, 1-6): ", 1, 6),
            cls.get_int_input("ICM_silver (внутренняя клеточная масса, 1-4): ", 1, 4),
            cls.get_int_input("TE_silver (оценка трофэктодермы, 1-4): ", 1, 4),
            cls.get_int_input("COC (количество ооцитов): ", 0),
            cls.get_int_input("MII (количество MII ооцитов): ", 0),
            cls.get_int_input("Возраст пациентки (лет, 18-50): ", 18, 50),
            cls.get_float_input("Толщина эндометрия (мм, 5.0-20.0): ", 5.0, 20.0)
        ]
        return data

class EmbryoPredictor:
    def __init__(self):
        self.model = EmbryoModel.load_from_checkpoint(
            checkpoint_path="C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst/IVFormer.ckpt",
            num_classes=5
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(CFG['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, clinical_data):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            morph_logits, birth_logits = self.model(img_tensor, clinical_tensor)

        morph_probs = F.softmax(morph_logits, dim=1)
        birth_prob = torch.sigmoid(birth_logits)

        return {
            'morphology_class': morph_probs.argmax().item() + 1,
            'morphology_probabilities': morph_probs.cpu().numpy()[0],
            'live_birth_probability': birth_prob.item()
        }

class EmbryoModel(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = SwinTransformer(
            img_size=CFG['img_size'],
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            num_classes=0
        )

        self.morphology_head = nn.Linear(1024, num_classes)
        self.live_birth_head = nn.Sequential(
            nn.Linear(1024 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x, clinical=None):
        features = self.backbone(x)
        morph_logits = self.morphology_head(features)

        if clinical is not None:
            birth_input = torch.cat([features, clinical], dim=1)
            birth_pred = self.live_birth_head(birth_input)
            return morph_logits, birth_pred

        return morph_logits

if __name__ == "__main__":
    predictor = EmbryoPredictor()

    # Пример пути к изображению:
    # "C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst/IVFormer/example.jpg"

    image_path = "C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst/IVFormer/0005_01.png"

    try:
        clinical_data = ClinicalDataInput.collect_data()
        prediction = predictor.predict(image_path, clinical_data)

        print("\nРезультаты предсказания:")
        print(f"Класс морфологии: {prediction['morphology_class']}")
        print(f"Вероятность живорождения: {prediction['live_birth_probability']:.2%}")
        print("\nВероятности по классам морфологии:")
        for i, prob in enumerate(prediction['morphology_probabilities'], 1):
            print(f"Класс {i}: {prob:.2%}")

    except Exception as e:
        print(f"\nОшибка: {str(e)}")
