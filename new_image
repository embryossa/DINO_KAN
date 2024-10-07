import timm
import torch
from PIL import Image
from kan import *
from torchvision import transforms

# Загрузка предобученной модели DINO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
model.eval()  # Переключение модели в режим оценки
model.to(device)  # Перемещение модели на соответствующее устройство

# Определение преобразований изображений
dino_transforms = transforms.Compose([
    transforms.Resize((518, 518)),  # Изменение размера изображений
    transforms.ToTensor(),  # Преобразование изображений в тензор
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Нормализация изображений
])


# Функция для загрузки и предобработки изображения
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB') 
    image = dino_transforms(image)  
    image = image.unsqueeze(0)  
    return image.to(device)


# Экстракция признаков с использованием DINO
def extract_features_dino(image_tensor):
    with torch.no_grad():
        features = model(image_tensor)  # Пропуск изображения через модель DINO
    return features


# Загрузка обученной модели KAN
class KAN(torch.nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.fc1 = torch.nn.Linear(width[0], width[1])
        self.fc2 = torch.nn.Linear(width[1], width[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Инициализация модели KAN (указываем те же параметры, что использовались при обучении)
kan_model = KAN(width=[1024, 10, 1], grid=5, k=3)  # 1024 – размер признаков от DINO
kan_model.load_state_dict(torch.load('kan_image_model.pth'))  # Загрузка сохранённой модели KAN
kan_model.eval()
kan_model.to(device)


# Функция для оценки нового изображения
def evaluate_image(image_path, model, kan_model, threshold=0.3):
    # Загрузка и предобработка изображения
    image = Image.open(image_path).convert('RGB')
    image = dino_transforms(image).unsqueeze(0).to(device)  # Преобразование изображения

    # Извлечение признаков с помощью DINO
    with torch.no_grad():
        extracted_features = model(image)

    # Прогноз с использованием модели KAN
    kan_model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(kan_model(extracted_features))

    # Преобразование предсказания в бинарный класс
    predicted_class = (prediction > threshold).float()

    # Вывод вероятности и класса
    print(f"Prediction (probability): {prediction.item():.4f}")
    if predicted_class.item() == 1:
        print("The image is in focus.")  # Сообщение, если вероятность больше порога
    else:
        print("The image is out of focus.")  # Сообщение, если вероятность меньше порога


# Путь к новому изображению
image_path = "C:/Users/User/Desktop/IVF/AI/Blastocyst_Dataset/Images/0018_03.png"
evaluate_image(image_path, model, kan_model)
