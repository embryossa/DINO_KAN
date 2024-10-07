import os

import lovely_tensors as lt
import torch

lt.monkey_patch()
import tqdm
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from PIL import Image

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description="Extract features using a pretrained image model.")
parser.add_argument('--model_name', type=str, default='vit_large_patch14_dinov2.lvd142m',
                    help='Model name to use for feature extraction.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to image dataset')
parser.add_argument('--save_path', type=str, required=True, help='Path to save extracted features')
args = parser.parse_args()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')


device = get_default_device()
print('device:', device)

# Загрузка предобученной модели DINO
model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
model.eval()  # Переключение модели в режим оценки
model.to(device)  # Перемещение модели на соответствующее устройство

# Определение преобразований изображений
dino_transforms = transforms.Compose([
    transforms.Resize((518, 518)),  # Изменение размера изображений
    transforms.ToTensor(),  # Преобразование изображений в тензор
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Нормализация изображений
])


# Определение функции для загрузки изображений из папки
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')  # Открытие изображения
            images.append((img, filename))  # Добавление изображения и его имени
    return images


# Загрузка изображений
dataset_path = args.dataset_path
images = load_images_from_folder(dataset_path)


# Создание DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, filename = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, filename


custom_dataset = CustomDataset(images, transform=dino_transforms)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=False)


# Функция для извлечения признаков
def extract_features(data_loader):
    features = []
    filenames = []
    with torch.no_grad():  # Нет необходимости отслеживать градиенты
        for inputs, batch_filenames in tqdm.tqdm(data_loader):
            inputs = inputs.to(device)  # Перемещение входных данных на соответствующее устройство
            output = model(inputs)
            features.append(output.cpu())
            filenames.extend(batch_filenames)

    features = torch.cat(features, dim=0)
    return features, filenames


# Извлечение и сохранение признаков
extracted_features, filenames = extract_features(data_loader)

# Сохранение извлеченных признаков и имён файлов
os.makedirs(args.save_path, exist_ok=True)  # Убедитесь, что каталог существует
torch.save((extracted_features, filenames), os.path.join(args.save_path, 'extracted_features.pt'))

print("Feature extraction completed and saved.")
