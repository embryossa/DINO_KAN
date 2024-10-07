import os
import pandas as pd

# Путь к папке с изображениями
path_to_images = "C:/Users/User/Desktop/IVF/AI/Blastocyst"

# Создание списка для хранения разметки
data = []

# Перебор файлов в папке
for filename in os.listdir(path_to_images):
    if filename.endswith(".png"):
        # Извлечение информации из имени файла
        parts = filename.split('_')  # Пример: b1_z_0.png -> ['b1', 'z', '0']
        blastocyst_id = parts[0]  # b1, b122 и т.д.
        z_value = parts[2].replace('.png', '')  # 0, 100, -200

        # Определение метки: в фокусе (1) или не в фокусе (0)
        if z_value == '0':
            label = 1  # В фокусе
        else:
            label = 0  # Не в фокусе

        # Добавление информации в список
        data.append([filename, blastocyst_id, z_value, label])

# Создание DataFrame из списка
df = pd.DataFrame(data, columns=['filename', 'blastocyst_id', 'z_value', 'label'])
# Преобразование меток в строки
df['label'] = df['label'].astype(str)
# Сохранение разметки в CSV
df.to_csv('image_labels.csv', index=False)
