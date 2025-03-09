import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Dataset dizini
dataset_dir = r"C:\Users\MP\PycharmProjects\ÖDEV"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Eğitim ve test dizinindeki dosyaları kontrol et
print("Dataset Directory:", dataset_dir)
print("Train Directory:", train_dir)
print("Test Directory:", test_dir)

# Eğitim ve test klasörlerinde dosyaları listele
print("Eğitim dizinindeki dosyalar:")
print(os.listdir(train_dir))
print("Test dizinindeki dosyalar:")
print(os.listdir(test_dir))

# Eğitim verisindeki alt klasörlere göz atma
train_images = []
test_images = []

# Klasör isimlerini sıralı bir şekilde etiket olarak alalım
class_names = sorted(os.listdir(train_dir))  # 'train' dizinindeki klasör isimlerini alıp sıralıyoruz

# labels.txt dosyasını atlamak için kontrol ekleyelim
if 'labels.txt' in class_names:
    class_names.remove('labels.txt')

# Eğitim verisindeki alt klasörlere göz atma
for label_name in class_names:
    subdir_path = os.path.join(train_dir, label_name)
    if os.path.isdir(subdir_path):
        for img_file in os.listdir(subdir_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Görsellerin .png, .jpg, .jpeg olduğunu kontrol et
                train_images.append((os.path.join(subdir_path, img_file), label_name))

# Test verisindeki alt klasörlere göz atma
for label_name in class_names:
    subdir_path = os.path.join(test_dir, label_name)
    if os.path.isdir(subdir_path):
        for img_file in os.listdir(subdir_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Görsellerin .png, .jpg, .jpeg olduğunu kontrol et
                test_images.append((os.path.join(subdir_path, img_file), label_name))

# Görüntülerin yüklendiğini kontrol et
print(f"Eğitimde {len(train_images)} eğitim görüntüsü bulundu.")
print(f"Testte {len(test_images)} test görüntüsü bulundu.")

# **PyTorch Dataset Sınıfı**
class FashionMNISTDataset(Dataset):
    def __init__(self, image_folder, is_test=False):
        self.image_folder = image_folder
        self.images = []
        self.labels = []

        # Etiket dosyasını test seti için kontrol et
        if not is_test:
            for label_name in class_names:
                subdir_path = os.path.join(image_folder, label_name)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path):
                        if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Görsel formatlarını kontrol et
                            self.images.append(os.path.join(subdir_path, img_file))
                            self.labels.append(label_name)  # Etiketi string olarak ekliyoruz

            print(f"✅ {len(self.images)} eğitim görüntüsü yüklendi!")
        else:
            # Test verisi için sadece görselleri al
            for label_name in class_names:
                subdir_path = os.path.join(image_folder, label_name)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path):
                        if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Görsel formatlarını kontrol et
                            self.images.append(os.path.join(subdir_path, img_file))

            print(f"✅ {len(self.images)} test görüntüsü yüklendi!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")  # Grayscale
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # [1, 28, 28] şeklinde

        if hasattr(self, 'labels'):
            label = self.labels[idx]
            return image, label
        return image  # Test seti için sadece resim döndürülür


# Eğitim ve test dataset'lerini oluştur
train_dataset = FashionMNISTDataset(train_dir, is_test=False)
test_dataset = FashionMNISTDataset(test_dir, is_test=True)

# Veri setlerinin düzgün yüklendiğini kontrol et
if len(train_dataset) == 0 or len(test_dataset) == 0:
    print("Eğitim veya test verisinde görüntü bulunamadı!")
else:
    print(f"Eğitim veri setinde {len(train_dataset)} görüntü yüklendi.")
    print(f"Test veri setinde {len(test_dataset)} görüntü yüklendi.")

# labels.txt dosyasına etiketleri yazma
labels_path = os.path.join(train_dir, "labels.txt")

with open(labels_path, "w") as f:
    for image_path, label_name in train_images:
        image_name = os.path.basename(image_path)  # Görselin adını al
        f.write(f"{image_name} {label_name}\n")  # Görselin ismini ve etiketini yaz

print(f"Labels are written to: {labels_path}")

# DataLoader'ları tanımla
batch_size = 128
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Tüm kategorilerden birer örnek alıp hepsini birleştirmek
fig, axes = plt.subplots(1, len(class_names), figsize=(20, 5))

# Her kategoriden bir görsel seç
for i, label_name in enumerate(class_names):
    # İlgili kategoriye ait ilk resmi alalım
    sample_image_path = None
    for img_path, label in train_images:
        if label == label_name:
            sample_image_path = img_path
            break

    # Görsel bulunamazsa kullanıcıyı bilgilendirelim
    if sample_image_path is None:
        print(f"{label_name} kategorisi için görsel bulunamadı.")
        continue  # Bu kategoriyi atlayıp diğer kategorilere geçelim

    # Görseli yükle ve işleme
    sample_image = Image.open(sample_image_path).convert("L")
    sample_image = np.array(sample_image, dtype=np.float32) / 255.0

    # Görseli eksende göster
    axes[i].imshow(sample_image, cmap='gray')
    axes[i].set_title(f"Label: {label_name}")
    axes[i].axis('off')  # Eksenleri kaldır

# İlk resim: Kategorilerden örnekler
example_image_path = os.path.join(dataset_dir, 'example_images.png')
try:
    plt.savefig(example_image_path)  # Grafik olarak kaydet
    plt.show()  # Ekranda göster
    print(f"Örnek görüntüler kaydedildi: {example_image_path}")
except Exception as e:
    print(f"Örnek görüntü kaydedilirken hata oluştu: {e}")

# İkinci resim: Eğitim doğruluğu
# Bu grafik örnek olarak eğitim verileri ile ilgili bir doğru orantı çizecek
# Bu kısmı eğitim doğruluğunuzla değiştirmeniz gerekebilir.

epochs = 10
train_accuracy = np.random.rand(epochs)  # Eğitim doğruluğu (örnek veriler)
test_accuracy = np.random.rand(epochs)   # Test doğruluğu (örnek veriler)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_accuracy, label="Eğitim Doğruluğu", marker='o')
plt.plot(range(epochs), test_accuracy, label="Test Doğruluğu", marker='x')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Eğitim ve Test Doğruluğu')
plt.legend()

# İkinci resim: Doğruluk grafiği
accuracy_image_path = os.path.join(dataset_dir, 'accuracy_graph.png')
try:
    plt.savefig(accuracy_image_path)  # Grafik olarak kaydet
    plt.show()  # Ekranda göster
    print(f"Doğruluk grafiği kaydedildi: {accuracy_image_path}")
except Exception as e:
    print(f"Doğruluk grafiği kaydedilirken hata oluştu: {e}")