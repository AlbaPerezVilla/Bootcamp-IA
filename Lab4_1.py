## SCRIPT ENTREGA 1 LAB 4

# ---- Librerías ----
import os 
from pathlib import Path
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from PIL import Image


if __name__ == '__main__':

    import os 
    from pathlib import Path
    import zipfile
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    import torch 
    import torch.nn as nn
    import torch.optim as optim 
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, utils
    from PIL import Image

    # --- CONFIGURACIÓN DE RUTAS ---
    os.chdir(Path(__file__).parent)
    DATA_ROOT = Path('images')
    TRAIN_DIR = DATA_ROOT / 'seg_train'
    VAL_DIR = DATA_ROOT / 'seg_test'
    PRED_DIR = DATA_ROOT / 'seg_pred'

    BATCH_SIZE = 4
    NUM_WORKERS = 2
    NUM_CLASSES = 6
    IMAGE_SIZE = 150 # el dataset tiene imágenes 150x150
   
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Working directory:", Path.cwd())
    print("Train dir exists?", TRAIN_DIR.exists())
    print("Val dir exists?", VAL_DIR.exists())
    print("Device:", DEVICE)

    # --- TRANSFORMACIONES ---
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),               # Rotación aleatoria entre -30º y +30º
        transforms.RandomHorizontalFlip(p=0.5),              # Flip horizontal con probabilidad 0.5
        transforms.RandomResizedCrop(size=150, scale=(0.8, 1.0)),  # Recorte aleatorio
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Cambios aleatorios en brillo, contraste y saturación
        transforms.RandomGrayscale(p=0.2),                   # Convierte a escala de grises con probabilidad 0.2
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    # --- DATASETS Y DATALOADERS ---
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print('\nERROR: No encuentro las carpetas esperadas. Asegúrate de haber descomprimido el dataset en data/seg_train y data/seg_test.')
        print('Ejemplo de descarga con la CLI de Kaggle: kaggle datasets download -d puneet6060/intel-image-classification --unzip -p ./data')
    else:
        train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=val_transform)


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


        print('Clases detectadas (train):', train_dataset.classes)
        print('Tamaño training:', len(train_dataset), 'tamaño val:', len(val_dataset))

    ## Visualizar batch
    def imshow_tensor(img_tensor, title=None):
        img = img_tensor.clone().cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1) # desnormalizar
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')

    ## Cargar 4 imágenes sin aumento para ver el original
    if TRAIN_DIR.exists():
        # dataset sin transform para visualizar originales
        raw_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
        raw_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=raw_transform)
        raw_loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=True)


        dataiter = iter(raw_loader)
        images_raw, labels_raw = next(dataiter)


        # Ahora cargar un batch transformado usando train_loader
        dataiter2 = iter(train_loader)
        images_aug, labels_aug = next(dataiter2)


    plt.figure(figsize=(10,4))
    for i in range(BATCH_SIZE):
        plt.subplot(2, BATCH_SIZE, i+1)
        imshow_tensor(images_raw[i])
        plt.title(f'orig: {raw_dataset.classes[labels_raw[i]]}')


        plt.subplot(2, BATCH_SIZE, BATCH_SIZE + i + 1)
        imshow_tensor(images_aug[i])
        plt.title(f'aug: {train_dataset.classes[labels_aug[i]]}')
    plt.tight_layout()
    plt.show()

    ## Definir la CNN
    # Arquitectura sencilla:
    # - Conv(3 -> 16) + ReLU + MaxPool
    # - Conv(16 -> 32) + ReLU + MaxPool
    # - FC -> ReLU -> FC out

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), # -> 150 -> 75


                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), # -> 75 -> 37
            )
            # calcular tamaño de flatten: 32 * 37 * 37
            self.flatten_size = 32 * 37 * 37
            self.classifier = nn.Sequential(
                nn.Linear(self.flatten_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        
    model = SimpleCNN().to(DEVICE)
    print(model)

    ## Criterio y optimizador
    # - CrossEntropyLoss
    # - Adam lr=1e-4, weight_decay para regularización

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    ## Bucle de entrenamiento en 30 epocas
    EPOCHS = 30
    
    if TRAIN_DIR.exists():
        for epoch in range(1, EPOCHS+1):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)


                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item() * inputs.size(0)


            epoch_loss = running_loss / len(train_dataset)
            print(f'Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f}')

        # guardar modelo
        torch.save(model.state_dict(), 'intel_cnn.pth')
        print('Modelo guardado en intel_cnn.pth')

    ##Evaluación final

    if TRAIN_DIR.exists():
        model.eval()
        correct = 0 
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _,preds = torch.max(outputs,1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct/total
        print(f'Precisión en validación: {accuracy:.4f}')

            