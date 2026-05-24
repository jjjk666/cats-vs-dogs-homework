import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------- 1. 超参数配置 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 0.001
epochs = 10
image_size = (128, 128)  # 统一图片尺寸

# ---------------------- 2. 数据预处理 ----------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------- 3. 自定义数据集 ----------------------
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # 读取文件夹下的图片（假设train文件夹下是cat和dog两个子文件夹）
        for label, class_name in enumerate(['cat', 'dog']):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------- 4. 搭建CNN模型 ----------------------
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 16x64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 32x32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 输出: 64x16x16
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 二分类：猫(0)/狗(1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

# ---------------------- 5. 训练与测试 ----------------------
def train():
    # 加载数据集（请根据你的实际路径修改）
    train_dataset = CatDogDataset(root_dir='./data/train', transform=transform)
    test_dataset = CatDogDataset(root_dir='./data/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数、优化器
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 打印训练信息
        train_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')

        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = 100 * test_correct / test_total
        print(f'Test Acc: {test_acc:.2f}%\n')

    # 保存模型
    torch.save(model.state_dict(), 'cat_dog_model.pth')
    print("训练完成，模型已保存为 cat_dog_model.pth")

# ---------------------- 6. 单张图片预测 ----------------------
def predict_image(img_path):
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load('cat_dog_model.pth'))
    model.eval()

    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        class_names = ['猫', '狗']
        result = class_names[predicted.item()]

    plt.imshow(image)
    plt.title(f'预测结果: {result}')
    plt.axis('off')
    plt.show()
    print(f"预测结果：这张图片是{result}")

if __name__ == '__main__':
    # 运行训练（首次使用时取消注释）
    # train()

    # 运行预测（训练完成后使用，替换成你的图片路径）
    predict_image('test_cat.jpg')
