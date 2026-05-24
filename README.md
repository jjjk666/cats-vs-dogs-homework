# 🐱 猫狗大战图像分类作业 🐶

这是一份基于卷积神经网络（CNN）的猫狗图像二分类课程作业，使用 Kaggle Dogs vs. Cats 数据集完成训练与预测。

---

## 📁 项目结构说明
cats-vs-dogs-homework/
├── README.md # 作业说明文档（当前文件）
├── 猫狗.py # 主程序（训练 + 预测）
├── requirements.txt # 项目依赖库列表
├── .gitignore # 忽略配置文件（避免上传大文件）
├── data/ # 数据集文件夹（需自行下载）
│ ├── train/
│ │ ├── cat/ # 训练集猫图片
│ │ └── dog/ # 训练集狗图片
│ └── test/
│ ├── cat/ # 测试集猫图片
│ └── dog/ # 测试集狗图片
└── cat_dog_model.pth # 训练后生成的模型文件


---

## 🛠️ 环境与运行步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt


