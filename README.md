# 🐱 猫狗大战图像分类作业 🐶

这是一份基于卷积神经网络（CNN）的猫狗图像二分类课程作业，使用 Kaggle Dogs vs. Cats 数据集完成训练与预测。

---

## 📁 项目结构说明
cats-vs-dogs-homework/
├── README.md # 作业说明文档（当前文件）
├── train.py # 模型训练主程序
├── predict.py # 单张图片预测脚本（可选）
├── requirements.txt # 项目依赖库列表
└── .gitignore # 忽略配置文件（避免上传大文件）

---

## 🛠️ 环境与运行步骤

### 1. 安装依赖库
在终端执行以下命令安装所有依赖：
```bash
pip install -r requirements.txt

requirements.txt 内容如下：
tensorflow==2.15.0
numpy==1.26.4
matplotlib==3.8.3
pillow==10.2.0
