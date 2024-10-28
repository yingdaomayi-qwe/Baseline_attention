import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一层目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# 将上一层目录添加到 sys.path
sys.path.append(parent_dir)
from attention_model import RSRP_Prediction_Model  # 假设你之前的模型代码保存在 model.py 中
from data_loader_attention import Dataset_attention
from torch.utils.data import DataLoader
from utils.tools import del_files, EarlyStopping
# 定义训练和验证的函数
def train_model(model, train_loader, val_loader, epochs, learning_rate, device,model_save_path):
    # 定义均方误差损失函数
    criterion = nn.MSELoss()

    # 使用 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将模型移动到指定的设备 (CPU 或 GPU)
    model.to(device)


    # 存储每个 epoch 的训练损失和验证损失
    train_loss_history = []
    val_loss_history = []
    # 开始训练循环
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_train_loss = 0.0
        print("--------------------Epoch{}--------------------".format(epoch))
        for i, (inputs, targets) in tqdm(enumerate(train_loader)):
            # 将数据移动到指定的设备
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            # 初始化梯度
            optimizer.zero_grad()

            # 前向传播
            bs_tx_rsrp = model(inputs)

            # 只计算 bs_tx 的损失
            loss_bs_tx = criterion(bs_tx_rsrp, targets)

            # 反向传播
            loss_bs_tx.backward()

            # 优化模型参数
            optimizer.step()

            # 累加损失值
            running_train_loss += loss_bs_tx.item()

            if (i + 1) % 100 == 0:
                print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_bs_tx.item()))

        # 计算训练集上的平均损失
        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        # 在验证集上评估模型
        model.eval()  # 设置模型为评估模式
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float().to(device), targets.float().to(device)

                # 前向传播
                bs_tx_rsrp = model(inputs)

                # 只计算 bs_tx 的损失
                loss_bs_tx = criterion(bs_tx_rsrp, targets)

                # 累加验证集损失
                running_val_loss += loss_bs_tx.item()

        # 计算验证集上的平均损失
        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)

        # 打印当前 epoch 的训练损失和验证损失
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        early_stopping(epoch_val_loss, model, model_save_path)
        if early_stopping.early_stop:
            break

    plot_loss(train_loss_history, val_loss_history)

    print("Training Finished!")


# 定义绘制训练和验证损失曲线的函数
def plot_loss(train_loss_history, val_loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Loss Over Epochs (Only bs_tx)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 假设你已经准备好了数据集


# 主函数
if __name__ == '__main__':
    # 设置参数
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Time-LLM Attention Model Training')

    # 模型和训练参数配置
    parser.add_argument('--d_embed', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--s_tf', type=int, default=10, help='Input token length')
    parser.add_argument('--k', type=int, default=10, help='Number of prediction tokens')
    parser.add_argument('--h_tf', type=int, default=6, help='Number of Transformer layers')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--dataset_path', type=str, default='./dataset_attention/', help='Path to the dataset')

    # 解析参数
    args = parser.parse_args()
    setting = 'dembed{}_stf{}_k{}_htf{}_lr{}_epochs{}_bs{}'.format(
        args.d_embed,
        args.s_tf,
        args.k,
        args.h_tf,
        args.learning_rate,
        args.train_epochs,
        args.batch_size
    )
    early_stopping=EarlyStopping(accelerator=None, patience=5)

    path = os.path.join('./dataset_attention/model_save/',
                        setting)  # unique checkpoint saving path

    if not os.path.exists(path) :
        os.makedirs(path)


    train_loader = DataLoader(
        Dataset_attention(root_path=args.dataset_path, S_TF=10,K=10,flag='train'),
        batch_size=args.batch_size,
        shuffle='True',
        num_workers=8)

    val_loader = DataLoader(
        Dataset_attention(root_path=args.dataset_path, S_TF=10, K=10, flag='val'),
        batch_size=args.batch_size,
        shuffle='False',
        num_workers=8)



    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = RSRP_Prediction_Model(args.d_embed, args.s_tf, args.k, args.h_tf)

    # 加载数据

    # 训练模型并验证
    train_model(model, train_loader, val_loader, args.train_epochs, args.learning_rate, device,path)
