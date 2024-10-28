import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from attention_model import RSRP_Prediction_Model  # 假设你之前的模型代码保存在 model.py 中
from data_loader_attention import Dataset_attention
from torch.utils.data import DataLoader

# 定义训练和验证的函数
def test_model(model, val_loader, device):
    # 定义均方误差损失函数
    criterion = nn.MSELoss()


    # 将模型移动到指定的设备 (CPU 或 GPU)
    model.to(device)

    # 存储每个 epoch 的训练损失和验证损失
    val_loss_history = []
    gain_matrix=[]
    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            # 前向传播
            bs_tx_rsrp = model(inputs)

            _,beam_best_prediction = torch.max(bs_tx_rsrp, dim=-1)
            beam_best_prediction=beam_best_prediction.unsqueeze(-1)
            normal_gain_predict = torch.gather(targets, dim=2, index=beam_best_prediction)
            gain_matrix.append(normal_gain_predict.detach().squeeze().cpu().numpy())
            # 只计算 bs_tx 的损失
            loss_bs_tx = criterion(bs_tx_rsrp, targets)
            # 累加验证集损失
            running_val_loss += loss_bs_tx.item()


        # 计算验证集上的平均损失
    epoch_val_loss = running_val_loss / len(val_loader)
    val_loss_history.append(epoch_val_loss)

    # 打印训练损失和验证损失
    print(f"Val Loss: {epoch_val_loss:.4f}")

    normal_gain_big = np.vstack(gain_matrix)
    normal_gain_average = np.average(normal_gain_big, axis=0).reshape(1, -1)
    print(normal_gain_average)


# 主函数
if __name__ == '__main__':
    # 设置参数
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

    path = os.path.join('./dataset_attention/model_save/',
                        setting)  # unique checkpoint saving path


    test_loader = DataLoader(
        Dataset_attention(root_path=args.dataset_path, S_TF=10, K=10, flag='test'),
        batch_size=args.batch_size,
        shuffle='False',
        num_workers=8)

    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    best_model_path = path + '/' + 'checkpoint'


    model = RSRP_Prediction_Model(args.d_embed, args.s_tf, args.k, args.h_tf)


    test_model(model, test_loader, device)
