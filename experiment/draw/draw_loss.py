import matplotlib.pyplot as plt

# 读取文件并提取损失值
def read_loss_file(file_path):
    epochs = []
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            # 提取 epoch 和 loss 值
            parts = line.split(',')
            epoch_part = parts[0].split('[')[1].split('/')[0]
            loss_part = parts[1].split(':')[1].strip()
            # 添加到列表
            epochs.append(int(epoch_part))
            losses.append(float(loss_part))
    return epochs, losses

# 绘制并保存损失曲线
def plot_loss_curve(file_path, save_path):
    epochs, losses = read_loss_file(file_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label="Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Loss曲线已保存至: {save_path}")

path = "/root/FE_Fusion/train/result_2024-10-27-12-19/"
plot_loss_curve(path + "loss.txt", path + "loss.png")  # 替换成你的损失文件路径和保存路径
