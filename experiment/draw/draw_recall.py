import matplotlib.pyplot as plt

# 读取文件并提取Recall@1和Recall@5
def read_recall_file(file_path):
    epochs = []
    recall_1 = []
    recall_5 = []
    with open(file_path, 'r') as file:
        for line in file:
            # 按照 "Epoch X, Recall@1: Y, Recall@5: Z" 的格式解析
            parts = line.split(',')
            epoch = int(parts[0].split()[1])
            r1 = float(parts[1].split(':')[1].strip())
            r5 = float(parts[2].split(':')[1].strip())
            # 添加到列表
            epochs.append(epoch)
            recall_1.append(r1)
            recall_5.append(r5)
    return epochs, recall_1, recall_5

# 绘制 Recall 曲线
def plot_recall_curve(file_path, save_path='recall_curve.png'):
    epochs, recall_1, recall_5 = read_recall_file(file_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, recall_1, label="Recall@1", marker='o')
    plt.plot(epochs, recall_5, label="Recall@5", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall@1 and Recall@5 Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Recall曲线已保存至: {save_path}")

# 使用示例，假设文件路径为 'recall_results.txt'
file_path = '/root/FE_Fusion/train/result_2024-10-26-15-18/saved_model/'
plot_recall_curve(file_path+"recall_results.txt", save_path=file_path+'recall_curve.png')
