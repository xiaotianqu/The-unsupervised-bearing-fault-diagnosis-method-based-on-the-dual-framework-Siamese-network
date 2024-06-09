import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss import ContrastiveLoss_1
from Dataset import SiameseDataset, TestSiameseDatasetA, TestSiameseDatasetB
from model import DeepSiameseNetwork
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(siamese_net, criterion, optimizer, train_dataloader, num_epochs, device):
    siamese_net.to(device)  # 将模型移动到GPU上
    train_losses = []  # 用于保存训练损失

    for epoch in range(num_epochs):
        for i, (x1, x2, y) in enumerate(train_dataloader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)  # 将输入数据移动到GPU上
            optimizer.zero_grad()
            output1, output2 = siamese_net(x1, x2)
            loss = criterion(output1, output2, y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}')
                train_losses.append(loss.item())  # 保存每次迭代的损失值

    return train_losses
def test1(test_loader_A, test_loader_B, model, cuda):
    model.eval()
    total = 0
    correct_euclidean = 0

    with torch.no_grad():
        for data_A, data_B in zip(test_loader_A, test_loader_B):


            if cuda:
                model.cuda()
                data_A = [item.cuda() for item in data_A]
                data_B = [item.cuda() for item in data_B]
                data_A = torch.stack(data_A)
                data_B = torch.stack(data_B)

            output_A, output_B = model(data_A, data_B)
            dist = F.pairwise_distance(output_A, output_B)
            target = torch.zeros_like(dist)  # 假设目标为0
            predicted_euclidean = (dist < 0.5).float()
            correct_euclidean += (predicted_euclidean == target).sum().item()

            total += target.size(0)
        accuracy_euclidean = correct_euclidean / total

        return accuracy_euclidean


def main():
    # 定义超参数
    learning_rate = 0.001
    num_epochs = 20
    in_channels = 3  # RGB 图像
    margin = 1

    # 初始化 Siamese 网络和损失函数
    siamese_net = DeepSiameseNetwork(in_channels)  # 传递必要的参数
    criterion = ContrastiveLoss_1(margin)

    # 创建数据集对象a
    train_dataset = SiameseDataset(train_folder_path)
    test_dataset_A = TestSiameseDatasetA(test_folder_path_A)
    test_dataset_B = TestSiameseDatasetB(test_folder_path_B)

    # 创建 DataLoader 对象
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader_A = DataLoader(test_dataset_A, batch_size=1, shuffle=False)
    test_loader_B = DataLoader(test_dataset_B, batch_size=1, shuffle=False)

    # 初始化优化器
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=learning_rate)

    # # 训练
    train_losses = train(siamese_net, criterion, optimizer, train_dataloader, num_epochs, device)
    torch.save(siamese_net.state_dict(), 'best_model.pth')
    #测试
    accuracy = test1(test_loader_A, test_loader_B, siamese_net, cuda=True)
    print(f"Test Accuracy: {accuracy}")

    # #绘制训练损失曲线
    plt.plot(list(range(1, len(train_losses) + 1)), train_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('train_losses.png')
    plt.show()
    plt.savefig('train_losses.png')


if __name__ == "__main__":
    train_folder_path = r"D:\故障诊断\是否故障\精炼数据集"
    test_folder_path_A = r'D:\故障诊断\是否故障\验证\108验证\外圈'
    test_folder_path_B = r'D:\故障诊断\是否故障\验证\sqv验证\内圈'

    main()
