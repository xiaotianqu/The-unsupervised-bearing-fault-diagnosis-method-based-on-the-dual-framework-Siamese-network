import torch
import torch.nn.functional as F
from model import DeepSiameseNetwork
from Dataset import TestSiameseDatasetA, TestSiameseDatasetB
from torch.utils.data import DataLoader
# Step 1: 加载模型
in_channels = 3  # RGB 图像
height, width = 227, 227
model = DeepSiameseNetwork(in_channels)
model.load_state_dict(torch.load(r'D:\故障诊断\是否故障\切比雪夫\best_model.pth'))
model.eval()
# Step 2: 准备数据
# 假设有一对需要验证的图像：img1 和 img2
img1_path = r'D:\故障诊断\是否故障\验证\108验证\外圈'
img2_path = r'D:\故障诊断\是否故障\验证\sqv验证\正常'

# 创建数据集对象
def test_code(img1_path, img2_path, model, label):
    test_dataset_A = TestSiameseDatasetA(img1_path)
    test_dataset_B = TestSiameseDatasetB(img2_path)
    test_dataset_A = DataLoader(test_dataset_A, batch_size=1, shuffle=False)
    test_dataset_B = DataLoader(test_dataset_B, batch_size=1, shuffle=False)

    accuracy = test1(test_dataset_A, test_dataset_B, model, label)
    return accuracy


def test1(test_loader_A, test_loader_B, model, true_label):
    model.eval()
    total = 0
    correct_nosame = 0
    correct_same = 0
    print(len(test_loader_A))
    print(len(test_loader_B))

    with torch.no_grad():
        for data_A, data_B in zip(test_loader_A, test_loader_B):
            output_A, output_B = model(data_A, data_B)
            dist = F.pairwise_distance(output_A, output_B)
            target = torch.zeros_like(dist)  # 假设目标为0
            predicted_euclidean = (dist < 0.5).float()
            correct_nosame += (predicted_euclidean == target).sum().item()

            # 计算与真实标签的比较结果
            correct_same += (predicted_euclidean == 1).sum().item()  # 标签为1的数量即为正确数量

            total += target.size(0)

        # 计算准确率
        nosame = correct_nosame / total
        same = correct_same / total

        return nosame, same


# 调用测试函数
label_tensor = torch.tensor(1)  # 将标签封装成张量 # 标签，1表示相似，0表示不相似

accuracy = test_code(img1_path, img2_path, model, label_tensor)
print("Accuracy第一个数是不相似，第二个数是相似：", accuracy)
#第一个数是不相似，第二个数是相似