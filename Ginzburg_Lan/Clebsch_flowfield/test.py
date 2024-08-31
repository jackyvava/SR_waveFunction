

# 数据
## 读取原始数据
import scipy.io as sio
import torch

# 读取速度场数据
velocity_data = sio.loadmat('D:/zjPhD/Programzj/psiToU/Ginzburg_Lan/Clebsch_flowfield/data/High_Resolution/train/velocity_field.mat')
ux0 = torch.tensor(velocity_data['ux'], dtype=torch.float32)
uy0 = torch.tensor(velocity_data['uy'], dtype=torch.float32)

# 读取波函数数据
wave_function_data = sio.loadmat('D:/zjPhD/Programzj/psiToU/Ginzburg_Lan/Clebsch_flowfield/data/High_Resolution/train/wave_function.mat')
psi1_0 = torch.tensor(wave_function_data['psi1'], dtype=torch.complex64)
psi2_0 = torch.tensor(wave_function_data['psi2'], dtype=torch.complex64)

# 读取误差场数据
error_field_data = sio.loadmat('D:/zjPhD/Programzj/psiToU/Ginzburg_Lan/Clebsch_flowfield/data/High_Resolution/train/error_field.mat')
vx_error0 = torch.tensor(error_field_data['vx_error'], dtype=torch.float32)
vy_error0 = torch.tensor(error_field_data['vy_error'], dtype=torch.float32)

# 打印数据形状以确认
print(f'ux shape: {ux0.shape}')
print(f'uy shape: {uy0.shape}')
print(f'psi1 shape: {psi1_0.shape}')
print(f'psi2 shape: {psi2_0.shape}')
print(f'vx_error shape: {vx_error0.shape}')
print(f'vy_error shape: {vy_error0.shape}')
device = 'cuda:0'
# device = 'cpu'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

print(f'Using device: {device}')
# 常用函数

# 归一化函数
def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()

# 模型
## 预测误差
import torch 
import torch.nn as nn

class ErrorPredictCNN(nn.Module):
    def __init__(self):
        super(ErrorPredictCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)

        return x
model  = ErrorPredictCNN().to(device)
ux = ux0.clone()
uy = uy0.clone()
psi1 = psi1_0.clone()
psi2 = psi2_0.clone()

psi1_real  = torch.real(psi1)
psi1_imag  = torch.imag(psi1)
psi2_real  = torch.real(psi2)
psi2_imag  = torch.imag(psi2)


vx_error = vx_error0.clone()
vy_error = vy_error0.clone()



print(f'ux shape: {ux.shape}')
print(f'uy shape: {uy.shape}')
print(f'psi1 shape: {psi1.shape}')
print(f'psi2 shape: {psi2.shape}')
print(f'psi1_real shape: {psi1_real.shape}')
print(f'psi1_imag shape: {psi1_imag.shape}')
print(f'psi2_real shape: {psi2_real.shape}')
print(f'psi2_imag shape: {psi2_imag.shape}')

print(f'vx_error shape: {vx_error.shape}')
print(f'vy_error shape: {vy_error.shape}')


# 读取和归一化数据
ux = normalize(ux0.clone().unsqueeze(0).unsqueeze(0)).to(device)
uy = normalize(uy0.clone().unsqueeze(0).unsqueeze(0)).to(device)
psi1 = psi1_0.clone().unsqueeze(0).unsqueeze(0).to(device)
psi2 = psi2_0.clone().unsqueeze(0).unsqueeze(0).to(device)

# 将复数数据拆分为实部和虚部，并进行归一化
psi1_real = normalize(torch.real(psi1)).to(device)
psi1_imag = normalize(torch.imag(psi1)).to(device)
psi2_real = normalize(torch.real(psi2)).to(device)
psi2_imag = normalize(torch.imag(psi2)).to(device)

# 错误数据不需要归一化，因为它们是训练目标
vx_error = vx_error0.clone().unsqueeze(0).unsqueeze(0).to(device)
vy_error = vy_error0.clone().unsqueeze(0).unsqueeze(0).to(device)

# 打印数据形状
print(f'ux shape: {ux.shape}')
print(f'uy shape: {uy.shape}')
print(f'psi1_real shape: {psi1_real.shape}')
print(f'psi1_imag shape: {psi1_imag.shape}')
print(f'psi2_real shape: {psi2_real.shape}')
print(f'psi2_imag shape: {psi2_imag.shape}')

print(f'vx_error shape: {vx_error.shape}')
print(f'vy_error shape: {vy_error.shape}')
input_data = torch.cat((ux, uy, psi1_real, psi1_imag, psi2_real, psi2_imag), dim=1)

input_data.shape
# 目标输出合并为2个通道
target_error = torch.cat((vx_error, vy_error), dim=1)

print(f'target_error shape: {target_error.shape}')
# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练循环
num_epochs = 100000  # 设定你想要的训练轮数
for epoch in range(num_epochs):
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    predicted_error = model(input_data)
    
    # 计算损失
    loss = 100000*criterion(predicted_error, target_error)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    # 打印每个 epoch 的损失
    if (epoch + 1) % 10 == 0:  # 每10个 epoch 打印一次
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 可视化结果（在 GPU 上训练，但绘图在 CPU 上）
predicted_error_cpu = predicted_error.detach().cpu().numpy()
vx_error_cpu = vx_error.detach().cpu().numpy()


import matplotlib.pyplot as plt

plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.imshow(vx_error_cpu[0,0,:,:],vmin=-0.006,vmax=0.004)
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(predicted_error_cpu[0,0,:,:],vmin=-0.006,vmax=0.004)
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(predicted_error_cpu[0,0,:,:] - vx_error_cpu[0,0,:,:],vmin=-0.006,vmax=0.004)
plt.colorbar()

plt.show()
## 超分辨
### 下采样数据
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

# 读取数据
velocity_data = sio.loadmat('D:/zjPhD/Programzj/psiToU/Ginzburg_Lan/Clebsch_flowfield/data/High_Resolution/train/velocity_field.mat')
ux0 = torch.tensor(velocity_data['ux'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
uy0 = torch.tensor(velocity_data['uy'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

wave_function_data = sio.loadmat('D:/zjPhD/Programzj/psiToU/Ginzburg_Lan/Clebsch_flowfield/data/High_Resolution/train/wave_function.mat')
psi1_0 = torch.tensor(wave_function_data['psi1'], dtype=torch.complex64)
psi2_0 = torch.tensor(wave_function_data['psi2'], dtype=torch.complex64)

# 分别提取实部和虚部，并添加通道维度
psi1_real = psi1_0.real.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
psi1_imag = psi1_0.imag.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
psi2_real = psi2_0.real.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
psi2_imag = psi2_0.imag.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

print(f'ux0 shape: {ux0.shape}')
print(f'uy0 shape: {uy0.shape}')
print(f'psi1_real shape: {psi1_real.shape}')
print(f'psi1_imag shape: {psi1_imag.shape}')
print(f'psi2_real shape: {psi2_real.shape}')
print(f'psi2_imag shape: {psi2_imag.shape}')

# 模型定义
class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # 上采样（从128x128到256x256）
        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()

        # 上采样（从256x256到512x512）
        self.upconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=6, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        x = self.relu4(self.upconv1(x))  # 第一次上采样
        x = self.upconv2(x)  # 第二次上采样

        return x

# 归一化函数
def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()

# 下采样函数
def downsample(ux, uy, psi1_real, psi1_imag, psi2_real, psi2_imag):
    downsample = nn.AvgPool2d(kernel_size=4, stride=4)
    ux_low_res = downsample(ux)
    uy_low_res = downsample(uy)
    psi1_real_low_res = downsample(psi1_real)
    psi1_imag_low_res = downsample(psi1_imag)
    psi2_real_low_res = downsample(psi2_real)
    psi2_imag_low_res = downsample(psi2_imag)
    low_res_input = torch.cat([ux_low_res, uy_low_res, psi1_real_low_res, psi1_imag_low_res, psi2_real_low_res, psi2_imag_low_res], dim=1)
    return low_res_input

# 数据预处理
low_res_input = downsample(ux0, uy0, psi1_real, psi1_imag, psi2_real, psi2_imag)

# 模型初始化
model = SuperResolutionCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
model.train()
epochs = 200000
for epoch in range(epochs):
    optimizer.zero_grad()

    # 将输入传递给模型
    output = model(low_res_input.to(device))

    # 计算损失
    target = torch.cat([ux0, uy0, psi1_real, psi1_imag, psi2_real, psi2_imag], dim=1).to(device)  # 目标是高分辨率的 ux 和 uy
    loss = 10000*criterion(output, target)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

output.shape,target.shape
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(output[0,0,:,:].detach().cpu().numpy())
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(target[0,0,:,:].detach().cpu().numpy())
plt.colorbar()


