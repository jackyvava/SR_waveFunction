clc;
clear;


load('data00000.mat');  
load('data00001.mat');  
load('data00010.mat');  
load('data00100.mat');  
load('data00500.mat');  
load('data01000.mat');  
%%
% 对速度场进行归一化
vx00000_normalized = (vx00000 - min(vx00000(:))) / (max(vx00000(:)) - min(vx00000(:)))*2 -1;
vx00001_normalized = (vx00001 - min(vx00001(:))) / (max(vx00001(:)) - min(vx00001(:)))*2 -1;
vx00010_normalized = (vx00010 - min(vx00010(:))) / (max(vx00010(:)) - min(vx00010(:)))*2 -1;
vx00100_normalized = (vx00100 - min(vx00100(:))) / (max(vx00100(:)) - min(vx00100(:)))*2 -1;
vx00500_normalized = (vx00500 - min(vx00500(:))) / (max(vx00500(:)) - min(vx00500(:)))*2 -1;
vx01000_normalized = (vx01000 - min(vx01000(:))) / (max(vx01000(:)) - min(vx01000(:)))*2 -1;

% 
vy00000_normalized = (vy00000 - min(vy00000(:))) / (max(vy00000(:)) - min(vy00000(:)))*2 -1;
vy00001_normalized = (vy00001 - min(vy00001(:))) / (max(vy00001(:)) - min(vy00001(:)))*2 -1;
vy00010_normalized = (vy00010 - min(vy00010(:))) / (max(vy00010(:)) - min(vy00010(:)))*2 -1;
vy00100_normalized = (vy00100 - min(vy00100(:))) / (max(vy00100(:)) - min(vy00100(:)))*2 -1;
vy00500_normalized = (vy00500 - min(vy00500(:))) / (max(vy00500(:)) - min(vy00500(:)))*2 -1;
vy01000_normalized = (vy01000 - min(vy01000(:))) / (max(vy01000(:)) - min(vy01000(:)))*2 -1;

%
vx_pre00000_normalized = (vx_pre00000 - min(vx_pre00000(:))) / (max(vx_pre00000(:)) - min(vx_pre00000(:)))*2 -1;
vx_pre00001_normalized = (vx_pre00001 - min(vx_pre00001(:))) / (max(vx_pre00001(:)) - min(vx_pre00001(:)))*2 -1;
vx_pre00010_normalized = (vx_pre00010 - min(vx_pre00010(:))) / (max(vx_pre00010(:)) - min(vx_pre00010(:)))*2 -1;
vx_pre00100_normalized = (vx_pre00100 - min(vx_pre00100(:))) / (max(vx_pre00100(:)) - min(vx_pre00100(:)))*2 -1;
vx_pre00500_normalized = (vx_pre00500 - min(vx_pre00500(:))) / (max(vx_pre00500(:)) - min(vx_pre00500(:)))*2 -1;
vx_pre01000_normalized = (vx_pre01000 - min(vx_pre01000(:))) / (max(vx_pre01000(:)) - min(vx_pre01000(:)))*2 -1;

%
vy_pre00000_normalized = (vy_pre00000 - min(vy_pre00000(:))) / (max(vy_pre00000(:)) - min(vy_pre00000(:)))*2 -1;
vy_pre00001_normalized = (vy_pre00001 - min(vy_pre00001(:))) / (max(vy_pre00001(:)) - min(vy_pre00001(:)))*2 -1;
vy_pre00010_normalized = (vy_pre00010 - min(vy_pre00010(:))) / (max(vy_pre00010(:)) - min(vy_pre00010(:)))*2 -1;
vy_pre00100_normalized = (vy_pre00100 - min(vy_pre00100(:))) / (max(vy_pre00100(:)) - min(vy_pre00100(:)))*2 -1;
vy_pre00500_normalized = (vy_pre00500 - min(vy_pre00500(:))) / (max(vy_pre00500(:)) - min(vy_pre00500(:)))*2 -1;
vy_pre01000_normalized = (vy_pre01000 - min(vy_pre01000(:))) / (max(vy_pre01000(:)) - min(vy_pre01000(:)))*2 -1;

%%
%计算 psi的误差
psi1_real_delta_1 = psi1_real_part_00001 - psi1_real_part_00000;
psi1_real_delta_2 = psi1_real_part_00010 - psi1_real_part_00000;
psi1_real_delta_3 = psi1_real_part_00100 - psi1_real_part_00000;
psi1_real_delta_4 = psi1_real_part_00500 - psi1_real_part_00000;
psi1_real_delta_5 = psi1_real_part_01000 - psi1_real_part_00000;

psi1_imag_delta_1 = psi1_imag_part_00001 - psi1_imag_part_00000;
psi1_imag_delta_2 = psi1_imag_part_00010 - psi1_imag_part_00000;
psi1_imag_delta_3 = psi1_imag_part_00100 - psi1_imag_part_00000;
psi1_imag_delta_4 = psi1_imag_part_00500 - psi1_imag_part_00000;
psi1_imag_delta_5 = psi1_imag_part_01000 - psi1_imag_part_00000;

psi2_real_delta_1 = psi2_real_part_00001 - psi2_real_part_00000;
psi2_real_delta_2 = psi2_real_part_00010 - psi2_real_part_00000;
psi2_real_delta_3 = psi2_real_part_00100 - psi2_real_part_00000;
psi2_real_delta_4 = psi2_real_part_00500 - psi2_real_part_00000;
psi2_real_delta_5 = psi2_real_part_01000 - psi2_real_part_00000;

psi2_imag_delta_1 = psi2_imag_part_00001 - psi2_imag_part_00000;
psi2_imag_delta_2 = psi2_imag_part_00010 - psi2_imag_part_00000;
psi2_imag_delta_3 = psi2_imag_part_00100 - psi2_imag_part_00000;
psi2_imag_delta_4 = psi2_imag_part_00500 - psi2_imag_part_00000;
psi2_imag_delta_5 = psi2_imag_part_01000 - psi2_imag_part_00000;

%% 归一化前
vx_delta_1 = vx00001 - vx00000;
vx_delta_2 = vx00010 - vx00000;
vx_delta_3 = vx00100 - vx00000;
vx_delta_4 = vx00500 - vx00000;
vx_delta_5 = vx01000 - vx00000;

%计算ux的预测误差
vx_pre_delta_1 = vx_pre00001 - vx_pre00000;
vx_pre_delta_2 = vx_pre00010 - vx_pre00000;
vx_pre_delta_3 = vx_pre00100 - vx_pre00000;
vx_pre_delta_4 = vx_pre00500 - vx_pre00000;
vx_pre_delta_5 = vx_pre01000 - vx_pre00000;

% 计算uy的原始误差
vy_delta_1 = vy00001 - vy00000;
vy_delta_2 = vy00010 - vy00000;
vy_delta_3 = vy00100 - vy00000;
vy_delta_4 = vy00500 - vy00000;
vy_delta_5 = vy01000 - vy00000;


%计算uy的预测误差
vy_pre_delta_1 = vy_pre00001 - vy_pre00000;
vy_pre_delta_2 = vy_pre00010 - vy_pre00000;
vy_pre_delta_3 = vy_pre00100 - vy_pre00000;
vy_pre_delta_4 = vy_pre00500 - vy_pre00000;
vy_pre_delta_5 = vy_pre01000 - vy_pre00000;

%% 
% 归一化后的
% 计算ux的原始误差
vx_delta_1_norm = vx00001_normalized - vx00000_normalized;
vx_delta_2_norm = vx00010_normalized - vx00000_normalized;
vx_delta_3_norm = vx00100_normalized - vx00000_normalized;
vx_delta_4_norm = vx00500_normalized - vx00000_normalized;
vx_delta_5_norm = vx01000_normalized - vx00000_normalized;

%计算ux的预测误差
vx_pre_delta_1_norm = vx_pre00001_normalized - vx_pre00000_normalized;
vx_pre_delta_2_norm = vx_pre00010_normalized - vx_pre00000_normalized;
vx_pre_delta_3_norm = vx_pre00100_normalized - vx_pre00000_normalized;
vx_pre_delta_4_norm = vx_pre00500_normalized - vx_pre00000_normalized;
vx_pre_delta_5_norm = vx_pre01000_normalized - vx_pre00000_normalized;

% 计算uy的原始误差
vy_delta_1_norm = vy00001_normalized - vy00000_normalized;
vy_delta_2_norm = vy00010_normalized - vy00000_normalized;
vy_delta_3_norm = vy00100_normalized - vy00000_normalized;
vy_delta_4_norm = vy00500_normalized - vy00000_normalized;
vy_delta_5_norm = vy01000_normalized - vy00000_normalized;


%计算uy的预测误差
vy_pre_delta_1_norm = vy_pre00001_normalized - vy_pre00000_normalized;
vy_pre_delta_2_norm = vy_pre00010_normalized - vy_pre00000_normalized;
vy_pre_delta_3_norm = vy_pre00100_normalized - vy_pre00000_normalized;
vy_pre_delta_4_norm = vy_pre00500_normalized - vy_pre00000_normalized;
vy_pre_delta_5_norm = vy_pre01000_normalized - vy_pre00000_normalized;

%%
figure;
subplot(1,3,1);
histogram(vx_delta_1_norm(:), 30);

subplot(1,3,2);
histogram(vx_pre_delta_2_norm(:), 30);

subplot(1,3,3);
histogram(vx_pre_delta_3_norm(:), 30);


%%
% 计算SSIM曲线
vx_SSIM_1 = ssim(vx00000, vx00000);
vx_SSIM_2 = ssim(vx00001, vx00000);
vx_SSIM_3 = ssim(vx00010, vx00000);
vx_SSIM_4 = ssim(vx00100, vx00000);
vx_SSIM_5 = ssim(vx00500, vx00000);
vx_SSIM_6 = ssim(vx01000, vx00000);

vy_SSIM_1 = ssim(vy00000, vy00000);
vy_SSIM_2 = ssim(vy00001, vy00000);
vy_SSIM_3 = ssim(vy00010, vy00000);
vy_SSIM_4 = ssim(vy00100, vy00000);
vy_SSIM_5 = ssim(vy00500, vy00000);
vy_SSIM_6 = ssim(vy01000, vy00000);

vx_pre_SSIM_1 = ssim(vx_pre00000, vx_pre00000);
vx_pre_SSIM_2 = ssim(vx_pre00001, vx_pre00000);
vx_pre_SSIM_3 = ssim(vx_pre00010, vx_pre00000);
vx_pre_SSIM_4 = ssim(vx_pre00100, vx_pre00000);
vx_pre_SSIM_5 = ssim(vx_pre00500, vx_pre00000);
vx_pre_SSIM_6 = ssim(vx_pre01000, vx_pre00000);

vy_pre_SSIM_1 = ssim(vy_pre00000, vy_pre00000);
vy_pre_SSIM_2 = ssim(vy_pre00001, vy_pre00000);
vy_pre_SSIM_3 = ssim(vy_pre00010, vy_pre00000);
vy_pre_SSIM_4 = ssim(vy_pre00100, vy_pre00000);
vy_pre_SSIM_5 = ssim(vy_pre00500, vy_pre00000);
vy_pre_SSIM_6 = ssim(vy_pre01000, vy_pre00000);





ssim_vx = [vx_SSIM_1,vx_SSIM_2, vx_SSIM_3,vx_SSIM_4,vx_SSIM_5, vx_SSIM_6];
ssim_vy = [vy_SSIM_1,vy_SSIM_2, vy_SSIM_3,vy_SSIM_4,vy_SSIM_5, vy_SSIM_6];

ssim_vx_pre = [vx_pre_SSIM_1,vx_pre_SSIM_2, vx_pre_SSIM_3,vx_pre_SSIM_4,vx_pre_SSIM_5, vx_pre_SSIM_6];
ssim_vy_pre = [vy_pre_SSIM_1,vy_pre_SSIM_2, vy_pre_SSIM_3,vy_pre_SSIM_4,vy_pre_SSIM_5, vy_pre_SSIM_6];


run_time = [1,2,3,4,5,6];


figure;

% 绘制 SSIM 曲线 vx 和 vx_pre
plot(run_time, ssim_vx, '-o', 'LineWidth', 2, 'DisplayName', 'SSIM vx');  % vx 曲线
hold on;
plot(run_time, ssim_vy, '-s', 'LineWidth', 2, 'DisplayName', 'SSIM vy');  % vy 曲线
plot(run_time, ssim_vx_pre, '-d', 'LineWidth', 2, 'DisplayName', 'SSIM vx pre');  % vx_pre 曲线
plot(run_time, ssim_vy_pre, '-x', 'LineWidth', 2, 'DisplayName', 'SSIM vy pre');  % vy_pre 曲线

% 添加图例
legend('show', 'Location', 'best');

% 添加轴标签和标题
xlabel('Runtime');
ylabel('SSIM');
title('SSIM vs Runtime for vx, vy, vx_pre, and vy_pre');

% 添加网格
grid on;

hold off;

%%
figure;
imagesc(vx_pre00000);
axis off; 
saveas(gcf, 'vx_pre_00000.png'); 


figure;
imagesc(vx_pre00500);
axis off; 
saveas(gcf, 'vx_pre_00500.png'); 

figure;
imagesc(psi1_real_part_00000);
axis off; 
saveas(gcf, 'psi1_real_part_00500.png'); 

figure;
imagesc(psi1_imag_part_00000);
axis off; 
saveas(gcf, 'psi1_imag_part_00500.png'); 


figure;
imagesc(psi1_real_part_00500);
axis off; 
saveas(gcf, 'psi1_real_part_00500.png'); 

figure;
imagesc(psi1_imag_part_00500);
axis off; 
saveas(gcf, 'psi1_imag_part_00500.png'); 

%%
figure;
subplot(1,2,1);
histogram(test(:), 30);

subplot(1,2,2);
histogram(psi1_real_delta_4(:), 30);
%%
%误差直方图（Histogram of Error Distribution）
figure;

% 绘制 psi1_real_delta_1 误差的直方图
subplot(3,2,1);
histogram(psi1_real_delta_1(:), 50); % 50 bins
title('psi1 real part error (Noise level: 0.00001)');
xlabel('Error value');
ylabel('Frequency');

subplot(3,2,2);
histogram(psi1_real_delta_2(:), 50);
title('psi1 real part error (Noise level: 0.00010)');
xlabel('Error value');
ylabel('Frequency');

subplot(3,2,3);
histogram(psi1_real_delta_3(:), 50);
title('psi1 real part error (Noise level: 0.00100)');
xlabel('Error value');
ylabel('Frequency');

subplot(3,2,4);
histogram(psi1_real_delta_4(:), 50);
title('psi1 real part error (Noise level: 0.00500)');
xlabel('Error value');
ylabel('Frequency');

subplot(3,2,5);
histogram(psi1_real_delta_5(:), 50);
title('psi1 real part error (Noise level: 0.01000)');
xlabel('Error value');
ylabel('Frequency');
%%
%误差场图（Error Field Map）
figure;

% 绘制 psi1_real_delta_1 误差场
subplot(3,2,1);
contourf(psi1_real_delta_1, 20, 'LineColor', 'none'); % 20个颜色层次
colorbar;
title('psi1 real part error (Noise level: 0.00001)');
xlabel('X');
ylabel('Y');

subplot(3,2,2);
contourf(psi1_real_delta_2, 20, 'LineColor', 'none');
colorbar;
title('psi1 real part error (Noise level: 0.00010)');
xlabel('X');
ylabel('Y');

subplot(3,2,3);
contourf(psi1_real_delta_3, 20, 'LineColor', 'none');
colorbar;
title('psi1 real part error (Noise level: 0.00100)');
xlabel('X');
ylabel('Y');

subplot(3,2,4);
contourf(psi1_real_delta_4, 20, 'LineColor', 'none');
colorbar;
title('psi1 real part error (Noise level: 0.00500)');
xlabel('X');
ylabel('Y');

subplot(3,2,5);
contourf(psi1_real_delta_5, 20, 'LineColor', 'none');
colorbar;
title('psi1 real part error (Noise level: 0.01000)');
xlabel('X');
ylabel('Y');
%%
%误差累积分布函数（CDF）
figure;

% 将误差转为向量并排序
psi1_real_delta_1_sorted = sort(abs(psi1_real_delta_1(:)));
cdf_1 = (1:length(psi1_real_delta_1_sorted)) / length(psi1_real_delta_1_sorted);

plot(psi1_real_delta_1_sorted, cdf_1, 'LineWidth', 2);
hold on;

psi1_real_delta_2_sorted = sort(abs(psi1_real_delta_2(:)));
cdf_2 = (1:length(psi1_real_delta_2_sorted)) / length(psi1_real_delta_2_sorted);
plot(psi1_real_delta_2_sorted, cdf_2, 'LineWidth', 2);

psi1_real_delta_3_sorted = sort(abs(psi1_real_delta_3(:)));
cdf_3 = (1:length(psi1_real_delta_3_sorted)) / length(psi1_real_delta_3_sorted);
plot(psi1_real_delta_3_sorted, cdf_3, 'LineWidth', 2);

psi1_real_delta_4_sorted = sort(abs(psi1_real_delta_4(:)));
cdf_4 = (1:length(psi1_real_delta_4_sorted)) / length(psi1_real_delta_4_sorted);
plot(psi1_real_delta_4_sorted, cdf_4, 'LineWidth', 2);

psi1_real_delta_5_sorted = sort(abs(psi1_real_delta_5(:)));
cdf_5 = (1:length(psi1_real_delta_5_sorted)) / length(psi1_real_delta_5_sorted);
plot(psi1_real_delta_5_sorted, cdf_5, 'LineWidth', 2);

legend('Noise level: 0.00001', 'Noise level: 0.00010', 'Noise level: 0.00100', 'Noise level: 0.00500', 'Noise level: 0.01000');
xlabel('Error');
ylabel('Cumulative Probability');
title('CDF of psi1 real part error');
hold off;

%%

figure();  % 设置图窗大小 (宽1200像素，高800像素)

% 设置colorbar的范围
color_range = [-0.9, 0.9];

% 调整每个子图的位置和大小 (Position = [left, bottom, width, height])
subplot(1, 3, 1);  % 第1个子图
imagesc(psi1_imag_part_00000);
title('psi1 imag part noise=0');
xlabel('X'); ylabel('Y');
% colorbar;
clim(color_range);
axis equal tight;
set(gca, 'Position', [0.05, 0.15, 0.25, 0.7]);  % 调整子图位置

subplot(1, 3, 2);  % 第2个子图
imagesc(psi1_imag_part_00001);
title('psi1 imag part noise=0.0001');
xlabel('X'); ylabel('Y');
% colorbar;
clim(color_range);
axis equal tight;
set(gca, 'Position', [0.35, 0.15, 0.25, 0.7]);  % 调整子图位置

subplot(1, 3, 3);  % 第3个子图
imagesc(psi1_imag_part_00001 - psi1_imag_part_00000);
title('psi1 imag part noise=0.0001');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);
axis equal tight;
set(gca, 'Position', [0.65, 0.15, 0.25, 0.7]);  % 调整子图位置


%%
% 绘图
figure;
subplot(1, 3, 1);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


subplot(1, 3, 2);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00010);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(1, 3, 3);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00010-psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

%%
figure;
subplot(1, 3, 1);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


subplot(1, 3, 2);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00100);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(1, 3, 3);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00100-psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


%%
figure;
subplot(1, 3, 1);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(1, 3, 2);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00500);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(1, 3, 3);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00500-psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


%%
figure;
subplot(1, 3, 1);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


subplot(1, 3, 2);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_01000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(1, 3, 2);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_imag_part_01000-psi1_imag_part_00000);  % 显示 vy_pre 的图像
title('psi1 imag part noise=0.001');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


