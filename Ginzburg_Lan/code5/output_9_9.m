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
% 将变量名存储在一个单元数组中
variable_names = {
    'psi1_real_part_00000', 'psi1_real_part_00001', 'psi1_real_part_00010', 'psi1_real_part_00100', ...
    'psi1_real_part_00500', 'psi1_real_part_01000', 'psi1_imag_part_00000', 'psi1_imag_part_00001', ...
    'psi1_imag_part_00010', 'psi1_imag_part_00100', 'psi1_imag_part_00500', 'psi1_imag_part_01000', ...
    'psi2_real_part_00000', 'psi2_real_part_00001', 'psi2_real_part_00010', 'psi2_real_part_00100', ...
    'psi2_real_part_00500', 'psi2_real_part_01000', 'psi2_imag_part_00000', 'psi2_imag_part_00001', ...
    'psi2_imag_part_00000', 'psi2_imag_part_00100', 'psi2_imag_part_00500', 'psi2_imag_part_01000'
};
% 对每个变量进行处理
for i = 1:length(variable_names)
    % 获取当前变量的值
    current_var = eval(variable_names{i});
    
    % 执行你需要的处理，例如归一化
    min_val = min(current_var(:));
    max_val = max(current_var(:));
    
    % 归一化到 [-1, 1]
    normalized_var = (current_var - min_val) / (max_val - min_val) * 2 - 1;
    
    % 将处理后的值存回原来的变量中
    eval([variable_names{i}, ' = normalized_var;']);
end



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
vx_pre_delta_1 = vx_pre00001 - vx00000;
vx_pre_delta_2 = vx_pre00010 - vx00000;
vx_pre_delta_3 = vx_pre00100 - vx00000;
vx_pre_delta_4 = vx_pre00500 - vx00000;
vx_pre_delta_5 = vx_pre01000 - vx00000;

% 计算uy的原始误差
vy_delta_1 = vy00001 - vy00000;
vy_delta_2 = vy00010 - vy00000;
vy_delta_3 = vy00100 - vy00000;
vy_delta_4 = vy00500 - vy00000;
vy_delta_5 = vy01000 - vy00000;


%计算uy的预测误差
vy_pre_delta_1 = vy_pre00001 - vy00000;
vy_pre_delta_2 = vy_pre00010 - vy00000;
vy_pre_delta_3 = vy_pre00100 - vy00000;
vy_pre_delta_4 = vy_pre00500 - vy00000;
vy_pre_delta_5 = vy_pre01000 - vy00000;

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

vx_pre_SSIM_1 = ssim(vx_pre00000, vx00000);
vx_pre_SSIM_2 = ssim(vx_pre00001, vx00000);
vx_pre_SSIM_3 = ssim(vx_pre00010, vx00000);
vx_pre_SSIM_4 = ssim(vx_pre00100, vx00000);
vx_pre_SSIM_5 = ssim(vx_pre00500, vx00000);
vx_pre_SSIM_6 = ssim(vx_pre01000, vx00000);

vy_pre_SSIM_1 = ssim(vy_pre00000, vy00000);
vy_pre_SSIM_2 = ssim(vy_pre00001, vy00000);
vy_pre_SSIM_3 = ssim(vy_pre00010, vy00000);
vy_pre_SSIM_4 = ssim(vy_pre00100, vy00000);
vy_pre_SSIM_5 = ssim(vy_pre00500, vy00000);
vy_pre_SSIM_6 = ssim(vy_pre01000, vy00000);

%% 可是化SSIM折线图
% 计算xy轴
ssim_vx = [vx_SSIM_1,vx_SSIM_2, vx_SSIM_3,vx_SSIM_4,vx_SSIM_5, vx_SSIM_6];
ssim_vy = [vy_SSIM_1,vy_SSIM_2, vy_SSIM_3,vy_SSIM_4,vy_SSIM_5, vy_SSIM_6];

ssim_vx_pre = [vx_pre_SSIM_1,vx_pre_SSIM_2, vx_pre_SSIM_3,vx_pre_SSIM_4,vx_pre_SSIM_5, vx_pre_SSIM_6];
ssim_vy_pre = [vy_pre_SSIM_1,vy_pre_SSIM_2, vy_pre_SSIM_3,vy_pre_SSIM_4,vy_pre_SSIM_5, vy_pre_SSIM_6];


run_time = [0,0.0001,0.001,0.01,0.05,0.1];

% 绘图
figure;

% 创建子图1：vx的SSIM对比
subplot(2, 1, 1);  % 2行1列的子图，绘制第1个
plot(run_time, ssim_vx, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vx', 'Color', [0, 0.4470, 0.7410]);  % 蓝色线
hold on;
plot(run_time, ssim_vx_pre, '-d', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vx pre', 'Color', [0.8500, 0.3250, 0.0980]);  % 红色线
legend('show', 'Location', 'northeast', 'FontSize', 10);  % 图例放在右上角，字体大小为10
xlabel('Noise', 'FontSize', 12, 'FontWeight', 'bold');  % X轴标签
ylabel('SSIM', 'FontSize', 12, 'FontWeight', 'bold');    % Y轴标签
title('SSIM vs Runtime for vx', 'FontSize', 14, 'FontWeight', 'bold');  % 标题
grid on;
set(gca, 'FontSize', 12);  % 设置轴字体大小

% 创建子图2：vy的SSIM对比
subplot(2, 1, 2);  % 2行1列的子图，绘制第2个
plot(run_time, ssim_vy, '-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vy', 'Color', [0.4660, 0.6740, 0.1880]);  % 绿色线
hold on;
plot(run_time, ssim_vy_pre, '-x', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vy pre', 'Color', [0.9290, 0.6940, 0.1250]);  % 黄色线
legend('show', 'Location', 'northeast', 'FontSize', 10);  % 图例放在右上角，字体大小为10
xlabel('Noise', 'FontSize', 12, 'FontWeight', 'bold');  % X轴标签
ylabel('SSIM', 'FontSize', 12, 'FontWeight', 'bold');    % Y轴标签
title('SSIM vs Runtime for vy', 'FontSize', 14, 'FontWeight', 'bold');  % 标题
grid on;
set(gca, 'FontSize', 12);  % 设置轴字体大小

% 调整图形大小使其适合论文排版
set(gcf, 'Position', [100, 100, 600, 600]);  % 设置图形大小

% 保存为高分辨率的PNG或EPS格式，用于论文插图
saveas(gcf, 'SSIM_vs_Runtime_12.png');



%% PSNR计算

% 计算PSNR曲线
vx_PSNR_1 = psnr(vx00000, vx00000);
vx_PSNR_2 = psnr(vx00001, vx00000);
vx_PSNR_3 = psnr(vx00010, vx00000);
vx_PSNR_4 = psnr(vx00100, vx00000);
vx_PSNR_5 = psnr(vx00500, vx00000);
vx_PSNR_6 = psnr(vx01000, vx00000);

vy_PSNR_1 = psnr(vy00000, vy00000);
vy_PSNR_2 = psnr(vy00001, vy00000);
vy_PSNR_3 = psnr(vy00010, vy00000);
vy_PSNR_4 = psnr(vy00100, vy00000);
vy_PSNR_5 = psnr(vy00500, vy00000);
vy_PSNR_6 = psnr(vy01000, vy00000);

vx_pre_PSNR_1 = psnr(vx_pre00000, vx00000);
vx_pre_PSNR_2 = psnr(vx_pre00001, vx00000);
vx_pre_PSNR_3 = psnr(vx_pre00010, vx00000);
vx_pre_PSNR_4 = psnr(vx_pre00100, vx00000);
vx_pre_PSNR_5 = psnr(vx_pre00500, vx00000);
vx_pre_PSNR_6 = psnr(vx_pre01000, vx00000);

vy_pre_PSNR_1 = psnr(vy_pre00000, vy00000);
vy_pre_PSNR_2 = psnr(vy_pre00001, vy00000);
vy_pre_PSNR_3 = psnr(vy_pre00010, vy00000);
vy_pre_PSNR_4 = psnr(vy_pre00100, vy00000);
vy_pre_PSNR_5 = psnr(vy_pre00500, vy00000);
vy_pre_PSNR_6 = psnr(vy_pre01000, vy00000);

%% 可视化PSNR折线图
% 计算xy轴
psnr_vx = [100,vx_PSNR_2, vx_PSNR_3,vx_PSNR_4,vx_PSNR_5, vx_PSNR_6];
psnr_vy = [100,vy_PSNR_2, vy_PSNR_3,vy_PSNR_4,vy_PSNR_5, vy_PSNR_6];

psnr_vx_pre = [100,vx_pre_PSNR_2, vx_pre_PSNR_3,vx_pre_PSNR_4,vx_pre_PSNR_5, vx_pre_PSNR_6];
psnr_vy_pre = [100,vy_pre_PSNR_2, vy_pre_PSNR_3,vy_pre_PSNR_4,vy_pre_PSNR_5, vy_pre_PSNR_6];

run_time = [1,2,3,4,5,6];

%% 合并图
figure;

% 创建子图1：vx的PSNR对比
subplot(2, 2, 1);  % 2行2列的子图，绘制第1个
plot(run_time, psnr_vx, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'PSNR vx', 'Color', [0, 0.4470, 0.7410]);  % 蓝色线
hold on;
plot(run_time, psnr_vx_pre, '-d', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'PSNR vx pre', 'Color', [0.8500, 0.3250, 0.0980]);  % 红色线
legend('show', 'Location', 'southwest', 'FontSize', 10);  % 图例放在左下角
xlabel('Noise', 'FontSize', 12, 'FontWeight', 'bold');  % X轴标签
ylabel('PSNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');    % Y轴标签
title('PSNR vs Noise for vx', 'FontSize', 14, 'FontWeight', 'bold');  % 标题
grid on;
set(gca, 'FontSize', 12);  % 设置轴字体大小
text(0.9, 0.95, '(a)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');  % 添加(a)

% 创建子图2：vy的PSNR对比
subplot(2, 2, 2);  % 2行2列的子图，绘制第2个
plot(run_time, psnr_vy, '-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'PSNR vy', 'Color', [0.4660, 0.6740, 0.1880]);  % 绿色线
hold on;
plot(run_time, psnr_vy_pre, '-x', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'PSNR vy pre', 'Color', [0.9290, 0.6940, 0.1250]);  % 黄色线
legend('show', 'Location', 'southwest', 'FontSize', 10);  % 图例放在左下角
xlabel('Noise', 'FontSize', 12, 'FontWeight', 'bold');  % X轴标签
ylabel('PSNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');    % Y轴标签
title('PSNR vs Noise for vy', 'FontSize', 14, 'FontWeight', 'bold');  % 标题
grid on;
set(gca, 'FontSize', 12);  % 设置轴字体大小
text(0.9, 0.95, '(b)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');  % 添加(b)

% 创建子图3：vx的SSIM对比
subplot(2, 2, 3);  % 2行2列的子图，绘制第3个
plot(run_time, ssim_vx, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vx', 'Color', [0, 0.4470, 0.7410]);  % 蓝色线
hold on;
plot(run_time, ssim_vx_pre, '-d', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vx pre', 'Color', [0.8500, 0.3250, 0.0980]);  % 红色线
legend('show', 'Location', 'southwest', 'FontSize', 10);  % 图例放在左下角
xlabel('Noise', 'FontSize', 12, 'FontWeight', 'bold');  % X轴标签
ylabel('SSIM', 'FontSize', 12, 'FontWeight', 'bold');    % Y轴标签
title('SSIM vs Noise for vx', 'FontSize', 14, 'FontWeight', 'bold');  % 标题
grid on;
set(gca, 'FontSize', 12);  % 设置轴字体大小
text(0.9, 0.95, '(c)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');  % 添加(c)

% 创建子图4：vy的SSIM对比
subplot(2, 2, 4);  % 2行2列的子图，绘制第4个
plot(run_time, ssim_vy, '-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vy', 'Color', [0.4660, 0.6740, 0.1880]);  % 绿色线
hold on;
plot(run_time, ssim_vy_pre, '-x', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SSIM vy pre', 'Color', [0.9290, 0.6940, 0.1250]);  % 黄色线
legend('show', 'Location', 'southwest', 'FontSize', 10);  % 图例放在左下角
xlabel('Noise', 'FontSize', 12, 'FontWeight', 'bold');  % X轴标签
ylabel('SSIM', 'FontSize', 12, 'FontWeight', 'bold');    % Y轴标签
title('SSIM vs Noise for vy', 'FontSize', 14, 'FontWeight', 'bold');  % 标题
grid on;
set(gca, 'FontSize', 12);  % 设置轴字体大小
text(0.9, 0.95, '(d)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');  % 添加(d)

% 调整图形大小适合论文排版
set(gcf, 'Position', [100, 100, 800, 600]);  % 设置图形大小为800x600像素

% 保存为高分辨率的PNG或EPS格式，用于论文插图
saveas(gcf, 'PSNR_SSIM_vs_Noise_with_labels_12.png');

%%

% %data_diff  = psi1_imag_delta_1;
% 
% figure;
% 
% % 设置子图的间距
% tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
% 
% % 子图 1
% nexttile;
% data_diff  = vx_pre_delta_1;
% data = data_diff(:);
% histogram(data, 'Normalization', 'pdf', 'FaceColor', [0.3 0.7 0.9], 'EdgeColor', 'none');
% mu = mean(data);
% sigma = std(data);
% x_values = linspace(min(data), max(data), 100);
% y_values = normpdf(x_values, mu, sigma);
% hold on;
% plot(x_values, y_values, 'r', 'LineWidth', 2);
% hold off;
% xlabel('Data Values', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
% title('vx pre delta 1', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
% text(min(x_values) - 0.1*abs(min(x_values)), max(y_values), '(a)', 'FontSize', 14, 'FontWeight', 'bold');  % 在左上角添加(a)
% 
% % 子图 2
% nexttile;
% data_diff  = vy_delta_1;
% data = data_diff(:);
% histogram(data, 'Normalization', 'pdf', 'FaceColor', [0.5 0.8 0.4], 'EdgeColor', 'none');
% mu = mean(data);
% sigma = std(data);
% x_values = linspace(min(data), max(data), 100);
% y_values = normpdf(x_values, mu, sigma);
% hold on;
% plot(x_values, y_values, 'r', 'LineWidth', 2);
% hold off;
% xlabel('Data Values', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
% title('vy delta 1', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
% text(min(x_values) - 0.1*abs(min(x_values)), max(y_values), '(b)', 'FontSize', 14, 'FontWeight', 'bold');  % 在左上角添加(b)
% 
% % 子图 3
% nexttile;
% data_diff  = psi1_imag_delta_1;
% data = data_diff(:);
% histogram(data, 'Normalization', 'pdf', 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'none');
% mu = mean(data);
% sigma = std(data);
% x_values = linspace(min(data), max(data), 100);
% y_values = normpdf(x_values, mu, sigma);
% hold on;
% plot(x_values, y_values, 'r', 'LineWidth', 2);
% hold off;
% xlabel('Data Values', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
% title('psi1 imag delta 1', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
% text(min(x_values) - 0.1*abs(min(x_values)), max(y_values), '(c)', 'FontSize', 14, 'FontWeight', 'bold');  % 在左上角添加(c)
% 
% % 子图 4
% nexttile;
% data_diff  = psi1_real_delta_1;
% data = data_diff(:);
% histogram(data, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.7], 'EdgeColor', 'none');
% mu = mean(data);
% sigma = std(data);
% x_values = linspace(min(data), max(data), 100);
% y_values = normpdf(x_values, mu, sigma);
% hold on;
% plot(x_values, y_values, 'r', 'LineWidth', 2);
% hold off;
% xlabel('Data Values', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
% title('psi1 real delta 1', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
% text(min(x_values) - 0.1*abs(min(x_values)), max(y_values), '(d)', 'FontSize', 14, 'FontWeight', 'bold');  % 在左上角添加(d)
% 
% % 调整整体布局，使图形更加协调
% sgtitle('Histograms with Normal Distribution Fits', 'FontSize', 16, 'FontWeight', 'bold');
% 
% % 保存图像
% saveas(gcf, 'Histograms_with_Normal_Distribution_Fits.png');
%%
% 计算两个数据集的直方图数据
% 计算两个数据集的直方图数据
test0 = (vx00500 - vx00000);
test = vx_pre00500 - vx00000;

% 绘制直方图时使用相同的 bin 范围
figure;
h1 = histogram(test0, 'Normalization', 'count'); % 绘制第一个直方图
hold on;
h2 = histogram(test, 'Normalization', 'count'); % 绘制第二个直方图

% 调整颜色为常用科学配色
h1.FaceColor = [0, 0.4470, 0.7410];  % 蓝色
h2.FaceColor = [0.8500, 0.3250, 0.0980];  % 橙色

% 调整透明度以便重叠部分可见
h1.FaceAlpha = 0.6;
h2.FaceAlpha = 0.6;

% 设置y轴频数一样
max_y = max([h1.Values, h2.Values]); % 获取两个直方图的最大频数
ylim([0, max_y]); % 设置 y 轴范围一致

% 添加图例和标签
legend({'test0', 'test'}, 'FontSize', 12, 'Location', 'northeast');
xlabel('Data Values', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Combined Histograms of test0 and test', 'FontSize', 14);

% 显示网格线
grid on;
hold off;
