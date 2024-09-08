%% 绘制图片
% 绘制第一个组图 (vx 和 vy)
figure;  % 创建新的图窗口

% 设置colorbar的范围，例如将范围设置为 [-1, 1]
color_range = [-0.01, 0.01];  

subplot(2, 2, 1);  % 创建2行2列的子图，选择第1个子图
imagesc(vy_pre);  % 显示 vy_pre 的图像
title('vy\_pre');
xlabel('X'); ylabel('Y');
colorbar;  % 添加色条
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(2, 2, 2);  % 选择第2个子图
imagesc(vy);  % 显示 vy 的图像
title('vy');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(2, 2, 3);  % 创建2行2列的子图，选择第3个子图
imagesc(vx_pre);  % 显示 vx_pre 的图像
title('vx\_pre');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(2, 2, 4);  % 选择第4个子图
imagesc(vx);  % 显示 vx 的图像
title('vx');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

filename0 = ['velocity_fields_2_', num2str(noise, '%.4f'), '.png'];
% 保存图像为PNG格式
saveas(gcf, filename0); 



%%
% 分别提取波函数的实部和虚部

psi1_real_part = real(psi1);  % 实部
psi1_imag_part = imag(psi1);  % 虚部

psi2_real_part = real(psi2);  % 实部
psi2_imag_part = imag(psi2);  % 虚部
%% 绘图

% 绘制第二个组图 (psi1 和 psi2 的实部与虚部)
figure;  % 创建新的图窗口

% 设置colorbar的范围，例如将范围设置为 [-1, 1]
color_range = [-1, 1];  

subplot(2, 2, 1);  % 创建2行2列的子图，选择第1个子图
imagesc(psi1_real_part);  % 显示 psi1_real_part 的图像
title('psi1 real part');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(2, 2, 2);  % 选择第2个子图
imagesc(psi1_imag_part);  % 显示 psi1_imag_part 的图像
title('psi1 imag part');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(2, 2, 3);  % 创建2行2列的子图，选择第3个子图
imagesc(psi2_real_part);  % 显示 psi2_real_part 的图像
title('psi2 real part');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;

subplot(2, 2, 4);  % 选择第4个子图
imagesc(psi2_imag_part);  % 显示 psi2_imag_part 的图像
title('psi2 imag part');
xlabel('X'); ylabel('Y');
colorbar;
clim(color_range);  % 设置 colorbar 范围
axis equal tight;


filename1 = ['wave_functions_2_', num2str(noise, '%.4f'), '.png'];
% 保存图像为PNG格式
saveas(gcf, filename1);  % 保存图像为 'wave_functions.png'


%%
% 数据
std_dev = [0, 0.0001, 0.001, 0.01, 0.05, 0.1];
run_time = [550.5405, 874.5828, 884.2416, 856.0888, 918.6934, 944.669];
rel_error = [0.097146, 0.041854, 0.040767, 0.43336, 0.5763, 0.6329];

% 创建一个新的图窗口
figure;

% 绘制运行时间的变化
yyaxis left; % 左侧y轴
plot(std_dev, run_time, '-o', 'LineWidth', 1.5, 'Color', 'b'); 
ylabel('Running Time (seconds)');
xlabel('Standard Deviation');
title('Running Time and Relative Error vs Standard Deviation');
grid on;

% 绘制相对误差的变化
yyaxis right; % 右侧y轴
plot(std_dev, rel_error, '-s', 'LineWidth', 1.5, 'Color', 'r'); 
ylabel('Relative Error');

% 图例
legend('Running Time', 'Relative Error');

% 设置x轴为对数坐标
set(gca, 'XScale', 'log');

% 设置图像的其他属性
xlim([min(std_dev), max(std_dev)]);

%%
% 数据
std_dev = [0, 0.0001, 0.001, 0.01, 0.05, 0.1];
run_time = [550.5405, 874.5828, 884.2416, 856.0888, 918.6934, 944.669];
rel_error = [0.097146, 0.041854, 0.040767, 0.43336, 0.5763, 0.6329];

% 创建一个新的图窗口
figure;

% 设置中文字体
set(gca, 'FontName', 'SimHei');  % 使用黑体（SimHei）作为中文字体

% 绘制运行时间的变化
yyaxis left; % 左侧y轴
plot(std_dev, run_time, '-o', 'LineWidth', 1.5, 'Color', 'b'); 
ylabel('运行时间 (秒)', 'FontName', 'SimHei');
xlabel('标准差', 'FontName', 'SimHei');
title('运行时间与相对误差随标准差的变化', 'FontName', 'SimHei');
grid on;

% 绘制相对误差的变化
yyaxis right; % 右侧y轴
plot(std_dev, rel_error, '-s', 'LineWidth', 1.5, 'Color', 'r'); 
ylabel('相对误差', 'FontName', 'SimHei');

% 图例
legend({'运行时间', '相对误差'}, 'FontName', 'SimHei');

% 设置x轴为对数坐标
set(gca, 'XScale', 'log');

% 设置图像的其他属性
xlim([min(std_dev), max(std_dev)]);

% 保存图像为PNG文件
saveas(gcf, 'running_time_vs_error_chinese_2.png');

