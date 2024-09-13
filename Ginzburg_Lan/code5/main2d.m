clear;
clc;
%% PARAMETERS
tic;  % 开始计时
lx = 2*pi;
ly = 2*pi;
nx = 41;
ny = nx;
vol_size = {lx, ly};   % box size
vol_res = {nx, ny};    % volume resolution
hbar = 0.1;            % Planck constant
Npsi = 2;
clebsch = Clebsch2d(vol_size{:}, vol_res{:}, hbar, Npsi);
px = clebsch.px;
py = clebsch.py;
dx = clebsch.dx;
dy = clebsch.dy;
noise = 0.0000;

%[vx, vy] = clebsch.TGVelocityOneForm(); % TG涡

%[vx00001, vy00001] = clebsch.TGVelocityOneForm_noise(noise); % TG涡+noise

[vx00001, vy00001] = clebsch.DealtWing();
% [vx00001, vy00001] =clebsch.RandomVortexFlow;

[wx, wy] = clebsch.DerivativeOfOneForm(vx00001, vy00001); % 速度场导数
wx = wx / dy;
wy = wy / dx;

%% initial wave function
psi00001 = zeros(nx, ny, Npsi);
for ii = 1:clebsch.Npsi
    real_part = sin(rand(size(clebsch.px)));
    imag_part = cos(rand(size(clebsch.py)));
    psi00001(:, :, ii) = real_part + 1i * imag_part;  % 随机复数矩阵
    % psi00001(:, :, ii) = sin(ii * (clebsch.px + clebsch.py)) + 1i * cos(ii * (clebsch.px + clebsch.py));
end
psi00001 = clebsch.Normalize(psi00001); % 归一化
nstep = 5001;
nsteps = zeros(1, nstep);
deviation = zeros(1, nstep);

%% Initialize velocity fields before the loop
vx_pre00001 = zeros(nx, ny);
vy_pre00001 = zeros(nx, ny);

%% 迭代求解
output_step = 500;
for iter = 1:nstep
    % if (mod(iter, output_step) == 1)
    %     nbox = 10;
    %     vbox = zeros(nx, ny, nbox);
    %     vbox(:, :, 1) = px;
    %     vbox(:, :, 2) = py;
    %     vbox(:, :, 3) = wx;
    %     vbox(:, :, 4) = wy;
    %     vbox(:, :, 5) = real(psi(:, :, 1));
    %     vbox(:, :, 6) = imag(psi(:, :, 1));
    %     vbox(:, :, 7) = real(psi(:, :, 2));
    %     vbox(:, :, 8) = vx;
    %     vbox(:, :, 9) = vy;
    %     vbox(:, :, 10) = vx_pre;
    % 
    %     name = ['TG_', num2str(nx), '_', num2str(iter), '_1', '.bin'];
    %     varname = {'x', 'y', 'u', 'v', 's', 'o', 'r', 'vx', 'vy', 'vx_pre'};
    %     writedate = output(vbox, nx, ny, nbox, name, varname);
    % end
    
    % 更新 vx_pre 和 vy_pre 变量
    [vx_pre00001, vy_pre00001] = clebsch.VelocityOneForm(psi00001);

    % 计算偏差
    Deviation = clebsch.CalDeviation(vx00001, vy00001, psi00001);
    nsteps(iter) = iter;
    deviation(iter) = Deviation;

    disp(['=============', 'Iteration ', num2str(iter), ' Deviation: ', num2str(Deviation), '=============']);
    
    % 更新 psi
    psi00001 = clebsch.VelocityOneForm2Psi(vx00001, vy00001, psi00001);
end

%% output文件
loglog(nsteps, deviation);
[fid, message] = fopen('deviation.dat', 'wb+');
for step = 1:nstep
    fprintf(fid, '%f %f \n', nsteps(step), deviation(step));
end
fclose(fid);

%% Time
elapsedTime = toc;  % 结束计时，并返回时间
disp(['运行时间: ', num2str(elapsedTime), ' 秒']);

%% 波函数转速度场场，并对比误差
[vx_pre00001, vy_pre00001] = clebsch.VelocityOneForm(psi00001);
error = sum(sum((vx00001 - vx_pre00001).^2 + (vy00001 - vy_pre00001).^2));
relative_error = error / sum(sum(vx00001.^2 + vy00001.^2));
disp(['相对误差: ', num2str(relative_error)]);



%% 保存数据集

% 指定数据存储路径
data_dir = 'D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\Clebsch_flowfield\data\High_Resolution\';

% 检查路径是否存在，不存在则创建
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

% 保存速度场数据 u_x 和 u_y 到 MAT 文件
ux = vx00001;
uy = vy00001;
save([data_dir 'velocity_field_deltwing.mat'], 'ux', 'uy');

% 保存波函数 psi_1 和 psi_2 到 MAT 文件
psi1 = real(psi00001(:, :, 1)) + 1i * imag(psi00001(:, :, 1));
psi2 = real(psi00001(:, :, 2)) + 1i * imag(psi00001(:, :, 2));
save([data_dir 'wave_function_deltwing.mat'], 'psi1', 'psi2');

% 计算并保存误差场 v_x 和 v_y 到 MAT 文件
vx_error = vx00001 - vx_pre00001;
vy_error = vy00001 - vy_pre00001;
save([data_dir 'error_field_deltwing.mat'], 'vx_error', 'vy_error');

% 输出文件保存完成信息
disp(['数据已成功保存为MAT格式到文件夹: ' data_dir]);

psi1_real_part_00001 = real(psi1);  % 实部
psi1_imag_part_00001 = imag(psi1);  % 虚部

psi2_real_part_00001 = real(psi2);  % 实部
psi2_imag_part_00001 = imag(psi2);  % 虚部

save('data00001_deltwing.mat','vx00001', 'vx_pre00001','vy00001','vy_pre00001','psi00001','psi1_real_part_00001','psi1_imag_part_00001','psi2_real_part_00001',"psi2_imag_part_00001")



figure 
imagesc(vx_pre00001)

figure 
imagesc(vx00001)