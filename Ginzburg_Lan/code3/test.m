clc;
clear;

% 测试参数
nx = 8;
ny = 8;
nz = 8;
Npsi = 2;
hbar = 0.01;

% 初始化 Clebsch 对象
clebsch = Clebsch(2*pi, 2*pi, 2*pi, nx, ny, nz, hbar, Npsi);

% 初始化 vx, vy, vz 和 psi
[vx, vy, vz] = clebsch.TGVelocityOneForm();
psi = (randn(nx, ny, nz, Npsi) + 1i * randn(nx, ny, nz, Npsi));
psi = clebsch.Normalize(psi);

% 运行原始版本的 VelocityOneForm2Psi
disp('Running original version...');
psi_original = psi;
psi_original = clebsch.VelocityOneForm2Psi_Original(vx, vy, vz, psi_original);

% 运行优化版本的 VelocityOneForm2Psi
disp('Running optimized version...');
psi_optimized = psi;
psi_optimized = clebsch.VelocityOneForm2Psi(vx, vy, vz, psi_optimized);

% 比较结果
difference = norm(psi_original(:) - psi_optimized(:));
disp(['Difference between original and optimized results: ', num2str(difference)]);



% 运行原始版本
disp('Running original version...');
deviation_original = clebsch.CalDeviation_Original(vx, vy, vz, psi);

% 运行优化版本
disp('Running optimized version...');
deviation_optimized = clebsch.CalDeviation(vx, vy, vz, psi);

% 比较结果
difference = abs(deviation_original - deviation_optimized);
disp(['Difference between original and optimized deviations: ', num2str(difference)]);



% 可视化对比
figure;
subplot(1,2,1);
imagesc(real(psi_original(:,:,1,1)));
colorbar;
title('Original Psi (Real Part)');

subplot(1,2,2);
imagesc(real(psi_optimized(:,:,1,1)));
colorbar;
title('Optimized Psi (Real Part)');
