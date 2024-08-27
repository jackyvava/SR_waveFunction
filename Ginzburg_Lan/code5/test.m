clear
clc
%% PARAMETERS
tic;  % 开始计时
lx = 2*pi;
ly = 2*pi;
lz = 2*pi;
nx = 64;
ny = nx;
nz = nx;
vol_size = {lx, ly, lz};   % box size
vol_res = {nx, ny, nz};    % volume resolution
hbar = 0.1;                % Planck constant
Npsi = 2;
clebsch = Clebsch(vol_size{:}, vol_res{:}, hbar, Npsi);
px = clebsch.px;
py = clebsch.py;
pz = clebsch.pz;
dx = clebsch.dx;
dy = clebsch.dy;
dz = clebsch.dz;

% 初始化速度场
[vx, vy, vz] = clebsch.TGVelocityOneForm(); 
[wx, wy, wz] = clebsch.DerivativeOfOneForm(vx, vy, vz); % 速度场导数
wx = wx/dy/dz;
wy = wy/dx/dz;
wz = wz/dx/dy;

%% initial wave function
psi = zeros(nx, ny, nz, Npsi);
for ii = 1:clebsch.Npsi
    psi(:,:,:,ii) = sin(ii*(clebsch.px + clebsch.py + clebsch.pz)) + 1i*cos(ii*(clebsch.px + clebsch.py + clebsch.pz));
end
[psi] = clebsch.Normalize(psi); % 归一化
nstep = 2001;
nsteps = zeros(1, nstep);
deviation = zeros(1, nstep);

% 初始化速度场变量
vx_pre = zeros(nx, ny, nz);
vy_pre = zeros(nx, ny, nz);
vz_pre = zeros(nx, ny, nz);

%% 添加用于保存训练数据的结构
training_data = struct();
training_data.vx_data = zeros(nx, ny, nz, nstep);
training_data.vy_data = zeros(nx, ny, nz, nstep);
training_data.vz_data = zeros(nx, ny, nz, nstep);
training_data.psi_real_data = zeros(nx, ny, nz, Npsi, nstep);
training_data.psi_imag_data = zeros(nx, ny, nz, Npsi, nstep);

%% 迭代求解
output_step = 500;
for iter = 1:nstep
    if (mod(iter, output_step) == 1)
        % 保存当前状态数据到结构体中
        training_data.vx_data(:,:,:,iter) = vx;
        training_data.vy_data(:,:,:,iter) = vy;
        training_data.vz_data(:,:,:,iter) = vz;
        for ii = 1:Npsi
            training_data.psi_real_data(:,:,:,ii,iter) = real(psi(:,:,:,ii));
            training_data.psi_imag_data(:,:,:,ii,iter) = imag(psi(:,:,:,ii));
        end

        % 继续现有代码的输出
        nbox = 18;
        vbox = zeros(nx, ny, nz, nbox);
        vbox(:,:,:,1) = px;
        vbox(:,:,:,2) = py;
        vbox(:,:,:,3) = pz;
        vbox(:,:,:,4) = wx;
        vbox(:,:,:,5) = wy;
        vbox(:,:,:,6) = wz;
        vbox(:,:,:,7) = real(psi(:,:,:,1));
        vbox(:,:,:,8) = imag(psi(:,:,:,1));
        vbox(:,:,:,9) = real(psi(:,:,:,2));
        vbox(:,:,:,10) = vx;
        vbox(:,:,:,11) = vy;
        vbox(:,:,:,12) = vz;
        vbox(:,:,:,13) = vx_pre;
        vbox(:,:,:,14) = vy_pre;
        vbox(:,:,:,15) = vz_pre;
        vbox(:,:,:,16) = vx - vx_pre;
        vbox(:,:,:,17) = vy - vy_pre;
        vbox(:,:,:,18) = vz - vz_pre;
        num2str(iter)
        name = ['TG_', num2str(nx), '_', num2str(iter), '_1', '.bin'];
        varname = ['x', 'y', 'z', 'u', 'v', 'w', 's', 'o', 'r', 'vx', 'vy', 'vz', 'vx_pre', 'vy_pre', 'vz_pre', 'err_x', 'err_y', 'err_z'];
        writedate = output(vbox, nx, ny, nz, nbox, name, varname);
    end
    
    % 更新 vx_pre 等变量
    [vx_pre, vy_pre, vz_pre] = clebsch.VelocityOneForm(psi);

    [Deviation] = clebsch.CalDeviation(vx, vy, vz, psi);
    nsteps(iter) = iter;
    deviation(iter) = Deviation;

    disp('=============')
    disp(['迭代次数', num2str(iter), '的Deviation：', num2str(Deviation)]);
    disp('=============')

    [psi] = clebsch.VelocityOneForm2Psi(vx, vy, vz, psi);
end

%% 保存训练数据到 .mat 文件
save('training_data_2.mat', '-struct', 'training_data', '-v7.3');
disp('Training data saved to training_data.mat');

%% 输出文件
loglog(nsteps, deviation)
[fid, message] = fopen('deviation.dat', 'wb+');
for step = 1:nstep
    fprintf(fid, '%f %f \n', nsteps(step), deviation(step));
end
fclose(fid);

%% Time
elapsedTime = toc;  % 结束计时，并返回时间
disp(['运行时间: ', num2str(elapsedTime), ' 秒']);

%% 波函数转速度场场，并对比误差
[vx_pre, vy_pre, vz_pre] = clebsch.VelocityOneForm(psi);
error = sum(sum(sum((vx - vx_pre).^2 + (vy - vy_pre).^2 + (vz - vz_pre).^2)));
relative_error = error / sum(sum(sum(vx.^2 + vy.^2 + vz.^2)));
disp(['相对误差: ', num2str(relative_error)]);

% 其余绘图代码保持不变
