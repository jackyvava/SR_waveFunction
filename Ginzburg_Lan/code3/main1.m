clear
clc
%% PARAMETERS
tic;  % 开始计时
lx = 2*pi;
ly = 2*pi;
lz = 2*pi;
nx = 64;
ny = 64;
nz = 64;
vol_size = {lx,ly,lz};   % box size
vol_res = {nx,ny,nz}; % volume resolution
hbar = 0.01;            % Planck constant
Npsi = 18;
clebsch = Clebsch(vol_size{:},vol_res{:},hbar,Npsi);
px = clebsch.px;
py = clebsch.py;
pz = clebsch.pz;
dx = clebsch.dx;
dy = clebsch.dy;
dz = clebsch.dz;

[vx,vy,vz] = clebsch.TGVelocityOneForm(); % 初始化速度场
[wx,wy,wz] = clebsch.DerivativeOfOneForm(vx,vy,vz); % 速度场导数
wx = wx/dy/dz;
wy = wy/dx/dz;
wz = wz/dx/dy;

%% initial wave function
psi = (randn(nx,ny,nz,Npsi)+1i*randn(nx,ny,nz,Npsi));% 随机初始化波函数
[psi] = clebsch.Normalize(psi); % 归一化
nstep = 101;
nsteps = zeros(1,nstep);
deviation = zeros(1,nstep);

%% 迭代求解
output_step = 10;
for iter = 1:nstep
    if (mod(iter,output_step) == 1)
        nbox = 9;
        vbox = zeros(nx,ny,nz,nbox);
        vbox(:,:,:,1) = px;
        vbox(:,:,:,2) = py;
        vbox(:,:,:,3) = pz;
        vbox(:,:,:,4) = wx;
        vbox(:,:,:,5) = wy;
        vbox(:,:,:,6) = wz;
        vbox(:,:,:,7) = real(psi(:,:,:,1));
        vbox(:,:,:,8) = imag(psi(:,:,:,1));
        vbox(:,:,:,9) = real(psi(:,:,:,2));
        num2str(iter)
        name = ['TG_',num2str(nx),'_',num2str(iter),'_1','.bin'];
        varname = ['x','y','z','u','v','w','s','o','r'];
        writedate = output(vbox,nx,ny,nz,nbox,name,varname);
    end
    [Deviation] = clebsch.CalDeviation(vx,vy,vz,psi);
    nsteps(iter) = iter;
    deviation(iter) = Deviation;
    [psi] = clebsch.VelocityOneForm2Psi(vx,vy,vz,psi);
end

%% output文件
loglog(nsteps,deviation)
[fid,message] = fopen('deviation.dat','wb+');
for step = 1:nstep
    fprintf(fid,'%f %f \n',nsteps(step),deviation(step));
end
fclose(fid);

%% Time
elapsedTime = toc;  % 结束计时，并返回时间
disp(['运行时间: ', num2str(elapsedTime), ' 秒']);

%% 计算波函数转速度场，并对比误差
% 从波函数计算速度场
[vx_pre, vy_pre, vz_pre] = clebsch.CalVelFromPsi(psi);

%% 计算每个方向的误差
error_x = vx - vx_pre;
error_y = vy - vy_pre;
error_z = vz - vz_pre;  % 即便z方向不考虑，计算误差也可以保持一致性

%% 计算相对误差
% Z方向不考虑相对误差，因此令其为0
relative_error_z = abs(error_z) * 0;

% 分子求和 (绝对误差求和)
numerator_x = sum(abs(error_x(:)));
numerator_y = sum(abs(error_y(:)));

% 分母求和 (真值绝对值求和)
denominator_x = sum(abs(vx(:)));
denominator_y = sum(abs(vy(:)));

% 计算整体的相对误差
average_relative_error_x = numerator_x / denominator_x;
average_relative_error_y = numerator_y / denominator_y;

%% 计算不同类型的误差
% 计算整体误差的L2范数
total_error = sqrt(sum(error_x(:).^2 + error_y(:).^2 + error_z(:).^2));

% 计算平均误差 (绝对误差的平均值)
average_error = mean(abs(error_x(:)) + abs(error_y(:)) + abs(error_z(:)));

% 计算每个方向的RMSE
rmse_x = sqrt(mean(error_x(:).^2));
rmse_y = sqrt(mean(error_y(:).^2));
rmse_z = sqrt(mean(error_z(:).^2));  % Z方向尽管未考虑，也计算误差以保持完整性

% 计算总体的RMSE（结合所有方向的误差）
total_rmse = sqrt(mean([error_x(:).^2; error_y(:).^2; error_z(:).^2]));

%% 输出误差结果
disp(['波函数转速度场误差 (L2范数): ', num2str(total_error)]);
disp(['平均误差: ', num2str(average_error)]);
disp(['RMSE (X方向): ', num2str(rmse_x)]);
disp(['RMSE (Y方向): ', num2str(rmse_y)]);
disp(['RMSE (Z方向): ', num2str(rmse_z)]);
disp(['总体RMSE: ', num2str(total_rmse)]);

%% 输出平均相对误差
disp(['平均相对误差 (X方向): ', num2str(average_relative_error_x * 100), '%']);
disp(['平均相对误差 (Y方向): ', num2str(average_relative_error_y * 100), '%']);
disp(['平均相对误差 (Z方向): ', num2str(0), '% (Z方向未计算)']);


%% 绘图

% 定义数据
x = [2, 4, 6, 8, 10, 18];
y1 = [0.098445, 0.094626, 0.093837, 0.093861, 0.093218,0.092906]; % RMSE
y2 = [0.19843, 0.18853, 0.18641, 0.18589, 0.1849,0.18407]; % MSE

% 创建图形
figure;

% 绘制左边的Y轴 (Y1)
yyaxis left
plot(x, y1, '-o', 'LineWidth', 2);
ylabel('RMSE'); % 左Y轴标签

% 绘制右边的Y轴 (Y2)
yyaxis right
plot(x, y2, '-s', 'LineWidth', 2);
ylabel('MSE'); % 右Y轴标签

% 设置X轴标签和标题
xlabel('Npsi');
title('误差随Npsi变化');

% 显示网格
grid on;

% 显示图例
legend('RMSE', 'MSE', 'Location', 'best');




