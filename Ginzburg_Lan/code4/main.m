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
vol_size = {lx,ly,lz};   % box size
vol_res = {nx,ny,nz}; % volume resolution
hbar = 1.;            % Planck constant
Npsi = 2;
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
nstep = 10001;
nsteps = zeros(1,nstep);
deviation = zeros(1,nstep);

%% 迭代求解
output_step = 1000;
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

%% 波函数转速度场场，并对比误差
[vx_pre, vy_pre, vz_pre] = clebsch.VelocityOneForm(psi);
error = sum(sum(sum((vx - vx_pre).^2+(vy - vy_pre).^2+(vz - vz_pre).^2)));
relative_error = error/sum(sum(sum(vx.^2+vy.^2+vz.^2)));
disp(['相对误差: ', num2str(relative_error)]);

% 
% %% 绘图
% 
% % 定义数据
% x = [2, 4, 6, 8, 10, 18];
% y1 = [0.098445, 0.094626, 0.093837, 0.093861, 0.093218,0.092906]; % RMSE
% y2 = [0.19843, 0.18853, 0.18641, 0.18589, 0.1849,0.18407]; % MSE
% 
% % 创建图形
% figure;
% 
% % 绘制左边的Y轴 (Y1)
% yyaxis left
% plot(x, y1, '-o', 'LineWidth', 2);
% ylabel('RMSE'); % 左Y轴标签
% 
% % 绘制右边的Y轴 (Y2)
% yyaxis right
% plot(x, y2, '-s', 'LineWidth', 2);
% ylabel('MSE'); % 右Y轴标签
% 
% % 设置X轴标签和标题
% xlabel('Npsi');
% title('误差随Npsi变化');
% 
% % 显示网格
% grid on;
% 
% % 显示图例
% legend('RMSE', 'MSE', 'Location', 'best');
% 


