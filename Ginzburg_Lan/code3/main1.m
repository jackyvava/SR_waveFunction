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
Npsi = 8;
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


%%% initial wave function
psi = (randn(nx,ny,nz,Npsi)+1i*randn(nx,ny,nz,Npsi));% 随机初始化波函数
[psi] = clebsch.Normalize(psi); % 归一化
nstep = 201;
nsteps = zeros(1,nstep);
deviation = zeros(1,nstep);

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
loglog(nsteps,deviation)
[fid,message] = fopen('deviation.dat','wb+');
for step = 1:nstep
   fprintf(fid,'%f %f \n',nsteps(step),deviation(step));
end
fclose(fid);

elapsedTime = toc;  % 结束计时，并返回时间
disp(['运行时间: ', num2str(elapsedTime), ' 秒']);

[vx_pre,vy_pre,vz_pre] = clebsch.CalVelFromPsi(psi);

error_x = vx - vx_pre;
error_y = vy - vy_pre;
error_z = vz - vz_pre;

% 计算整体误差的L2范数
total_error = sqrt(sum(error_x(:).^2 + error_y(:).^2 + error_z(:).^2));

disp(['波函数转速度场误差: ', num2str(total_error)]);