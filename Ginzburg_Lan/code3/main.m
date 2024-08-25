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

[vx,vy,vz] = clebsch.TGVelocityOneForm();
[wx,wy,wz] = clebsch.DerivativeOfOneForm(vx,vy,vz);
wx = wx/dy/dz;
wy = wy/dx/dz;
wz = wz/dx/dy;


%%% initial wave function
psi = (randn(nx,ny,nz,Npsi)+1i*randn(nx,ny,nz,Npsi));
[psi] = clebsch.Normalize(psi);
initial_psi = psi;  % 保存初始的 psi 状态

nstep = 101;
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
        name = ['TG_',num2str(nx),'_',num2str(iter),'_1_0','.bin'];
        varname = ['x','y','z','u','v','w','s','o','r'];
        writedate = output(vbox,nx,ny,nz,nbox,name,varname);
    end
    [Deviation] = clebsch.CalDeviation_Original(vx,vy,vz,psi);
    nsteps(iter) = iter;
    deviation(iter) = Deviation;
    [psi] = clebsch.VelocityOneForm2Psi_Original(vx,vy,vz,psi);
end
loglog(nsteps,deviation)
[fid,message] = fopen('deviation.dat','wb+');
for step = 1:nstep
   fprintf(fid,'%f %f \n',nsteps(step),deviation(step));
end
fclose(fid);

elapsedTime = toc;  % 结束计时，并返回时间
disp(['运行时间: ', num2str(elapsedTime), ' 秒']);

save('deviation_before.mat', 'deviation');

%% 比较优化后的代码
% Load deviation from the previous run
load('deviation_before.mat', 'deviation');  % Load deviation from unoptimized code
deviation_before = deviation;

% 重新初始化 psi
psi = initial_psi;

% Initialize new deviation array for optimized code
deviation_after = zeros(1,nstep);
time_cal_deviation = zeros(1, nstep);  % 初始化时间测量数组
time_velocity_oneform2psi = zeros(1, nstep);

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
        name = ['TG_',num2str(nx),'_',num2str(iter),'_1_Youhua','.bin'];
        varname = ['x','y','z','u','v','w','s','o','r'];
        writedate = output(vbox,nx,ny,nz,nbox,name,varname);
    end

    % Measure time for CalDeviation
    tic;
    [Deviation] = clebsch.CalDeviation(vx,vy,vz,psi);
    time_cal_deviation(iter) = toc;
    
    nsteps(iter) = iter;
    deviation_after(iter) = Deviation;  % Store deviation for optimized code
    
    % Measure time for VelocityOneForm2Psi
    tic;
    [psi] = clebsch.VelocityOneForm2Psi(vx,vy,vz,psi);
    time_velocity_oneform2psi(iter) = toc;
end

elapsedTime = toc;  % 结束计时，并返回时间
disp(['运行时间: ', num2str(elapsedTime), ' 秒']);

% Save the deviation for optimized code
save('deviation_after.mat', 'deviation_after');

% Compare deviations (absolute difference)
error_difference = abs(deviation_before - deviation_after);
max_error = max(error_difference);  % Find the maximum error difference
mean_error = mean(error_difference);  % Calculate the mean error difference

% Display results
disp(['Maximum deviation difference: ', num2str(max_error)]);
disp(['Mean deviation difference: ', num2str(mean_error)]);

% Plot comparison
figure;
loglog(nsteps, deviation_before, 'b', nsteps, deviation_after, 'k');
legend('Before Optimization', 'After Optimization');
title('Comparison of Deviation Before and After Optimization');
xlabel('Step');
ylabel('Deviation');

