% 生成数据
nstep = 100;  % 少量数据用于初始训练
vx_data = zeros(nx, ny, nz, nstep);
vy_data = zeros(nx, ny, nz, nstep);
vz_data = zeros(nx, ny, nz, nstep);
psi_data = zeros(nx, ny, nz, Npsi, nstep);
deviation_data = zeros(1, nstep);

for iter = 1:nstep
    % 生成随机初始速度场
    [vx,vy,vz] = clebsch.TGVelocityOneForm();
    psi = zeros(nx,ny,nz,Npsi);
    for ii = 1:clebsch.Npsi
        psi(:,:,:,ii) = sin(ii*(clebsch.px+clebsch.py+clebsch.pz))+1i*cos(ii*(clebsch.px+clebsch.py+clebsch.pz));
    end
    psi = clebsch.Normalize(psi);

    % 计算偏差
    [Deviation] = clebsch.CalDeviation(vx,vy,vz,psi);
    
    % 存储数据
    vx_data(:,:,:,iter) = vx;
    vy_data(:,:,:,iter) = vy;
    vz_data(:,:,:,iter) = vz;
    psi_data(:,:,:,:,iter) = psi;
    deviation_data(iter) = Deviation;
end

% 保存数据
save('data.mat', 'vx_data', 'vy_data', 'vz_data', 'psi_data', 'deviation_data', '-v7.3');

