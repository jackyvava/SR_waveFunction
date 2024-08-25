clear
clc
%% PARAMETERS
tic;
lx = 2*pi;
ly = 2*pi;
vol_size = {lx,ly};   % box size
nx = 128;
ny = nx;
vol_res = {nx,ny}; % volume resolution
Gamma = 1;
hbar = Gamma/(4*pi);
Npsi = 20;
vol_par = {hbar,Npsi};
%% INITIALIZATION
Opt = operator(vol_size{:},vol_res{:},vol_par{:});
[w3] = cos(2*Opt.px).*cos(2*Opt.py);
[u1, u2] = Opt.CalVelFromVor(w3);
[psi] = Opt.GetRandPsi();
rand_lambda = 0;
u1_rand = u1+rand_lambda*randn(Opt.resx,Opt.resy);
u2_rand = u2+rand_lambda*randn(Opt.resx,Opt.resy);
u1_hbar = 2*u1_rand./Opt.hbar;
u2_hbar = 2*u2_rand./Opt.hbar;
nstep =1000;
dt = 0.01;
plot_iter=0;
Q = zeros(Opt.resx,Opt.resy,Opt.Npsi);
for iter = 1:nstep
    iter
    plot_iter=plot_iter+1;
    [Energy] = Opt.CalDeviation(u1,u2,psi);
    energy(plot_iter)=Energy;
    "energy ="
    Energy
    P = zeros(Opt.resx,Opt.resy);
    R = zeros(Opt.resx,Opt.resy);
    for ii = 1:Opt.Npsi
        phi = fftshift(fftn(psi(:,:,ii)));
        psix = ifftn(fftshift(phi.*Opt.kx));
        psiy = ifftn(fftshift(phi.*Opt.ky));
%         [ux, uy] = Opt.CalVelFromPsi(psi);
%         ux_drive = u1_hbar - (1-epsilon)*ux/hbar;
%         uy_drive = u2_hbar - (1-epsilon)*uy/hbar;
        Q(:,:,ii) = -2i*(psix.*u1_hbar + psiy.*u2_hbar)*dt;
        P = P + real(psi(:,:,ii).*conj(Q(:,:,ii)));
        R = R + abs(Q(:,:,ii)).^2;
    end
    R =  sqrt(1 + P.*P - R)-real(P);
    fac = exp(dt*Opt.k20);
    for ii = 1:Opt.Npsi
        psi(:,:,ii) = ifftn(fftshift(fftshift(fftn(R.*psi(:,:,ii)+Q(:,:,ii))).*fac));
    end
    [psi] = Opt.Normalize(psi);
    [psi] = Opt.PressureProject(psi);
end
t=toc;
plot(1:plot_iter,energy,'r');
xlabel('迭代次数'); % 设置横坐标标题
ylabel('能量'); % 设置纵坐标标题

[ux, uy] = Opt.CalVelFromPsi(psi);
[wz] = Opt.CalVorFromVel(ux,uy);
[wz_rand] = Opt.CalVorFromVel(u1_rand,u2_rand);
errorMagnitude = sqrt((w3-wz).^2);
rmse = sqrt(mean(errorMagnitude(:).^2))%均方根误差
mae = mean(abs(errorMagnitude(:)))%平均绝对误差
maxError = max(abs(errorMagnitude(:)))


nbox = 7;
vbox = zeros(nx, ny, 1,nbox);
vbox(:,:,1,1) = Opt.px;
vbox(:,:,1,2) = Opt.py;
vbox(:,:,1,3) = u1;
vbox(:,:,1,4) = u2;
vbox(:,:,1,5) = w3;
vbox(:,:,1,6) = wz;
vbox(:,:,1,7) = wz_rand;
name = 'abcd512.bin';
varname = ['x','y','w','u','a','b','c'];
writedate = output(vbox,nx, ny, 1,nbox,name,varname);