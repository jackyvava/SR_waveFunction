clear
clc
%% PARAMETERS
lx = 2.*pi;
ly = lx;
nx = 2048;
ny = nx;
vol_size = {lx,ly};   % box size
vol_res = {nx,ny}; % volume resolution
dt = 1./10000;             % time step
tmax = 5;             % max time
nu = 1E-6;
hbar = 1.;


%% INITIALIZATION
ns = NS(vol_size{:},vol_res{:});
ns.dt = dt;
ns.nu = nu;
ns.hbar = hbar;
ns.viscosityMask;
[psi1, psi2] = ns.GetPsi();
[ux, uy] =ns.CalVelFromPsi(psi1,psi2);



colors = [-100:20:100];
px = ns.px;
py = ns.py;


%% MAIN ITERATION
itermax = ceil(tmax/dt);
for iter = 0:itermax
    iter
    if (mod(iter,100)==0)
        [wz] = ns.velTOvor(ux,uy);
        hold off
        hel = contourf(px,py,wz,colors,'linestyle','none');
        axis equal;
        axis([-0.1,lx+0.1,-0.1,ly+0.1]);
        title(['iter = ',num2str(iter)])
        drawnow
        mm = getframe;
        [I, map] = rgb2ind(mm.cdata,256);
        if (iter == 0)
            imwrite(I,map,'vor.gif','DelayTime',0,'LoopCount',Inf)
        end
        imwrite(I,map,'vor.gif','WriteMode','append','DelayTime',0)
    end
    if (mod(iter,100)==0)
        name = ".\data\fluid"+string(iter)+".bin";
        nbox = 5;
        vbox = zeros(ns.resx,ns.resy,1,nbox);
        vbox(:,:,:,1) = ns.px;
        vbox(:,:,:,2) = ns.py;
        vbox(:,:,:,3) = ux;
        vbox(:,:,:,4) = uy;
        vbox(:,:,:,5) = wz;
        varname = ['x','y','u','v','w'];
        writedate = output(vbox,nx,ny,1,nbox,name,varname);
    end
    
    [ux,uy]=ns.NSFlow(ux,uy);
end


