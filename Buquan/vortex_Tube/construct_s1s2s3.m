clear;
clc;

sizex = 2*pi;
sizey = 2*pi;
sizez = 2*pi;

resolution = 128;
resx = resolution;
resy = resolution;
resz = resolution;
dx = sizex/resx;
dy = sizey/resy;
dz = sizez/resz;
ix = 1:resx;
iy = 1:resy;
iz = 1:resz;
[iix,iiy,iiz] = ndgrid(ix,iy,iz);
px = (iix-1)*dx;
py = (iiy-1)*dy;
pz = (iiz-1)*dz;

kx = 1i * 2. * pi * (iix-1-resx/2)/(sizex);
ky = 1i * 2. * pi * (iiy-1-resy/2)/(sizey);
kz = 1i * 2. * pi * (iiz-1-resz/2)/(sizez);
k2 = kx.*kx + ky.*ky + kz.*kz;
k2(1+resx/2,1+resy/2,1+resz/2) = -1;
kd = zeros(resx,resy,resz);
kd(-k2<(resx.^2 + resy.^2 + resz.^2)/102.) = 1.;
kx = kx.*kd;
ky = ky.*kd;
kz = kz.*kd;

rho = zeros(resx,resy,resz) + 4;
zeta_all = zeros(resx,resy,resz);
Tv = zeros(resx,resy,resz,3);
Nv = zeros(resx,resy,resz,3);
Bv = zeros(resx,resy,resz,3);

resc = 20;
dt = 2*pi/resc;
parfor i = 1 : resx
    for j = 1 : resy
        for k = 1 : resz
            for t = 1 : resc
                zeta = dt*(t-1)-pi;
                czeta = curve(zeta);
                rho_temp = sqrt((px(i,j,k)-czeta(1))^2+(py(i,j,k)-czeta(2))^2+(pz(i,j,k)-czeta(3))^2);
                if(rho(i,j,k)>rho_temp)
                    rho(i,j,k) = rho_temp;
                    zeta_all(i,j,k) = zeta;
                end
            end
        end
    end
end
% '11111111111'
resc = 20;
dt = 2*dt/resc;
parfor i = 1 : resx
    for j = 1 : resy
        for k = 1 : resz
             if(rho(i,j,k)<2)
                zeta_temp = zeta_all(i,j,k);
                for t = 0 : resc
                    zeta = zeta_temp - resc*dt/2 + t * dt;
                    czeta = curve(zeta);
                    rho_temp = sqrt((px(i,j,k)-czeta(1))^2+(py(i,j,k)-czeta(2))^2+(pz(i,j,k)-czeta(3))^2);
                    if(rho(i,j,k)>rho_temp)
                        zeta_all(i,j,k) = zeta;
                        rho(i,j,k) = rho_temp;
                    end
                end
             end
        end
    end
end
% '22222222222'
resc = 20;
dt = 2*dt/resc;
parfor i = 1 : resx
    for j = 1 : resy
        for k = 1 : resz
             if(rho(i,j,k)<1)
                zeta_temp = zeta_all(i,j,k);
                for t = 0 : resc
                    zeta = zeta_temp - resc*dt/2 + t * dt;
                    czeta = curve(zeta);
                    rho_temp = sqrt((px(i,j,k)-czeta(1))^2+(py(i,j,k)-czeta(2))^2+(pz(i,j,k)-czeta(3))^2);
                    if(rho(i,j,k)>rho_temp)
                        zeta_all(i,j,k) = zeta;
                        rho(i,j,k) = rho_temp;
                    end
                end
             end
        end
    end
end
% '3333333333'
resc = 20;
dt = 2*dt/resc;
parfor i = 1 : resx
    for j = 1 : resy
        for k = 1 : resz
             if(rho(i,j,k)<0.8)
                zeta_temp = zeta_all(i,j,k);
                for t = 0 : resc
                    zeta = zeta_temp - resc*dt/2 + t * dt;
                    czeta = curve(zeta);
                    rho_temp = sqrt((px(i,j,k)-czeta(1))^2+(py(i,j,k)-czeta(2))^2+(pz(i,j,k)-czeta(3))^2);
                    if(rho(i,j,k)>rho_temp)
                        zeta_all(i,j,k) = zeta;
                        rho(i,j,k) = rho_temp;
                    end
                end
             end
        end
    end
end
% '44444444444'
resc = 20;
dt = 2*dt/resc;
parfor i = 1 : resx
    for j = 1 : resy
        for k = 1 : resz
            if(rho(i,j,k)<0.55)
                zeta_temp = zeta_all(i,j,k);
                for t = 0 : resc
                    zeta = zeta_temp - resc*dt/2 + t * dt;
                    czeta = curve(zeta);
                    rho_temp = sqrt((px(i,j,k)-czeta(1))^2+(py(i,j,k)-czeta(2))^2+(pz(i,j,k)-czeta(3))^2);
                    if(rho(i,j,k)>=rho_temp)
                        dczeta = d1curve(zeta);
                        ddczeta = d2curve(zeta);
                        Tvtemp = dczeta/norm(dczeta);
                        Bvtemp = cross(dczeta,ddczeta)/norm(cross(dczeta,ddczeta));
                        Nv(i,j,k,:) = cross(Bvtemp,Tvtemp)/norm(cross(Bvtemp,Tvtemp));
                        Tv(i,j,k,:) = Tvtemp;
                        Bv(i,j,k,:) = Bvtemp;
                        zeta_all(i,j,k) = zeta;
                        rho(i,j,k) = rho_temp;
                    end
                end
            end
        end
    end
end
% '555555555'

costheta = zeros(resx, resy, resz);
sintheta = zeros(resx, resy, resz);
parfor i = 1:resx
    for j = 1:resy
        for k = 1:resz
            czeta = curve(zeta_all(i,j,k));
            dczeta = [px(i,j,k),py(i,j,k),pz(i,j,k)]-czeta;
            dczeta = normalize(dczeta,"norm");
            Nvzeta = Nv(i,j,k,:);
            Bvzeta = Bv(i,j,k,:);
            costheta(i,j,k) = dczeta(1)*Nvzeta(1)+dczeta(2)*Nvzeta(2)+dczeta(3)*Nvzeta(3);
            sintheta(i,j,k) = dczeta(1)*Bvzeta(1)+dczeta(2)*Bvzeta(2)+dczeta(3)*Bvzeta(3);
        end
    end
end

cos2theta = costheta.*costheta - sintheta.*sintheta;
sin2theta = 2*sintheta.*costheta;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct wave funtion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gamma = 1;
hbar = Gamma/(4*pi);
Xi = zeros(resx, resy, resz);
sigma = 0.21;
F_divided_by_Gamma = 1-exp(-(rho/sigma).^4);

temp = 1 - F_divided_by_Gamma;
s1 = 2*F_divided_by_Gamma - 1;
s2 = 2*sqrt(F_divided_by_Gamma.*temp).* (sin2theta.*cos((Gamma/hbar)*Xi)-cos2theta.*sin((Gamma/hbar)*Xi));
s3 = 2*sqrt(F_divided_by_Gamma.*temp).* (cos2theta.*cos((Gamma/hbar)*Xi)+sin2theta.*sin((Gamma/hbar)*Xi));
max(max(max(abs(s1.^2 + s2.^2 + s3.^2-1))))
ds1dx = real(ifftn(fftshift(kx.*fftshift(fftn(s1)))));
ds1dy = real(ifftn(fftshift(ky.*fftshift(fftn(s1)))));
ds1dz = real(ifftn(fftshift(kz.*fftshift(fftn(s1)))));
ds2dx = real(ifftn(fftshift(kx.*fftshift(fftn(s2)))));
ds2dy = real(ifftn(fftshift(ky.*fftshift(fftn(s2)))));
ds2dz = real(ifftn(fftshift(kz.*fftshift(fftn(s2)))));
ds3dx = real(ifftn(fftshift(kx.*fftshift(fftn(s3)))));
ds3dy = real(ifftn(fftshift(ky.*fftshift(fftn(s3)))));
ds3dz = real(ifftn(fftshift(kz.*fftshift(fftn(s3)))));

max(max(max(abs(ds1dx.*s1 + ds2dx.*s2 + ds3dx.*s3))))

w1 = 0.5*hbar*(s1.*(ds2dy.*ds3dz-ds2dz.*ds3dy) + s2.*(ds3dy.*ds1dz-ds3dz.*ds1dy) + s3.*(ds1dy.*ds2dz-ds1dz.*ds2dy));
w2 = 0.5*hbar*(s1.*(ds2dz.*ds3dx-ds2dx.*ds3dz) + s2.*(ds3dz.*ds1dx-ds3dx.*ds1dz) + s3.*(ds1dz.*ds2dx-ds1dx.*ds2dz));
w3 = 0.5*hbar*(s1.*(ds2dx.*ds3dy-ds2dy.*ds3dx) + s2.*(ds3dx.*ds1dy-ds3dy.*ds1dx) + s3.*(ds1dx.*ds2dy-ds1dy.*ds2dx));
u1 = real(ifftn(fftshift((kz.*fftshift(fftn(w2))-ky.*fftshift(fftn(w3)))./k2)));
u2 = real(ifftn(fftshift((kx.*fftshift(fftn(w3))-kz.*fftshift(fftn(w1)))./k2)));
u3 = real(ifftn(fftshift((ky.*fftshift(fftn(w1))-kx.*fftshift(fftn(w2)))./k2)));


Eb = 0.5*sum(w1.*w1+w2.*w2+w3.*w3, 'all')*dx*dy*dz/sizex/sizey/sizez
Eu = 0.5*sum(u1.*u1+u2.*u2+u3.*u3, 'all')*dx*dy*dz/sizex/sizey/sizez
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % 
nbox = 10;
vbox1 = zeros(resx,resy,resz,nbox);
vbox1(:,:,:,1) = px;
vbox1(:,:,:,2) = py;
vbox1(:,:,:,3) = pz;
vbox1(:,:,:,4) = w1;
vbox1(:,:,:,5) = w2;
vbox1(:,:,:,6) = w3;
vbox1(:,:,:,7) = u1;
vbox1(:,:,:,8) = u2;
vbox1(:,:,:,9) = u3;
vbox1(:,:,:,10) = sqrt(w1.^2+w2.^2+w3.^2);
name1 = 'wave_function_s1s2s3.bin';
varname1 = ['x','y','z','a','b','c','u','v','w','e'];
writedate1 = output(vbox1,resx,resy,resz,nbox,name1,varname1);

function [czeta] = curve(zeta)
cx = pi + cos(zeta);
cy = pi + sin(zeta);
cz = pi;
czeta = [cx,cy,cz];
end


function [dczeta] = d1curve(zeta)
eps = 0.0001;
czeta_p = curve(zeta+eps);
czeta_m = curve(zeta-eps);
dczeta = 0.5*(czeta_p-czeta_m)/eps;
end
function [ddczeta] = d2curve(zeta)
eps = 0.0001;
czeta_p = curve(zeta+eps);
czeta_m = curve(zeta-eps);
czeta_0 = curve(zeta);
ddczeta = (czeta_p+czeta_m-2*czeta_0)/(eps*eps);
end
