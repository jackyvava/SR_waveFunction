classdef Clebsch < handle

    properties (SetAccess=protected)
        hbar
        px, py, pz          % coordinates of grid points
        ix, iy, iz          % 1D index array
        iix,iiy,iiz         % 3D index array
        dx, dy, dz          % edge length
        sizex, sizey, sizez % size of grid
        resx, resy, resz    % number of grid points in each dimension
        Npsi,
        kx, ky, kz
    end
    methods % Constructor
        function obj = Clebsch(varargin)
            switch nargin
                case 8
                    obj.sizex = varargin{1};
                    obj.sizey = varargin{2};
                    obj.sizez = varargin{3};
                    obj.resx = round(varargin{4});
                    obj.resy = round(varargin{5});
                    obj.resz = round(varargin{6});
                    obj.hbar = varargin{7};
                    obj.Npsi = varargin{8};
                    obj.dx = obj.sizex/obj.resx;
                    obj.dy = obj.sizey/obj.resy;
                    obj.dz = obj.sizez/obj.resz;
                    obj.ix = 1:obj.resx;
                    obj.iy = 1:obj.resy;
                    obj.iz = 1:obj.resz;
                    [obj.iix,obj.iiy,obj.iiz] = ndgrid(obj.ix,obj.iy,obj.iz);
                    obj.px = (obj.iix-1)*obj.dx;
                    obj.py = (obj.iiy-1)*obj.dy;
                    obj.pz = (obj.iiz-1)*obj.dz;
                    obj.kx = 1i * 2. * pi * (obj.iix-1-obj.resx/2)/(obj.sizex);
                    obj.ky = 1i * 2. * pi * (obj.iiy-1-obj.resy/2)/(obj.sizey);
                    obj.kz = 1i * 2. * pi * (obj.iiz-1-obj.resz/2)/(obj.sizez);

                otherwise
                    error('Clebsch:badinput',...
                        'Wrong number of inputs.')
            end
        end
    end
    methods % Class Methods


        function [vx,vy,vz] = TGVelocityOneForm(obj)
            %%% initial TG flow
            vx = obj.dx*sin(obj.px + 0.5*obj.dx).* cos(obj.py) .* cos(obj.pz);
            vy = -obj.dx*cos(obj.px).* sin(obj.py + 0.5*obj.dy) .* cos(obj.pz);
            vz = 0*vx;
        end

        function [wx,wy,wz] = DerivativeOfOneForm(obj,vx,vy,vz)
            % DerivativeOfOneForm
            % For a 1-form v compute the 2-form dv
            ixp = mod(obj.ix,obj.resx) + 1;
            iyp = mod(obj.iy,obj.resy) + 1;
            izp = mod(obj.iz,obj.resz) + 1;
            wx = vy - vy(:,:,izp) + vz(:,iyp,:) - vz;
            wy = vz - vz(ixp,:,:) + vx(:,:,izp) - vx;
            wz = vx - vx(:,iyp,:) + vy(ixp,:,:) - vy;
        end

        function psi = Normalize(obj,psi)
            psi_norm = zeros(obj.resx,obj.resy,obj.resz);
            for i = 1:obj.Npsi
                psi_norm = psi_norm+psi(:,:,:,i).*conj(psi(:,:,:,i));
            end
            for i = 1:obj.Npsi
                psi(:,:,:,i) = psi(:,:,:,i)./sqrt(psi_norm);
            end
        end

        function [Deviation] = CalDeviation_Original(obj,vx,vy,vz,psi)
            Deviation = 0.;
            for ri = 1:obj.resx
                for rj = 1:obj.resy
                    for rk = 1:obj.resz
                        rip = mod(ri,obj.resx) + 1;
                        rjp = mod(rj,obj.resy) + 1;
                        rkp = mod(rk,obj.resz) + 1;
                        rim = mod(ri-2,obj.resx) + 1;
                        rjm = mod(rj-2,obj.resy) + 1;
                        rkm = mod(rk-2,obj.resz) + 1;
                        vxm = vx(rim,rj,rk);
                        vxp = vx(ri,rj,rk);
                        vym = vy(ri,rjm,rk);
                        vyp = vy(ri,rj,rk);
                        vzm = vz(ri,rj,rkm);
                        vzp = vz(ri,rj,rk);
                        for ii = 1:obj.Npsi
                            psi000 = psi(ri,rj,rk,ii);
                            psim00 = psi(rim,rj,rk,ii)*exp(1i*vxm/obj.hbar);
                            psip00 = psi(rip,rj,rk,ii)*exp(-1i*vxp/obj.hbar);
                            psi0m0 = psi(ri,rjm,rk,ii)*exp(1i*vym/obj.hbar);
                            psi0p0 = psi(ri,rjp,rk,ii)*exp(-1i*vyp/obj.hbar);
                            psi00m = psi(ri,rj,rkm,ii)*exp(1i*vzm/obj.hbar);
                            psi00p = psi(ri,rj,rkp,ii)*exp(-1i*vzp/obj.hbar);
                            Deviation = Deviation + abs(psi000-psim00)^2;
                            Deviation = Deviation + abs(psi000-psip00)^2;
                            Deviation = Deviation + abs(psi000-psi0m0)^2;
                            Deviation = Deviation + abs(psi000-psi0p0)^2;
                            Deviation = Deviation + abs(psi000-psi00m)^2;
                            Deviation = Deviation + abs(psi000-psi00p)^2;
                        end
                    end
                end
            end
            Deviation = Deviation/6*obj.dx*obj.dy*obj.dz;
        end

        function [Deviation] = CalDeviation(obj,vx,vy,vz,psi)
            Deviation = 0.;
            ri = 1:obj.resx;
            rj = 1:obj.resy;
            rk = 1:obj.resz;
            rip = mod(ri,obj.resx) + 1;
            rjp = mod(rj,obj.resy) + 1;
            rkp = mod(rk,obj.resz) + 1;
            rim = mod(ri-2,obj.resx) + 1;
            rjm = mod(rj-2,obj.resy) + 1;
            rkm = mod(rk-2,obj.resz) + 1;

            vxm = vx(rim,rj,rk);
            vxp = vx(ri,rj,rk);
            vym = vy(ri,rjm,rk);
            vyp = vy(ri,rj,rk);
            vzm = vz(ri,rj,rkm);
            vzp = vz(ri,rj,rk);

            for ii = 1:obj.Npsi
                psi000 = psi(ri,rj,rk,ii);
                psim00 = psi(rim,rj,rk,ii).*exp(1i*vxm/obj.hbar);
                psip00 = psi(rip,rj,rk,ii).*exp(-1i*vxp/obj.hbar);
                psi0m0 = psi(ri,rjm,rk,ii).*exp(1i*vym/obj.hbar);
                psi0p0 = psi(ri,rjp,rk,ii).*exp(-1i*vyp/obj.hbar);
                psi00m = psi(ri,rj,rkm,ii).*exp(1i*vzm/obj.hbar);
                psi00p = psi(ri,rj,rkp,ii).*exp(-1i*vzp/obj.hbar);

                % Calculate deviation
                Deviation = Deviation + sum(abs(psi000 - psim00).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psip00).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psi0m0).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psi0p0).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psi00m).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psi00p).^2, 'all');
            end

            Deviation = Deviation/6*obj.dx*obj.dy*obj.dz;
        end

        function [psi] = VelocityOneForm2Psi_Original(obj,vx,vy,vz,psi)
            for i2 = 0:1
                for j2 = 0:1
                    for k2 = 0:1
                        for i = 1:obj.resx/2
                            for j = 1:obj.resy/2
                                for k = 1:obj.resz/2
                                    for ii = 1:obj.Npsi
                                        ri = 2*i + i2 -1;
                                        rj = 2*j + j2 -1;
                                        rk = 2*k + k2 -1;
                                        rip = mod(ri,obj.resx) + 1;
                                        rjp = mod(rj,obj.resy) + 1;
                                        rkp = mod(rk,obj.resz) + 1;
                                        rim = mod(ri-2,obj.resx) + 1;
                                        rjm = mod(rj-2,obj.resy) + 1;
                                        rkm = mod(rk-2,obj.resz) + 1;
                                        vxm = vx(rim,rj,rk);
                                        vxp = vx(ri,rj,rk);
                                        vym = vy(ri,rjm,rk);
                                        vyp = vy(ri,rj,rk);
                                        vzm = vz(ri,rj,rkm);
                                        vzp = vz(ri,rj,rk);
                                        psim00 = psi(rim,rj,rk,ii)*exp(1i*vxm/obj.hbar);
                                        psip00 = psi(rip,rj,rk,ii)*exp(-1i*vxp/obj.hbar);
                                        psi0m0 = psi(ri,rjm,rk,ii)*exp(1i*vym/obj.hbar);
                                        psi0p0 = psi(ri,rjp,rk,ii)*exp(-1i*vyp/obj.hbar);
                                        psi00m = psi(ri,rj,rkm,ii)*exp(1i*vzm/obj.hbar);
                                        psi00p = psi(ri,rj,rkp,ii)*exp(-1i*vzp/obj.hbar);
                                        psi(ri,rj,rk,ii) = (psim00+psip00+psi0m0+psi0p0+psi00m+psi00p);
                                    end
                                end
                            end
                        end
                    end
                end
            end
            psi = obj.Normalize(psi);
        end

        function [psi] = VelocityOneForm2Psi(obj, vx, vy, vz, psi)
            for i2 = 0:1
                for j2 = 0:1
                    for k2 = 0:1
                        for i = 1:obj.resx/2
                            % 向量化处理j和k维度
                            ri = 2*i + i2 - 1;
                            rj = 2*(1:obj.resy/2) + j2 - 1;
                            rk = 2*(1:obj.resz/2) + k2 - 1;

                            % 处理周期性边界条件
                            rip = mod(ri, obj.resx) + 1;
                            rjp = mod(rj, obj.resy) + 1;
                            rkp = mod(rk, obj.resz) + 1;
                            rim = mod(ri-2, obj.resx) + 1;
                            rjm = mod(rj-2, obj.resy) + 1;
                            rkm = mod(rk-2, obj.resz) + 1;

                            % 提取速度场的分量 (向量化处理)
                            vxm = vx(rim, rj, rk);
                            vxp = vx(ri, rj, rk);
                            vym = vy(ri, rjm, rk);
                            vyp = vy(ri, rj, rk);
                            vzm = vz(ri, rj, rkm);
                            vzp = vz(ri, rj, rk);

                            % 使用向量化处理 psi
                            psim00 = psi(rim, rj, rk, :) .* exp(1i * vxm / obj.hbar);
                            psip00 = psi(rip, rj, rk, :) .* exp(-1i * vxp / obj.hbar);
                            psi0m0 = psi(ri, rjm, rk, :) .* exp(1i * vym / obj.hbar);
                            psi0p0 = psi(ri, rjp, rk, :) .* exp(-1i * vyp / obj.hbar);
                            psi00m = psi(ri, rj, rkm, :) .* exp(1i * vzm / obj.hbar);
                            psi00p = psi(ri, rj, rkp, :) .* exp(-1i * vzp / obj.hbar);

                            % 将结果累加
                            psi(ri, rj, rk, :) = (psim00 + psip00 + psi0m0 + psi0p0 + psi00m + psi00p);
                        end
                    end
                end
            end

            % 最后进行归一化处理
            psi = obj.Normalize(psi);
        end

        function [ux, uy, uz] = CalVelFromPsi(obj, psi)
            % 调试
            %disp(size(obj.kx));
            %disp(size(psi));

            % 初始化速度场为零
            ux = zeros(obj.resx, obj.resy, obj.resz);
            uy = zeros(obj.resx, obj.resy, obj.resz);
            uz = zeros(obj.resx, obj.resy, obj.resz);

            % 循环处理每个波函数分量
            for i = 1:obj.Npsi
                % 对每个 psi 分量进行傅里叶变换
                phi = fftshift(fftn(psi(:,:,:,i)));

                % 计算 psi 在 x, y, z 方向上的导数
                psix = ifftn(fftshift(phi .* obj.kx));
                psiy = ifftn(fftshift(phi .* obj.ky));
                psiz = ifftn(fftshift(phi .* obj.kz));

                % 根据 psi 的实部和虚部计算速度场分量
                ux = ux + real(psi(:,:,:,i)) .* imag(psix) - imag(psi(:,:,:,i)) .* real(psix);
                uy = uy + real(psi(:,:,:,i)) .* imag(psiy) - imag(psi(:,:,:,i)) .* real(psiy);
                uz = uz + real(psi(:,:,:,i)) .* imag(psiz) - imag(psi(:,:,:,i)) .* real(psiz);
            end

            % 乘以 hbar 以得到最终的速度场
            ux = ux * obj.hbar;
            uy = uy * obj.hbar;
            uz = uz * obj.hbar;
        end

        function [vx,vy,vz] = VelocityOneForm(obj,psi)
            ixp = mod(obj.ix,obj.resx) + 1;
            iyp = mod(obj.iy,obj.resy) + 1;
            izp = mod(obj.iz,obj.resz) + 1;
            thetax = zeros(obj.resx,obj.resy,obj.resz);
            thetay = zeros(obj.resx,obj.resy,obj.resz);
            thetaz = zeros(obj.resx,obj.resy,obj.resz);
            for ii = 1:obj.Npsi
                thetax = thetax + conj(psi(:,:,:,ii)).*psi(ixp,:,:,ii);
                thetay = thetay + conj(psi(:,:,:,ii)).*psi(:,iyp,:,ii);
                thetaz = thetaz + conj(psi(:,:,:,ii)).*psi(:,:,izp,ii);
            end
            vx = obj.hbar*angle(thetax);
            vy = obj.hbar*angle(thetay);
            vz = obj.hbar*angle(thetaz);
        end

    end
end