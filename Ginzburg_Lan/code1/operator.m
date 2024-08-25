classdef operator < handle
    properties (SetAccess=protected)
        hbar
        Npsi
        sizex, sizey % size of grid
        resx, resy    % number of grid points in each dimension
        ix,iy,ixp,iyp,ixm,iym
        iix, iiy
        dx, dy
        px, py
        kx, ky, k2, k20, kd
    end
    methods % Constructor
        function obj = operator(varargin)
            switch nargin
                case 6
                    obj.sizex = varargin{1};
                    obj.sizey = varargin{2};
                    obj.resx = round(varargin{3});
                    obj.resy = round(varargin{4});
                    obj.hbar = varargin{5};
                    obj.Npsi = varargin{6};
                    obj.dx = obj.sizex/obj.resx;
                    obj.dy = obj.sizey/obj.resy;
                    obj.ix = 1:obj.resx;
                    obj.iy = 1:obj.resy;
                    obj.ixp = mod(obj.ix,obj.resx) + 1;
                    obj.iyp = mod(obj.iy,obj.resy) + 1;
                    obj.ixm = mod(obj.ix-2,obj.resx) + 1;
                    obj.iym = mod(obj.iy-2,obj.resy) + 1;
                    [obj.iix,obj.iiy] = ndgrid(obj.ix,obj.iy);
                    obj.px = (obj.iix-1)*obj.dx;
                    obj.py = (obj.iiy-1)*obj.dy;
                    obj.kx = 1i * 2. * pi * (obj.iix-1-obj.resx/2)/(obj.sizex);
                    obj.ky = 1i * 2. * pi * (obj.iiy-1-obj.resy/2)/(obj.sizey);
                    obj.k2 = obj.kx.^2 + obj.ky.^2;
                    obj.k20 = obj.kx.^2 + obj.ky.^2;
                    obj.k2(1+obj.resx/2,1+obj.resy/2) = -1;
                    obj.kd = zeros(obj.resx,obj.resy);
                    obj.kd(-obj.k2<(obj.resx.^2 + obj.resy.^2)/30) = 1.;
                    obj.kx=obj.kx.*obj.kd;
                    obj.ky=obj.ky.*obj.kd;
                otherwise
                    error('NS:badinput',...
                        'Wrong number of inputs.')
            end
        end
    end

    methods % Class Methods
        function psi = Normalize(obj,psi)
            psi_norm = zeros(obj.resx,obj.resy);
            for i = 1:obj.Npsi
                psi_norm = abs(psi(:,:,i)).^2;
            end
            psi_norm = sqrt(psi_norm);
            for i = 1:obj.Npsi
                psi(:,:,i) = psi(:,:,i)./psi_norm;
            end
        end

        function [u1, u2] = CalVelFromVor(obj,w3)
            u1 = real(ifftn(fftshift((-obj.ky.*fftshift(fftn(w3)))./obj.k2)));
            u2 = real(ifftn(fftshift((obj.kx.*fftshift(fftn(w3)))./obj.k2)));
        end

        function [w3] = CalVorFromVel(obj,u1,u2)
            fu1 = fftshift(fftn(u1));
            fu2 = fftshift(fftn(u2));
            w3 = real(ifftn(fftshift(obj.kx.*fu2 - obj.ky.*fu1)));
        end

        function [psix,psiy] = gradpsi(obj,psi)
            psix = zeros(obj.resx,obj.resy,obj.Npsi);
            psiy = zeros(obj.resx,obj.resy,obj.Npsi);
            for i = 1:obj.Npsi
                phi = fftshift(fftn(psi(:,:,i)));
                psix(:,:,i)=ifftn(fftshift(phi.*obj.kx));
                psiy(:,:,i)=ifftn(fftshift(phi.*obj.ky));
            end
        end


        function [psi] = GetRandPsi(obj)
            psi = randn(obj.resx,obj.resy,obj.Npsi);
            [psi] = obj.Normalize(psi);
            for i = 1:20
                [psi] = obj.PressureProject(psi);
                div = DivPsi(obj,psi);
                sum(sum(sum(abs(div))))
            end
        end

        function div = DivPsi(obj,psi)
            v1 = zeros(obj.resx,obj.resy);
            v2 = zeros(obj.resx,obj.resy);
            for i = 1:obj.Npsi
                v1 = v1 + conj(psi(:,:,i)).*psi(obj.ixp,:,i);
                v2 = v2 + conj(psi(:,:,i)).*psi(:,obj.iyp,i);
            end
            v1 = angle(v1);
            v2 = angle(v2);
            div = (v1 - v1(obj.ixm,:))+(v2 - v2(:,obj.iym));
        end

        function f = PoissonSolve(obj,f)
            % PoissonSolve by Spectral method
            fac = -0.25./(sin((obj.iix-1-obj.resx/2)*pi/obj.resx).^2 + sin((obj.iiy-1-obj.resy/2)*pi/obj.resy).^2);
            fac(1+obj.resx/2,1+obj.resy/2) = 1;
            f = ifftn(fftshift(fftshift(fftn(f)).* fac));
        end

        function [psi] = PressureProject(obj,psi)
            eiq = exp(-1i*obj.PoissonSolve(obj.DivPsi(psi)));
            for i = 1:obj.Npsi
                psi(:,:,i) = psi(:,:,i).*eiq;
            end
        end

        function [ux, uy] = CalVelFromPsi(obj,psi)
            ux = zeros(obj.resx,obj.resy);
            uy = zeros(obj.resx,obj.resy);
            for i = 1:obj.Npsi
                phi = fftshift(fftn(psi(:,:,i)));
                psix = ifftn(fftshift(phi.*obj.kx));
                psiy = ifftn(fftshift(phi.*obj.ky));
                ux = ux + real(psi(:,:,i)).*imag(psix)-imag(psi(:,:,i)).*real(psix);
                uy = uy + real(psi(:,:,i)).*imag(psiy)-imag(psi(:,:,i)).*real(psiy);
            end
        end

        function [Deviation] = CalDeviation(obj,ux,uy,psi)
            [uxC, uyC] = obj.CalVelFromPsi(psi);
            Deviation = sum(sum(sqrt((ux-uxC).^2+(uy-uyC).^2)))/obj.resx/obj.resy;
        end

    end

end