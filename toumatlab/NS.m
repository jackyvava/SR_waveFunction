classdef NS < TorusDEC


    properties
        nu
        dt
        hbar
        viscosityMask
    end
    methods

        function obj = NS(varargin)
            obj = obj@TorusDEC(varargin{:});
        end

        function Buildviscosity(obj)
            % builds coefficients in Fourier space.
            obj.viscosityMask = exp(obj.nu*obj.k2*obj.dt);
        end


        function [psi1, psi2] = GetPsi(obj)
            cx = [pi/3, pi/1.5];
            cy = [pi, pi];
            Ra = [pi/4, pi/4];
            psi1 = 1;
            psi2 = 0.04;
            for i = 1:2
                rx = (obj.px-cx(i)) / Ra(i);
                ry = (obj.py-cy(i)) / Ra(i);
                r2 = rx.^2 + ry.^2;
                decay = exp(-(r2/4).^4);
                psi1 = psi1.*(2 * rx.*decay + 1i * (r2+1 - 2*decay)) ./ (1 + r2);
            end

%             cx = [pi/2, pi/2, pi/2, pi/2];
%             cy = [pi+pi/3, pi-pi/3, pi+pi/6, pi-pi/6];
%             Ra = 1;
%             psi1 = 1;
%             psi2 = 1;
%             for i = 1:1
%                 rx = (obj.px-cx(i));
%                 ry = (obj.py-cy(i));
%                 r2 = (rx.^2 + ry.^2).^4;
%                 psi1 = psi1.* r2 ./ (Ra^8 + r2);
%             end
            for i = 1:10
                psi_norm = sqrt(conj(psi1).*psi1 + conj(psi2).*psi2);
                psi1 = psi1 ./ psi_norm;
                psi2 = psi2 ./ psi_norm;
                [psi1,psi2] = obj.PressureProject(psi1,psi2);
                [ux, uy] =obj.CalVelFromPsi(psi1,psi2);
                obj.Cal_divergence(ux,uy)
                psi1 = ifftn(fftshift(fftshift(fftn(psi1)).*obj.kd));
                psi2 = ifftn(fftshift(fftshift(fftn(psi2)).*obj.kd)); 
            end
        end
        function div = DivPsi(obj,psi1,psi2)
            v1 = angle(conj(psi1).*psi1(obj.ixp,:,:)+conj(psi2).*psi2(obj.ixp,:,:));
            v2 = angle(conj(psi1).*psi1(:,obj.iyp,:)+conj(psi2).*psi2(:,obj.iyp,:));
            div = ((v1 - v1(obj.ixm,:,:))+(v2 - v2(:,obj.iym,:)))/obj.dx/obj.dy;
        end
        function [psi1,psi2] = PressureProject(obj,psi1,psi2)
        % Pressure projection of 2-component wave function.
        %
            phi = ifftn(fftshift(fftshift(fftn(obj.DivPsi(psi1,psi2)))./obj.k2));
            phi = exp(-1i*phi);
            psi1 = psi1.*phi;
            psi2 = psi2.*phi;
        end

        function [u1, u2] = CalVelFromPsi(obj,psi1,psi2)
            psi = fftshift(fftn(psi1)).*obj.kd;
            psix = ifftn(fftshift(psi.*obj.kx));
            psiy = ifftn(fftshift(psi.*obj.ky));
            u1 = real(1i*conj(psix).*psi1);
            u2 = real(1i*conj(psiy).*psi1);
            psi = fftshift(fftn(psi2)).*obj.kd;
            psix = ifftn(fftshift(psi.*obj.kx));
            psiy = ifftn(fftshift(psi.*obj.ky));
            u1 = obj.hbar*(u1+real(1i*conj(psix).*psi2));
            u2 = obj.hbar*(u2+real(1i*conj(psiy).*psi2));
        end

        function [ux,uy]=NSFlow(obj,ux,uy)
            [k1x,k1y] = obj.EulerFlow_du(ux,uy);
            [k2x,k2y] = obj.EulerFlow_du(ux+k1x*obj.dt/2,uy+k1y*obj.dt/2);
            [k3x,k3y] = obj.EulerFlow_du(ux+k2x*obj.dt/2,uy+k2y*obj.dt/2);
            [k4x,k4y] = obj.EulerFlow_du(ux+k3x*obj.dt,uy+k3y*obj.dt);
            ux = ux + obj.dt/6*(k1x+2*k2x+2*k3x+k4x);
            uy = uy + obj.dt/6*(k1y+2*k2y+2*k3y+k4y);
            [ux,uy] = obj.impTOvel(ux,uy);
            [ux,uy] = obj.viciousFlow(ux,uy);

        end

        function [dux,duy]=EulerFlow_du(obj,ux,uy)
            wz = real(ifftn(fftshift(fftshift(fftn(uy)).*obj.kx-fftshift(fftn(ux)).*obj.ky)));
            dux = uy.*wz;
            duy = -ux.*wz;
        end

        function [u1,u2] = viciousFlow(obj,u1,u2)
            f = exp(obj.nu*obj.k2*obj.dt);
            u1 = real(ifftn(fftshift(fftshift(fftn(u1)).*f)));
            u2 = real(ifftn(fftshift(fftshift(fftn(u2)).*f)));
        end
    end
end