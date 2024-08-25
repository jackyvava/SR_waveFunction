classdef Clebsch < handle

    properties (SetAccess=protected)
        hbar
        px, py, pz          % coordinates of grid points
        ix, iy, iz          % 1D index array
        iix,iiy,iiz         % 3D index array
        dx, dy, dz          % edge length
        sizex, sizey, sizez % size of grid
        resx, resy, resz    % number of grid points in each dimension
        Npsi
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
                otherwise
                    error('Clebsch:badinput',...
                        'Wrong number of inputs.')
            end
        end
    end
    methods % Class Methods


        function [vx,vy,vz] = TGVelocityOneForm(obj)
            %%% initial TG flow
            qx = obj.px + 0.5*obj.dx;
            qy = obj.py + 0.5*obj.dy;
            qz = obj.pz + 0.5*obj.dz;
            vx = obj.dx*sin(qx).* cos(qy) .* cos(qz);
            vy = -obj.dx*cos(qx).* sin(qy) .* cos(qz);
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
            psi_norm = sqrt(psi_norm);
            for i = 1:obj.Npsi
                psi(:,:,:,i) = psi(:,:,:,i)./psi_norm;
            end
        end

        function [Deviation] = CalDeviation(obj, vx, vy, vz, psi)
            % Initialize deviation
            Deviation = 0;
        
            % Get shifted versions of psi using circshift to handle periodic boundary conditions
            psi_xm = circshift(psi, [1, 0, 0]);  % Shift left along x-axis
            psi_xp = circshift(psi, [-1, 0, 0]); % Shift right along x-axis
            psi_ym = circshift(psi, [0, 1, 0]);  % Shift down along y-axis
            psi_yp = circshift(psi, [0, -1, 0]); % Shift up along y-axis
            psi_zm = circshift(psi, [0, 0, 1]);  % Shift forward along z-axis
            psi_zp = circshift(psi, [0, 0, -1]); % Shift backward along z-axis
        
            % Compute the phase factors for the velocity components
            phase_xm = exp(1i * vx / obj.hbar);
            phase_xp = exp(-1i * vx / obj.hbar);
            phase_ym = exp(1i * vy / obj.hbar);
            phase_yp = exp(-1i * vy / obj.hbar);
            phase_zm = exp(1i * vz / obj.hbar);
            phase_zp = exp(-1i * vz / obj.hbar);
        
            % Calculate deviations for each component of psi
            for ii = 1:obj.Npsi
                psi000 = psi(:,:,:,ii);
        
                % Compute deviations by comparing with the shifted and phase-modified psi
                deviation_xm = abs(psi000 - psi_xm(:,:,:,ii) .* phase_xm).^2;
                deviation_xp = abs(psi000 - psi_xp(:,:,:,ii) .* phase_xp).^2;
                deviation_ym = abs(psi000 - psi_ym(:,:,:,ii) .* phase_ym).^2;
                deviation_yp = abs(psi000 - psi_yp(:,:,:,ii) .* phase_yp).^2;
                deviation_zm = abs(psi000 - psi_zm(:,:,:,ii) .* phase_zm).^2;
                deviation_zp = abs(psi000 - psi_zp(:,:,:,ii) .* phase_zp).^2;
        
                % Sum all deviations
                Deviation = Deviation + sum(deviation_xm(:)) + sum(deviation_xp(:)) + ...
                                        sum(deviation_ym(:)) + sum(deviation_yp(:)) + ...
                                        sum(deviation_zm(:)) + sum(deviation_zp(:));
            end

            % Normalize the deviation
            Deviation = Deviation / (6 * obj.dx * obj.dy * obj.dz);
        end


        function [psi] = VelocityOneForm2Psi(obj, vx, vy, vz, psi)
            % Get shifted indices using circshift to handle periodic boundary conditions
            psi_xm = circshift(psi, [1, 0, 0]);  % Shift left along x-axis
            psi_xp = circshift(psi, [-1, 0, 0]); % Shift right along x-axis
            psi_ym = circshift(psi, [0, 1, 0]);  % Shift down along y-axis
            psi_yp = circshift(psi, [0, -1, 0]); % Shift up along y-axis
            psi_zm = circshift(psi, [0, 0, 1]);  % Shift forward along z-axis
            psi_zp = circshift(psi, [0, 0, -1]); % Shift backward along z-axis

            % Compute the phase factors for the velocity components
            phase_xm = exp(1i * vx / obj.hbar);
            phase_xp = exp(-1i * vx / obj.hbar);
            phase_ym = exp(1i * vy / obj.hbar);
            phase_yp = exp(-1i * vy / obj.hbar);
            phase_zm = exp(1i * vz / obj.hbar);
            phase_zp = exp(-1i * vz / obj.hbar);

            % Update psi using the shifted psi and phase factors
            psi = (psi_xm .* phase_xm + psi_xp .* phase_xp + ...
                   psi_ym .* phase_ym + psi_yp .* phase_yp + ...
                   psi_zm .* phase_zm + psi_zp .* phase_zp);

            % Normalize the updated psi
            psi = obj.Normalize(psi);
        end

    end
end