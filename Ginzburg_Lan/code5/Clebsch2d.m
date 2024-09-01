classdef Clebsch2d < handle

    properties (SetAccess=protected)
        hbar
        px, py          % coordinates of grid points
        ix, iy          % 1D index array
        iix,iiy         % 2D index array
        dx, dy          % edge length
        sizex, sizey    % size of grid
        resx, resy      % number of grid points in each dimension
        Npsi,
        kx, ky
    end

    methods % Constructor
        function obj = Clebsch2d(varargin)
            switch nargin
                case 6
                    obj.sizex = varargin{1};
                    obj.sizey = varargin{2};
                    obj.resx = round(varargin{3});
                    obj.resy = round(varargin{4});
                    obj.hbar = varargin{5};
                    obj.Npsi = varargin{6};
                    obj.dx = obj.sizex / obj.resx;
                    obj.dy = obj.sizey / obj.resy;
                    obj.ix = 1:obj.resx;
                    obj.iy = 1:obj.resy;
                    [obj.iix, obj.iiy] = ndgrid(obj.ix, obj.iy);
                    obj.px = (obj.iix - 1) * obj.dx;
                    obj.py = (obj.iiy - 1) * obj.dy;
                    obj.kx = 1i * 2. * pi * (obj.iix - 1 - obj.resx / 2) / obj.sizex;
                    obj.ky = 1i * 2. * pi * (obj.iiy - 1 - obj.resy / 2) / obj.sizey;
                otherwise
                    error('Clebsch:badinput', 'Wrong number of inputs.')
            end
        end
    end

    methods % Class Methods
%%      TG涡
        function [vx, vy] = TGVelocityOneForm(obj)
            %%% initial TG flow in 2D
            vx = obj.dx * sin(obj.px + 0.5 * obj.dx) .* cos(obj.py);
            vy = -obj.dx * cos(obj.px) .* sin(obj.py + 0.5 * obj.dy);
        end
%%      TG涡增加noise
        function [vx, vy] = TGVelocityOneForm_noise(obj)
            %%% initial TG flow in 2D with random noise
            noise_level = 0.1; % 设置噪声水平，可以根据需要调整
            vx = obj.dx * sin(obj.px + 0.5 * obj.dx) .* cos(obj.py) + noise_level * randn(size(obj.px));
            vy = -obj.dx * cos(obj.px) .* sin(obj.py + 0.5 * obj.dy) + noise_level * randn(size(obj.py));
        end
%%      简单的旋转流场
        function [vx, vy] = DoubleVortexFlow(obj)
            % Double Vortex Flow in 2D
            vx = obj.dx * cos(obj.px + 0.5 * obj.dx) .* cos(obj.py);
            vy = -obj.dx * cos(obj.px) .* sin(obj.py + 0.5 * obj.dy);
        
        end
%% 计算梯度
        function [wx, wy] = DerivativeOfOneForm(obj, vx, vy)
            % DerivativeOfOneForm
            % For a 1-form v compute the 2-form dv in 2D
            ixp = mod(obj.ix, obj.resx) + 1;
            iyp = mod(obj.iy, obj.resy) + 1;
            wx = vy(:, ixp) - vy;
            wy = vx(iyp, :) - vx;
        end

        function psi = Normalize(obj, psi)
            psi_norm = zeros(obj.resx, obj.resy);
            for i = 1:obj.Npsi
                psi_norm = psi_norm + psi(:, :, i) .* conj(psi(:, :, i));
            end
            for i = 1:obj.Npsi
                psi(:, :, i) = psi(:, :, i) ./ sqrt(psi_norm);
            end
        end

        function [Deviation] = CalDeviation(obj, vx, vy, psi)
            Deviation = 0.;
            ri = 1:obj.resx;
            rj = 1:obj.resy;
            rip = mod(ri, obj.resx) + 1;
            rjp = mod(rj, obj.resy) + 1;
            rim = mod(ri - 2, obj.resx) + 1;
            rjm = mod(rj - 2, obj.resy) + 1;

            vxm = vx(rim, rj);
            vxp = vx(ri, rj);
            vym = vy(ri, rjm);
            vyp = vy(ri, rj);

            for ii = 1:obj.Npsi
                psi000 = psi(ri, rj, ii);
                psim00 = psi(rim, rj, ii) .* exp(1i * vxm / obj.hbar);
                psip00 = psi(rip, rj, ii) .* exp(-1i * vxp / obj.hbar);
                psi0m0 = psi(ri, rjm, ii) .* exp(1i * vym / obj.hbar);
                psi0p0 = psi(ri, rjp, ii) .* exp(-1i * vyp / obj.hbar);

                % Calculate deviation
                Deviation = Deviation + sum(abs(psi000 - psim00).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psip00).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psi0m0).^2, 'all');
                Deviation = Deviation + sum(abs(psi000 - psi0p0).^2, 'all');
            end

            Deviation = Deviation / 4 * obj.dx * obj.dy;
        end

        function [psi] = VelocityOneForm2Psi(obj, vx, vy, psi)
            for i2 = 0:1
                for j2 = 0:1
                    for i = 1:obj.resx/2
                        % 向量化处理j维度
                        ri = 2*i + i2 - 1;
                        rj = 2*(1:obj.resy/2) + j2 - 1;

                        % 处理周期性边界条件
                        rip = mod(ri, obj.resx) + 1;
                        rjp = mod(rj, obj.resy) + 1;
                        rim = mod(ri - 2, obj.resx) + 1;
                        rjm = mod(rj - 2, obj.resy) + 1;

                        % 提取速度场的分量 (向量化处理)
                        vxm = vx(rim, rj);
                        vxp = vx(ri, rj);
                        vym = vy(ri, rjm);
                        vyp = vy(ri, rj);

                        % 使用向量化处理 psi
                        psim00 = psi(rim, rj, :) .* exp(1i * vxm / obj.hbar);
                        psip00 = psi(rip, rj, :) .* exp(-1i * vxp / obj.hbar);
                        psi0m0 = psi(ri, rjm, :) .* exp(1i * vym / obj.hbar);
                        psi0p0 = psi(ri, rjp, :) .* exp(-1i * vyp / obj.hbar);

                        % 将结果累加
                        psi(ri, rj, :) = (psim00 + psip00 + psi0m0 + psi0p0);
                    end
                end
            end

            % 最后进行归一化处理
            psi = obj.Normalize(psi);
        end

        function [ux, uy] = CalVelFromPsi(obj, psi)
            % 初始化速度场为零
            ux = zeros(obj.resx, obj.resy);
            uy = zeros(obj.resx, obj.resy);

            % 循环处理每个波函数分量
            for i = 1:obj.Npsi
                % 对每个 psi 分量进行傅里叶变换
                phi = fftshift(fftn(psi(:, :, i)));

                % 计算 psi 在 x, y 方向上的导数
                psix = ifftn(fftshift(phi .* obj.kx));
                psiy = ifftn(fftshift(phi .* obj.ky));

                % 根据 psi 的实部和虚部计算速度场分量
                ux = ux + real(psi(:, :, i)) .* imag(psix) - imag(psi(:, :, i)) .* real(psix);
                uy = uy + real(psi(:, :, i)) .* imag(psiy) - imag(psi(:, :, i)) .* real(psiy);
            end

            % 乘以 hbar 以得到最终的速度场
            ux = ux * obj.hbar;
            uy = uy * obj.hbar;
        end

        function [vx, vy] = VelocityOneForm(obj, psi)
            ixp = mod(obj.ix, obj.resx) + 1;
            iyp = mod(obj.iy, obj.resy) + 1;
            thetax = zeros(obj.resx, obj.resy);
            thetay = zeros(obj.resx, obj.resy);
            for ii = 1:obj.Npsi
                thetax = thetax + conj(psi(:, :, ii)) .* psi(ixp, :, ii);
                thetay = thetay + conj(psi(:, :, ii)) .* psi(:, iyp, ii);
            end
            vx = obj.hbar * angle(thetax);
            vy = obj.hbar * angle(thetay);
        end

    end
end
