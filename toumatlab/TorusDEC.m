classdef TorusDEC < handle
    properties (SetAccess=protected)
        px, py          % coordinates of grid points
        ix, iy          % 1D index array
        ixp, iyp          % 1D index array
        ixm, iym          % 1D index array
        iix,iiy         % 3D index array
        dx, dy          % edge length
        sizex, sizey % size of grid
        resx, resy    % number of grid points in each dimension
        kx, ky, k2, kd
    end
    methods % Constructor
        function obj = TorusDEC(varargin)
            switch nargin
                case 4
                    obj.sizex = varargin{1};
                    obj.sizey = varargin{2};
                    obj.resx = round(varargin{3});
                    obj.resy = round(varargin{4});
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
                    obj.k2(1+obj.resx/2,1+obj.resy/2) = -1;
                    obj.kd = zeros(obj.resx,obj.resy);
                    obj.kd(-obj.k2<(obj.resx.^2 + obj.resy.^2)/18.) = 1.;
                    obj.kx = obj.kx .* obj.kd;
                    obj.ky = obj.ky .* obj.kd;
                otherwise
                    error('TorusDEC:badinput',...
                        'Wrong number of inputs.')
            end
        end
    end
    methods % Class Methods
        
        function [vx,vy] = DerivativeOfFunction(obj,f)
        % DerivativeOfFunction
        % For a function f compute the 1-form df
            
            vx = f(obj.ixp,:) - f;
            vy = f(:,obj.iyp) - f;
        end
        
        function [wz] = DerivativeOfOneForm(obj,vx,vy)
        % DerivativeOfOneForm
        % For a 1-form v compute the 2-form dv
            wz = vx - vx(:,obj.iyp) + vy(obj.ixp,:) - vy;
        end
               
        
        function f = DerivativeOfTwoForm(obj,wx,wy)
        % DerivativeOfTwoForm
        % For a 2-form w compute the 3-form dw
            f =     wx(obj.ixp,:,:) - wx;
            f = f + wy(:,obj.iyp,:) - wy;
        end
        
        
        function f = Div(obj,vx,vy)
        % Div
        % For a 1-form v compute the function *d*v
            f =     (vx - vx(obj.ixm,:))/(obj.dx^2);
            f = f + (vy - vy(:,obj.iym))/(obj.dy^2);
        end
        
        function [ux,uy] = Sharp(obj,vx,vy)
        % Sharp
        % For a 1-form v compute the corresponding vector field v^sharp by
        % averaging to vertices
            ux = 0.5*( vx(obj.ixm,:) + vx )/obj.dx;
            uy = 0.5*( vy(:,obj.iym) + vy )/obj.dy;
        end
        
        function [w3] = Star2form(obj,wz)
        % Sharp
        % For a 1-form v compute the corresponding vector field v^sharp by
        % averaging to vertices
            w3 = 0.25*( wz(obj.ixm,obj.iym) + wz(:,obj.iym) + wz(obj.ixm,:) + wz(:,:) )./obj.dx./obj.dy;
        end
        
        
        function [ux,uy] = StaggeredSharp(obj,vx,vy)
        % StaggeredSharp
        % For a 1-form v compute the corresponding vector field v^sharp as
        % a staggered vector field living on edges
            ux = vx/obj.dx;
            uy = vy/obj.dy;
        end
        
        function [ux,uy] = impTOvel(obj,mx,my)
            phi = (fftshift(fftn(mx)).*obj.kx+fftshift(fftn(my)).*obj.ky)./obj.k2;
            ux = mx - real(ifftn(fftshift(phi.*obj.kx)));
            uy = my - real(ifftn(fftshift(phi.*obj.ky)));
        end
        
        function [meandiv] = Cal_divergence(obj,ux,uy)
            div = abs(ifftn(fftshift(fftshift(fftn(ux)).*obj.kx+fftshift(fftn(uy)).*obj.ky)));
            meandiv = max(max(div));
        end
        function [wz] = velTOvor(obj,ux,uy)
            wz = real(ifftn(fftshift(fftshift(fftn(uy)).*obj.kx)) - ifftn(fftshift(fftshift(fftn(ux)).*obj.ky)));
        end
        function [ux, uy] = vorTOvel(obj,wz)
            ux = -real(ifftn(fftshift(fftshift(fftn(wz)).*obj.ky./obj.k2)));
            uy = real(ifftn(fftshift(fftshift(fftn(wz)).*obj.kx./obj.k2)));
        end
        
        function f = PoissonSolve(obj,f)
        % PoissonSolve by Spectral method
            f = fftn(f);
            sx = sin(pi*(obj.iix-1)/obj.resx)/obj.dx;
            sy = sin(pi*(obj.iiy-1)/obj.resy)/obj.dy;
            denom = sx.^2 + sy.^2;
            fac = -0.25./denom;
            fac(1,1) = 0;
            f = f .* fac;
            f = ifftn(f);
        end
    end
end