classdef PixelShuffleLayer < nnet.layer.Layer 

    %%%%%%
    % PixelShuffleLayer
    % ref : https://github.com/pytorch/pytorch/blob/3a66a1cb99d7e1c1fe7abf8f390e177b9183a436/aten/src/ATen/native/PixelShuffle.cpp#L13-L58
    %%%
    % Args:
    %   - scale(double) : factor to increase resolution
    % Shape:
    %   - Input : (H, W, C, B)
    %   - Output: (H*sH, W*sW, C/(sH*sW), B)
    %%%%%%

    properties
        scaleW
        scaleH
    end

    methods
        function obj = PixelShuffleLayer(scale,args)

            arguments
                scale = 2 ;
                args.Name = "" ;
            end

            obj.Name = args.Name ;

            if numel(scale) <2
                scale = [scale scale] ;
            end
            
            obj.scaleH = scale(1) ;
            obj.scaleW = scale(2) ;
        end

        function z = predict(obj,x)
            
            %input size 
            sx = size(x) ;
            if numel(sx) < 4
                sx(4) = 1 ;
            end
            % Output Size
            y = zeros( sx(1)*obj.scaleH, sx(2)*obj.scaleW, sx(3)/(obj.scaleH*obj.scaleW), sx(4) ) ;
            sy = size(y) ;
            if numel(sy) < 4
                sy(4) = 1 ;
            end

%             for c = 1:sy(3)
%                 for h = 1: sy(2)
%                     for w = 1:sy(1)
%                         hidx = floor( (h-1)/obj.scale  ) + 1 ;
%                         widx = floor( (w-1)/obj.scale  ) + 1 ;
%                         cidx = (obj.scale * mod(h-1, obj.scale)) + mod(w-1, obj.scale) + (c-1)*(obj.scale^2) + 1 ;
%                         y(w, h, c,:) = x(widx, hidx, cidx,:) ;
%                     end
%                 end
%             end

            %               ih     iw     sh        sw       oc    b
            y = reshape(x,[sx(1) sx(2) obj.scaleH obj.scaleW sy(3) sy(4)]) ;
            %              sh ih sw iw oc b  
            y = permute(y,[ 3 1 4 2 5 6 ]) ;
            %              oh     ow    oc   b
            y = reshape(y,[sy(1) sy(2) sy(3) sy(4)]) ;

            z = dlarray(y,"SSCB") ;
            z = stripdims(z) ;
        end
    end
end