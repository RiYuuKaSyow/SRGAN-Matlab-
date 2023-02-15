classdef preluLayer < nnet.layer.Layer % & nnet.layer.Acceleratable
    % Example custom PReLU layer.
    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficient
        Alpha
    end
    
    methods
        function layer = preluLayer(ch, args) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % with numChannels channels and specifies the layer name.
            % Set layer name.
            arguments
                ch = 64 ;
                args.Name = "" ;
            end
            layer.Name = args.Name;
            % Set layer description.
            layer.Description = "PReLU" + args.Name ;
            layer.Alpha = rand([1 1 ch]) ;
        end

%         function layer = initialize(layer,layout)
%             % layer = initialize(layer,layout) initializes the layer
%             % learnable parameters using the specified input layout.
% 
%             % Input data size.
%             sz = layout.Size;
%             ndims = numel(sz);
% 
%             % Find number of channels.
%             idx = finddim(layout,"C");
%             numChannels = sz(idx);
% 
%             % Initialize Alpha.
%             szAlpha = ones(1,ndims);
%             szAlpha(idx) = numChannels;
%             layer.Alpha = rand(szAlpha);
%         end

        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            if isnan(layer.Alpha)
                fprintf(layer.Name) ;
            end
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end
    end
end