function [u,v,net] = spectralNormalize(u,v,net)

    if ~iscell(u) && ~iscell(v)
        layerSize = size(net.Learnables.Value,1) ;
        u = cell(layerSize,1) ;
        v = cell(layerSize,1) ;
        for i = 1:layerSize
            ks = size(net.Learnables.Value{i,1}) ;
            if net.Learnables.Parameter{i} == "Weights" && numel(ks) == 4
                u{i,1} = rand(ks(4),1) ;
                v{i,1} = rand(1,ks(1)*ks(2)*ks(3)) ;
            else
                u{i,1} = nan ;
                v{i,1} = nan ;
            end
        end
    end

    layerSize = size(net.Learnables.Value,1) ;
    
    for i = 1:layerSize
        if net.Learnables.Parameter{i} == "Weights" && size(u{i},1)~=1 && size(v{i},1)~=1
            nf = size(net.Learnables.Value{i,1},4) ;
            bar = reshape(net.Learnables.Value{i} , nf ,[] ) ;
        
            v{i} = l2normalize( bar' * u{i} ) ;
            u{i} = l2normalize( bar * v{i} ) ;
        
            sigma = u{i}' * bar * v{i} ;
        
            net.Learnables.Value{i} =  net.Learnables.Value{i} ./ sigma  ;
        end
    end

end

function v = l2normalize(v, eps)
            if ~exist('eps','var')
                eps = 1e-12 ;
            end
            v = v ./ ( sum( v .^ 2,"all" )^ 0.5  +eps) ;
%             v = v ./ ( norm(v) + eps) ;
end