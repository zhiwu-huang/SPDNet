function net = spdnet_init_afew(varargin)
% spdnet_init initializes a spdnet

rng('default');
rng(0) ;

opts.layernum = 3;

Winit = cell(opts.layernum,1);
opts.datadim = [400, 200, 100, 50];


for iw = 1 : opts.layernum
    A = rand(opts.datadim(iw));
    [U1, S1, V1] = svd(A * A');
    Winit{iw} = U1(:,1:opts.datadim(iw+1));
end

f=1/100 ;
classNum = 7;
fdim = size(Winit{iw},2)*size(Winit{iw},2);
theta = f*randn(fdim, classNum, 'single');
Winit{end+1} = theta;

net.layers = {} ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{3}) ;
net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc', ...
                           'weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;



