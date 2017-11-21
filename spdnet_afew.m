function [net, info] = spdnet_afew(varargin)
%set up the path
confPath;
%parameter setting
opts.dataDir = fullfile('./data/afew') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'spddb_afew_train_spd400_int_histeq.mat');
opts.batchSize = 30 ;
opts.test.batchSize = 1;
opts.numEpochs = 500 ;
opts.gpus = [] ;
opts.learningRate = 0.01*ones(1,opts.numEpochs);
opts.weightDecay = 0.0005 ;
opts.continue = 1;
%spdnet initialization
net = spdnet_init_afew() ;
%loading metadata 
load(opts.imdbPathtrain) ;
%spdnet training
[net, info] = spdnet_train_afew(net, spd_train, opts);


