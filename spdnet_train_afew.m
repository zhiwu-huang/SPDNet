function [net, info] = spdnet_train_afew(net, spd_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ;
opts.val = find(spd_train.spd.set==2) ;
    
for epoch=1:opts.numEpochs
    learningRate = opts.learningRate(epoch);
    
     % fast-forward to last checkpoint
     modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
     if opts.continue
         if exist(modelPath(epoch),'file')
             if epoch == opts.numEpochs
                 load(modelPath(epoch), 'net', 'info') ;
             end
             continue ;
         end
         if epoch > 1
             fprintf('resuming by loading epoch %d\n', epoch-1) ;
             load(modelPath(epoch-1), 'net', 'info') ;
         end
     end
    
    train = opts.train(randperm(length(opts.train))) ; % shuffle
    val = opts.val;

    [net,stats.train] = process_epoch(opts, epoch, spd_train, train, learningRate, net) ;
    [net,stats.val] = process_epoch(opts, epoch, spd_train, val, 0, net) ;

   
    % save
    evaluateMode = 0;
    
    if evaluateMode, sets = {'train'} ; else sets = {'train', 'val'} ; end

    for f = sets
        f = char(f) ;
        n = numel(eval(f)) ; %
        info.(f).objective(epoch) = stats.(f)(2) / n ;
        info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
    end
    if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end

    figure(1) ; clf ;
    hasError = 1 ;
    subplot(1,1+hasError,1) ;
    if ~evaluateMode
        semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
        hold on ;
    end
    semilogy(1:epoch, info.val.objective, '.--') ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend(sets) ;
    set(h,'color','none');
    title('objective') ;
    if hasError
        subplot(1,2,2) ; leg = {} ;
        if ~evaluateMode
            plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
            hold on ;
            leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
        end
        plot(1:epoch, info.val.error', '.--') ;
        leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
        set(legend(leg{:}),'color','none') ;
        grid on ;
        xlabel('training epoch') ; ylabel('error') ;
        title('error') ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
end
    
    
    
function [net,stats] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net)

training = learningRate > 0 ;
if training, mode = 'training' ; else mode = 'validation' ; end

stats = [0 ; 0 ; 0] ;
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;


for ib = 1 : batchSize : length(trainInd)
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ;
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;
    else
        batchSize_r = batchSize;
    end
    spd_data = cell(batchSize_r,1);    
    spd_label = zeros(batchSize_r,1);
    for ib_r = 1 : batchSize_r
        spdPath = [spd_train.spdDir '\' spd_train.spd.name{trainInd(ib+ib_r-1)}];
        load(spdPath); spd_data{ib_r} = Y1;
        spd_label(ib_r) = spd_train.spd.label(trainInd(ib+ib_r-1));

    end
    net.layers{end}.class = spd_label ;

    %forward/backward spdnet
    if training, dzdy = one; else dzdy = [] ; end
    res = vl_myforbackward(net, spd_data, dzdy, res) ;
   
    %accumulating graidents
    if numGpus <= 1
      [net,res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end
          
    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;

    numDone = numDone + batchSize_r ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    stats = stats+[batchTime ; res(end).x ; error]; % works even when stats=[]
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;

    fprintf(' error: %.5f', stats(3)/numDone) ;
    fprintf(' obj: %.5f', stats(2)/numDone) ;

    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf('\n') ;
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=numel(net.layers):-1:1
  if isempty(res(l).dzdw)==0
    if ~isfield(net.layers{l}, 'learningRate')
       net.layers{l}.learningRate = 1 ;
    end
    if ~isfield(net.layers{l}, 'weightDecay')
       net.layers{l}.weightDecay = 1;
    end
    thisLR = lr * net.layers{l}.learningRate ;

    if isfield(net.layers{l}, 'weight')
        if strcmp(net.layers{l}.type,'bfc')==1
            W1=net.layers{l}.weight;
            W1grad = (1/batchSize)*res(l).dzdw;
            %gradient update on Stiefel manifolds
            problemW1.M = stiefelfactory(size(W1,1), size(W1,2));
            W1Rgrad = (problemW1.M.egrad2rgrad(W1, W1grad));
            net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); %%!!!NOTE
        else
            net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize)* res(l).dzdw ;
        end
    
    end
  end
end

