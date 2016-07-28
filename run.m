addpath('util/')
addpath('preprocess/')
addpath('baseline/')
addpath('fasttag/')
addpath(genpath('spams-matlab/'))
addpath('multitask/')
% addpath('/media/pris/Study/Tools/DimensionalityReduction/drtoolbox/techniques/')


topK = 5;
dimen = 500;


datasets = {'corel5k'; 'espgame'; 'iaprtc12'};

for i = 1%size(datasets, 1)

	dataset = datasets{i};
	dataFolder = ['./feature/', dataset, '/'];

	filename=[dataFolder, 'data,dimen=', num2str(dimen), '.mat']; 
	if exist(filename)
		load(filename);
	else
		% preprocessing, includes approximated additive kernel mapping and random projection to reduce dimension
		[xTr, yTr, xTe, yTe, valIdx] = loaddata(dataFolder, dimen);	
    end
    
%     for j = 1:length(xTr)
%         meanX{j} = mean(xTr{j},2);
%         stdX{j} = std(xTr{j}')';
%         xTr{j} = xTr{j} - repmat(meanX{j}, [1, size(xTr{j},2)]);
%         xTr{j} = xTr{j}./(repmat(stdX{j}, [1, size(xTr{j},2)])+1e-20);
%         xTe{j} = xTe{j} - repmat(meanX{j}, [1, size(xTe{j},2)]);
%         xTe{j} = xTe{j}./(repmat(stdX{j}, [1, size(xTe{j},2)])+1e-20);
%     end
    

	yTr = double(yTr);
	yTe = double(yTe);

% 	xTr = [xTr; ones(1, size(xTr, 2))];
% 	xTe = [xTe; ones(1, size(xTe, 2))];
    

	%linear regression baseline
%     [W_lr] = linear_regression(xTr(:, ~valIdx), yTr(:, ~valIdx), xTr(:, valIdx), yTr(:, valIdx), topK, valIdx);
%     [W_lr] = linear_regression(xTr, yTr, xTe, yTe, topK, valIdx);
    
	%linear regression baseline
%     [W_lmr] = linear_map_regression(xTr, yTr, xTe, yTe, topK, valIdx);
%     [W_baseline] = linear_mv_regression(xTr, yTr, xTe, yTe, topK, valIdx);
    
%     [W_str] = linear_str_regression(xTr, yTr, xTe, yTe, topK, valIdx);

    %Multitask linear regression
%  	[W_mlr] = multitask_linear_regression(xTr(:, ~valIdx), yTr(:, ~valIdx), xTr(:, valIdx), yTr(:, valIdx), topK);   
% 	[W_mlr] = multitask_linear_regression(xTr, yTr, xTe, yTe, topK);

    %Group Sparsity Multitask
% 	[W_gsml] = group_multitask(xTr(:, ~valIdx), yTr(:, ~valIdx), xTr(:, valIdx), yTr(:, valIdx), topK);
% 	[W_gsml] = group_multitask(xTr, yTr, xTe, yTe, topK);

	% fasttag
% 	[W_fasttag] = fasttag(xTr, yTr, xTe, yTe, topK, valIdx);

	%low-rank
%     [W_lrr] = lowrank_regression(xTr, yTr, xTe, yTe, topK, valIdx);
    
    %multi-task
%     [W_mtl] = multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx);

    %group-multi-task
%     [W_gmtl] = group_multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx);

    %group-low-rank
%     [W_glrr] = group_lowrank_regression(xTr, yTr, xTe, yTe, topK, valIdx);

    %multi-view linear regression
%     [W_mvlr] = mv_linear_regression(xTr, yTr, xTe, yTe, topK, valIdx);
    
    % multi-view multi-task
%      [W_mvmtl] = mv_multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx);
     
    %multi-view group-multi-task
%      [W_mvgmtl] = mv_group_multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx);
     
	%multi-view low-rank
%      [W_mvlrr] = mv_lowrank_regression(xTr, yTr, xTe, yTe, topK, valIdx);

    %multi-view fasttag
% 	[W_mvfasttag] = mv_fasttag(xTr, yTr, xTe, yTe, topK, valIdx);


    % multi-view and semantic mapping
% 	[W_mvsmp] = mv_semantic(xTr, yTr, xTe, yTe, topK, valIdx);
    
     % multi-view and semantic mapping
% 	[W_mvsmpn] = mv_semantic_n(xTr, yTr, xTe, yTe, topK, valIdx);


     % multi-view and semantic mapping
% 	[W_mvsmpnn] = mv_semantic_nn(xTr, yTr, xTe, yTe, topK, valIdx);
    
     % multi-view and semantic mapping
	[W_mvsmpnnn] = mv_semantic_nnn(xTr, yTr, xTe, yTe, topK, valIdx);


end
