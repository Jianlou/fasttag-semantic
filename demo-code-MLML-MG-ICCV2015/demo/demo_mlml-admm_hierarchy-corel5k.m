clear 
clear all
chdir('D:\demo-code-ML-MG-ICCV2015')
addpath(genpath(pwd))

%% load the train and test data, Dense sift features, 4999 = 4500 + 499 samples, 1000 features
dataset_train=double(vec_read('corel5k_train_DenseSift.hvecs'));
dataset_test=double(vec_read('corel5k_test_DenseSift.hvecs'));
dataset_matrix=[dataset_train;dataset_test];

[num_sample,num_dimension] = size(dataset_matrix);
num_sample_train = size(dataset_train,1);
num_sample_test = size(dataset_test,1);

%% normalize each feature to [-1,1]
for iter_fold=1:num_dimension
    ma=max(dataset_matrix(:,iter_fold));
    mi=min(dataset_matrix(:,iter_fold));
    range=ma-mi;
    dataset_matrix(:,iter_fold)=2.*((dataset_matrix(:,iter_fold)-mi)./range-0.5);
end
dataset_train = dataset_matrix(1:num_sample_train,:);
dataset_test = dataset_matrix(1+num_sample_train:end,:);

%% compute V_x and L_x, based on kd-tree
%run('.\vlfeat-0.9.19\toolbox\vl_setup')
matlabpool('open',16);
num_neighbor_size = 20;  num_kernel_size = 7;
batch = 50; 
V_x_kdtree_compute % return V_x and L_x

%% read the ture label matrix, 268 classes
label_train_original=double(vec_read('corel5k_train_annot.hvecs'))'; % num_c x num_sample_train
label_test_original =double(vec_read('corel5k_test_annot.hvecs'))';  % num_c x num_sample_test
[num_c, ~] = size(label_train_original);

%% load the label hierarchy information for Corel 5k
load('corel5k_hierarchy_structure');
global edgeMatrix
edgeMatrix = corel5k_hierarchy_structure.edgeMatrix;
parent_matrix = corel5k_hierarchy_structure.parent_matrix;
ancestor_matrix = corel5k_hierarchy_structure.ancestor_matrix; % num_c x num_c
leaf_class_index =  (sum(ancestor_matrix)==0)'; % 206 leaf nodes
label_train_full = full(corel5k_hierarchy_structure.label_train_full); % num_c x num_sample_train
label_test_full = full(corel5k_hierarchy_structure.label_test_full);   % num_c x num_sample_test
numEdges = size(edgeMatrix,1); 

view(biograph(parent_matrix'))

Phi_matrix_1 = zeros(num_c, numEdges);
for e = 1:numEdges
    child_e = edgeMatrix(e, 1);
    parent_e = edgeMatrix(e, 2);
    Phi_matrix_1(parent_e, e) = 1;
    Phi_matrix_1(child_e, e) = -1;
end
Phi_matrix_1 = sparse(Phi_matrix_1);

% this two are used as ground-truth label matrices for evaluation
y_train = label_train_full; y_train(y_train==0)=-1; 
y_test = label_test_full; y_test(y_test==0)=-1; 


%% 
global positive_label_weight
positive_label_weight=100;
unlabel_value = 0.5;
thresh_vector = 1; %[0.8,0.4,0.2, 0.1, 0.05];

lambda_x = 0.1;
lambda_c = 1e-2;

tic
max_iter=1;
len_thresh = length(thresh_vector);

result_cell_train_admm_1 = cell(len_thresh, max_iter);
result_cell_train_admm_2 = cell(len_thresh, max_iter);
result_cell_train_admm_3 = cell(len_thresh, max_iter);
result_cell_train_admm_4 = cell(len_thresh, max_iter);
result_cell_test_admm_1 = cell(len_thresh, max_iter);
result_cell_test_admm_2 = cell(len_thresh, max_iter);
result_cell_test_admm_3 = cell(len_thresh, max_iter);
result_cell_test_admm_4 = cell(len_thresh, max_iter);

HL_0_cell_train_1 = cell(len_thresh, max_iter);
HL_0_cell_test_1 = cell(len_thresh, max_iter);

HL_0_cell_train_4 = cell(len_thresh, max_iter);
HL_0_cell_test_4 = cell(len_thresh, max_iter);


for iter_thresh=1:len_thresh
    Thresh=thresh_vector(iter_thresh);

        for iter=1:max_iter

        %% generating the missing labels
            label_train_missing_original=label_train_original;
            hide = rand(num_c,num_sample_train)>Thresh;   %The element at the position whose value is larger than Thresh will be deleted
            [m,n] = find( hide &  repmat(leaf_class_index, 1, num_sample_train) );
            for k=1:length(m)
                label_train_missing_original(m(k), n(k)) = unlabel_value;
            end
            initial_assign_matrix_original=[label_train_missing_original, unlabel_value.*ones(num_c,num_sample_test)];  

         % complete the label matrix according to the label hierarchy
            label_train_missing = label_train_missing_original; 
            for i = 1:num_c
                ancestor_i = ancestor_matrix(i,:)>0;
                sample_i = label_train_missing(i,:)==1;
                label_train_missing(ancestor_i, sample_i) = 1;
            end

            %% class-level correlation
            initial_assign_matrix=[label_train_missing, unlabel_value.*ones(num_c,num_sample_test)];  
            num_neighbor =10;
            tic
            [V_c_normalized,V_c, Vc_cell]=Vc_compute(2.*initial_assign_matrix-1, num_neighbor);
            toc
            L_c = eye(num_c, num_c) - V_c_normalized;

      %% call MLML-ADMM method
            topk = 5;

             %% case 1, no semantic hierarchy constraint, with filled initial label matrix
              
             lambda_x = 1e-1;
             lambda_c = 1e-2;             
             tic
             Phi_matrix = zeros(num_c, numEdges);
             options.max_iter =100;
             options.gap_compute = 3;
             options.rate_step = 0.9;
             options.alpha_rate = 3;
             [Z_admm_complete_1, Z_cell_1, obj_L_1, obj_Z_1] = MLML_MG_ICCV_2015(initial_assign_matrix,L_x, L_c, lambda_x, lambda_c, Phi_matrix, options);
             toc
             
            % evaluation using multiple metrics
            [result_compact_train, result_compact_test] = multi_label_evaluation_wrapped(Z_admm_complete_1, y_train, y_test, topk);
            result_compact_train = [Thresh; lambda_x; lambda_c;  result_compact_train];
            result_compact_test  = [Thresh; lambda_x; lambda_c;  result_compact_test];
            [result_compact_train,result_compact_test]
            result_cell_train_admm_1{iter_thresh, iter} = result_compact_train;
            result_cell_test_admm_1{iter_thresh, iter} = result_compact_test;    
            
            % hierarchical hamming loss
            topk_vector = [ 5 10 20 50 100 150 ];
            for i = 1: length(topk_vector)
                  opt.topk = topk_vector(i);
                  HL_0_train_1(i) = multi_label_evaluation_HL(Z_admm_complete_1(:,1:num_sample_train), y_train, opt);
                  HL_0_test_1(i)  = multi_label_evaluation_HL(Z_admm_complete_1(:,num_sample_train+1:end), y_test, opt);
            end
            HL_0_cell_train_1{iter_thresh, iter} = HL_0_train_1';
            HL_0_cell_test_1{iter_thresh, iter} = HL_0_test_1';  
            
             %% case 2, with semantic hierarchy constraint, with filled initial label matrix
             lambda_x = 1e-1;
             lambda_c = 1e-2;             
             tic
             Phi_matrix = Phi_matrix_1;
             % parameters of PGD
             options.max_iter = 10;
             options.gap_compute = 1;
             options.rate_step = 0.9;
             options.alpha_rate = 5;  
             % parameters of ADMM
             options.max_iter_overall =50;
             options.rho_0 = 10;
             options.rho_gap = 10;
             options.rho_rate = 10;
             options.initialization = 'zero';
             [Z_admm_complete_2, Z_cell_2, obj_L_2, obj_Z_2]=MLML_MG_ICCV_2015(initial_assign_matrix,L_x, L_c, lambda_x, lambda_c, Phi_matrix, options);
             toc

             % evaluation using multiple metrics
            [result_compact_train, result_compact_test] = multi_label_evaluation_wrapped(Z_admm_complete_2, y_train, y_test, topk);
            result_compact_train = [Thresh; lambda_x; lambda_c;  result_compact_train];
            result_compact_test  = [Thresh; lambda_x; lambda_c;  result_compact_test];
            [result_compact_train,result_compact_test]
            result_cell_train_admm_2{iter_thresh, iter} = result_compact_train;
            result_cell_test_admm_2{iter_thresh, iter} = result_compact_test;   
            
            % when using semantic hierarchy constraint, hierarchical hamming loss=0, thus we don't need to compute it
            
              %% case 3, no semantic hierarchy constraint, with unfilled initial label matrix
             tic
             Phi_matrix = zeros(num_c, numEdges);
             % parameters of PGD
             options.max_iter =100;
             options.gap_compute = 3;
             options.rate_step = 0.9;
             options.alpha_rate = 1;
             [Z_admm_complete_3, obj_L_3, obj_Z_3] = MLML_MG_ICCV_2015(initial_assign_matrix_original,L_x, L_c, lambda_x, lambda_c, Phi_matrix, options);
             toc

            % evaluation using multiple metrics
            [result_compact_train, result_compact_test] = multi_label_evaluation_wrapped(Z_admm_complete_3, y_train, y_test, topk);
            result_compact_train = [Thresh; lambda_x; lambda_c;  result_compact_train];
            result_compact_test  = [Thresh; lambda_x; lambda_c;  result_compact_test];
            [result_compact_train,result_compact_test]
            result_cell_train_admm_3{iter_thresh, iter} = result_compact_train;
            result_cell_test_admm_3{iter_thresh, iter} = result_compact_test;   
            
            % when using semantic hierarchy constraint, hierarchical hamming loss=0, thus we don't need to compute it
            
             %% case 4, with semantic hierarchy constraint, with unfilled initial label matrix
             lambda_x = 1e2;
             lambda_c = 1e-1;             
             tic
             Phi_matrix = Phi_matrix_1;
             % parameters of PGD
             options.max_iter = 1;
             options.gap_compute = 1;
             options.rate_step = 1;
             options.alpha_rate =1;  
             % parameters of ADMM
             options.max_iter_overall =100;
             options.rho_0 = 1e2;
             options.rho_gap = 50;
             options.rho_rate = 10;
             options.initialization = 'zero';
             [Z_admm_complete_4, obj_L_4, obj_Z_4] = MLML_MG_ICCV_2015(initial_assign_matrix_original,L_x, L_c, lambda_x, lambda_c, Phi_matrix, options);
             toc
             
             % evaluation using multiple metrics
             [result_compact_train, result_compact_test] = multi_label_evaluation_wrapped(Z_admm_complete_4, y_train, y_test, topk);
             result_compact_train = [Thresh; lambda_x; lambda_c;  result_compact_train];
             result_compact_test  = [Thresh; lambda_x; lambda_c;  result_compact_test];
             [result_compact_train,result_compact_test]
             result_cell_train_admm_4{iter_thresh, iter} = result_compact_train;
             result_cell_test_admm_4{iter_thresh, iter} = result_compact_test;   

            % hierarchical hamming loss
            topk_vector = [ 5 10 20 50 100 150 ];
            for i = 1: length(topk_vector)
                  opt.topk = topk_vector(i);
                  HL_0_train_4(i) = multi_label_evaluation_HL(Z_admm_complete_4(:,1:num_sample_train), y_train, opt);
                  HL_0_test_4(i) = multi_label_evaluation_HL(Z_admm_complete_4(:,num_sample_train+1:end), y_test, opt);
            end
            HL_0_test_4        
            HL_0_cell_train_4{iter_thresh, iter} = HL_0_train_4';
            HL_0_cell_test_4{iter_thresh, iter} = HL_0_test_4';  
            
%             figure; plot(1:length(obj_L_4), obj_L_4, 'g-', 1:length(obj_Z_4), obj_Z_4, 'r-'), hold on
%             legend('obj of L', 'obj of f(Z)'), 
%             title('the convergence curve of MLML-ADMM method')                  
        end
end


%% the final results of MLML-ADMM
for iter_thresh = 1:len_thresh
    result_final_train = []; 
    result_final_test = []; 
    aa = [result_cell_train_admm_1{iter_thresh, :}];
    result_final_train = [mean(aa,2),std(aa,0,2)];
    
    bb = [result_cell_test_admm_1{iter_thresh, :}];
    result_final_test = [mean(bb,2),std(bb,0,2)];  
    [result_final_train, result_final_test]
end

for iter_thresh = 1:len_thresh
    result_final_train = []; 
    result_final_test = []; 
    aa = [result_cell_train_admm_2{iter_thresh, :}];
    result_final_train = [mean(aa,2),std(aa,0,2)];
    
    bb = [result_cell_test_admm_2{iter_thresh, :}];
    result_final_test = [mean(bb,2),std(bb,0,2)];  
    [result_final_train, result_final_test]
end

for iter_thresh = 1:len_thresh
    result_final_train = []; 
    result_final_test = []; 
    aa = [result_cell_train_admm_3{iter_thresh, :}];
    result_final_train = [mean(aa,2),std(aa,0,2)];
    
    bb = [result_cell_test_admm_3{iter_thresh, :}];
    result_final_test = [mean(bb,2),std(bb,0,2)];  
    [result_final_train, result_final_test]
end

for iter_thresh = 1:len_thresh
    result_final_train = []; 
    result_final_test = []; 
    aa = [result_cell_train_admm_4{iter_thresh, :}];
    result_final_train = [mean(aa,2),std(aa,0,2)];
    
    bb = [result_cell_test_admm_4{iter_thresh, :}];
    result_final_test = [mean(bb,2),std(bb,0,2)];  
    [result_final_train, result_final_test]
end

%% final results of hierarchical hamming loss
for iter_thresh = 1:len_thresh
    result_final_train = []; 
    result_final_test = []; 
    aa = [HL_0_cell_train_1{iter_thresh, :}];
    result_final_train = [mean(aa,2), std(aa,0,2)];
    
    bb = [HL_0_cell_test_1{iter_thresh, :}];
    result_final_test = [mean(bb,2), std(bb,0,2)];    
    
    [result_final_train, result_final_test]
end

for iter_thresh = 1:len_thresh
    result_final_train = []; 
    result_final_test = []; 
    aa = [HL_0_cell_train_4{iter_thresh, :}];
    result_final_train = [mean(aa,2), std(aa,0,2)];
    
    bb = [HL_0_cell_test_4{iter_thresh, :}];
    result_final_test = [mean(bb,2), std(bb,0,2)]; 
    
    [result_final_train, result_final_test]
end

