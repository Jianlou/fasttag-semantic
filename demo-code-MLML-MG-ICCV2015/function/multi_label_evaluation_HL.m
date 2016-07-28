function hierarchical_loss_average_0 = multi_label_evaluation_HL(label_matrix_prediction,label_matrix_gt, options)
% this function implements evaluation metric 'semantic hiearchical loss', please see our paper (ML-MG, ICCV 2015) for detailed definition

% Input: 
%       label_matrix_prediction, m x n matrix
%       label_matrix_gt, m x n matrix, each entry \in {-1, 1}
%       options, has the following fields
%            -- top-k, the top-k labels are set as 1, while others as -1, for each instance

if nargin == 2
    options.topk = 5;
end

[m, n] = size(label_matrix_prediction);

%%  transform the ranking matrix to discrete top-k label matrix
    % if the prediction is already discrete
    if length(unique(label_matrix_prediction(:))) == 2  
        label_matrix_prediction_binary = false(m,n);
        label_matrix_prediction_binary(label_matrix_prediction==1) = 1;
    else
        label_matrix_prediction_binary = false(m,n); % logical matrix
        topk = options.topk;
            for i = 1:n
                label_vector_i = label_matrix_prediction(:,i);
                [~, indx] = sort(label_vector_i, 'descend');
                binary_label_i = false(m,1);
                binary_label_i(indx(1:topk)) = 1;
                label_matrix_prediction_binary(:,i) = binary_label_i;
            end
    end

%% Evaluation metric
label_matrix_gt(label_matrix_gt==-1) = 0; % transform to binary ground-truth label matrix
label_matrix_gt = logical(label_matrix_gt); % transform to logical

global edgeMatrix  % see the demo
numEdges = size(edgeMatrix, 1); 

hierarchical_loss_vector_0 = zeros(1,n);
hierarchical_loss_vector_1 = zeros(1,n);
%              tic
     for i = 1:n
            z_i = label_matrix_prediction_binary(:,i); % m x 1 vector, prediction label vector of instance i
            y_i = label_matrix_gt(:,i); % m x 1 vector, ground-truth label vector of instance i

            for e = 1:numEdges
                  child_e = edgeMatrix(e,1);
                  parent_e = edgeMatrix(e,2);
                  if ( z_i(parent_e) == 0) && ( y_i(parent_e) == 0) && ( z_i(child_e) ~= y_i(child_e) )  % parents are 0, but children are different
                      hierarchical_loss_vector_0(i) = hierarchical_loss_vector_0(i) + 1;
                  end
            end
     end
     hierarchical_loss_average_0 = mean(hierarchical_loss_vector_0)/numEdges; 


