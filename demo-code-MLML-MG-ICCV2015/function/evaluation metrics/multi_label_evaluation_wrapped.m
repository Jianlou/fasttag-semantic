function [result_compact_train, result_compact_test] = multi_label_evaluation_wrapped(label_prediction, label_train, label_test, topk)
% label_prediction, num_c x num_sample matrix
% label_train, num_c x num_sample_train matrix
% label_test, num_c x num_sample_test matrix

if nargin == 4
   options.topk = topk;
elseif nargin ==3
   options.topk = 3;
else
    fprintf('Error: the number of input varaibles is incorrect..\n')
end

num_sample_train = size(label_train,2);
label_prediction_train = label_prediction(:,1:num_sample_train); % num_c x num_sample_train
label_prediction_test = label_prediction(:,1+num_sample_train:end); % num_c x num_sample_train

%% evaluation on training data 
options.type= 'ranking'; % for continuous ranking
result_struct_train_1 = multi_label_evaluation(label_prediction_train,label_train, options);

options.type= 'AP-information-retrieval'; % for continuous ranking
result_struct_train_2 = multi_label_evaluation(label_prediction_train,label_train, options);

options.type= 'example-based-partition'; % for discrete labels
result_struct_train_3 = multi_label_evaluation(label_prediction_train,label_train, options);

options.type= 'label-based-partition'; % for discrete labels
result_struct_train_4 = multi_label_evaluation(label_prediction_train,label_train, options);

%% evaluation on testing data 
options.type= 'ranking'; % for continuous ranking
result_struct_test_1 = multi_label_evaluation(label_prediction_test,label_test, options);

options.type= 'AP-information-retrieval'; % for continuous ranking
result_struct_test_2 = multi_label_evaluation(label_prediction_test,label_test, options);

options.type= 'example-based-partition'; % for discrete labels
result_struct_test_3 = multi_label_evaluation(label_prediction_test,label_test, options);

options.type= 'label-based-partition'; % for discrete labels
result_struct_test_4 = multi_label_evaluation(label_prediction_test,label_test, options);

%%  combine above training structures
names = [fieldnames(result_struct_train_1); fieldnames(result_struct_train_2); fieldnames(result_struct_train_3); fieldnames(result_struct_train_4)];
result_struct_combine_train = cell2struct([struct2cell(result_struct_train_1); struct2cell(result_struct_train_2); ...
                                    struct2cell(result_struct_train_3); struct2cell(result_struct_train_4)], names, 1);
result_struct_combine_train.topk = topk;

result_compact_train = [result_struct_combine_train.topk, result_struct_combine_train.AUC, ...
                                  result_struct_combine_train.AP, result_struct_combine_train.MAP, result_struct_combine_train.topk_example_accuracy, ...
                                  result_struct_combine_train.topk_example_precision, result_struct_combine_train.topk_example_F1, ...
                                  result_struct_combine_train.topk_label_F1_macro_average, result_struct_combine_train.topk_label_F1_micro, ...
                                  result_struct_combine_train.topk_example_hamming_loss]'; 

%%  combine above testing structures                              
names = [fieldnames(result_struct_test_1); fieldnames(result_struct_test_2); fieldnames(result_struct_test_3); fieldnames(result_struct_test_4)];
result_struct_combine_test = cell2struct([struct2cell(result_struct_test_1); struct2cell(result_struct_test_2); ...
                                                             struct2cell(result_struct_test_3); struct2cell(result_struct_test_4)], names, 1);
result_struct_combine_test.topk = topk;

result_compact_test = [result_struct_combine_test.topk, result_struct_combine_test.AUC, ...
                                  result_struct_combine_test.AP, result_struct_combine_test.MAP, result_struct_combine_test.topk_example_accuracy, ...
                                  result_struct_combine_test.topk_example_precision, result_struct_combine_test.topk_example_F1, ...
                                  result_struct_combine_test.topk_label_F1_macro_average, result_struct_combine_test.topk_label_F1_micro, ...
                                  result_struct_combine_test.topk_example_hamming_loss]'; 
