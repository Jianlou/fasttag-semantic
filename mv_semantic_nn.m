function [W_mvsmp] = mv_semantic_nn(xTr, yTr, xTe, yTe, topK, valIdx)

for par=[1e-1 1e-2 1e-3 1e-4 1e-5 0]
%% 1.Multi-view learning
ParaM1.alpha1 = 1e-2;
ParaM1.alpha2 = 1e-2;
ParaM1.alpha3 = 1e-2;
ParaM1.d = 3000;
ParaM1.kNN = 10;
ParaM1.typeProj = 1;
ParaM1.typeLap = 1;
% [MultiM, vTrain, indx, normP] = mutiview_feature_learning_nn(xTr, yTr, ParaM1);
% name = [num2str(par) 'multiview_learning_new_alpha1.mat'];
% save(name,'MultiM', 'vTrain', 'indx', 'normP');
load('0multiview_learning_new_alpha1.mat');

%% 2.Semantic mapping
ParaM2.beta1 = 1e-2;
ParaM2.beta2 = 0;
ParaM2.beta3 = 0;
ParaM2.beta4 = 0;
ParaM2.beta5 = 0;
ParaM2.bs = 1;
ParaM2.wlabel = 1;
ParaM2.kNN = ParaM1.kNN;
load('/Users/lou/Desktop/corel5k_tagVector.mat');
tagV = tagVector';
tagV = tagV./repmat(sqrt(sum(tagV.^2,1)),[size(tagV,1),1]);
[W_mvsmp] = semantic_mapping_nn(vTrain, yTr, tagV, ParaM2);

%% 3.Tag inference
ParaM3.alpha2 = ParaM1.alpha2;
ParaM3.alpha3 = ParaM1.alpha2;
ParaM3.kNN = ParaM2.kNN;
ParaM3.beta2 = ParaM2.beta2;
ParaM3.beta3 = ParaM2.beta3;
ParaM3.beta4 = ParaM2.beta4;
ParaM3.beta5 = ParaM2.beta5;
ParaM3.typeProj = 1;
[vTest] = mutiview_feature_learning_test(xTe, MultiM, indx, normP, ParaM3);
[predTe] = semantic_mapping_test(vTest, yTr, W_mvsmp, tagV,ParaM3);

%% 4.Evaluate
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);

end
end