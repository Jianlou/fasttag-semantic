function [W_mvsmp] = mv_semantic_nnn(xTr, yTr, xTe, yTe, topK, valIdx)

for par=[1e-1]
%% 1.Multi-view learning
ParaM1.alpha1 = 1e-2;
ParaM1.alpha2 = 1e-2;
ParaM1.alpha3 = 1e-2;
ParaM1.d = 3000;
ParaM1.kNN = 10;
ParaM1.typeProj = 1;
ParaM1.typeLap = 1;
% [MultiM, vFeature, indx, normP, trteindx] = mutiview_feature_learning_nnn(xTr, yTr, xTe, ParaM1);
% name = [num2str(par) 'se_multiview_learning7.mat'];
% save(name,'MultiM', 'vFeature', 'indx', 'normP','trteindx');
load('0se_multiview_learning.mat');

%% 2.Semantic mapping
ParaM2.beta1 = 1e-4;
ParaM2.beta2 = 1e-2;
ParaM2.beta3 = 1e-1;
ParaM2.beta4 = 0;
ParaM2.beta5 = 0;
ParaM2.bs = 1;
ParaM2.wlabel = 1;
ParaM2.kNN = ParaM1.kNN;
load('/Users/lou/Desktop/corel5k_tagVector.mat');
tagV = tagVector';
tagV = tagV./repmat(sqrt(sum(tagV.^2,1)),[size(tagV,1),1]);
[predTe] = semantic_mapping_nnn(vFeature, yTr, tagV, trteindx, ParaM2);

%% 3.Evaluate
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);

end
end