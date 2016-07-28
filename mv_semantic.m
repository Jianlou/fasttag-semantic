function [W_mvsmp] = mv_semantic(xTr, yTr, xTe, yTe, topK, valIdx)

for par=[1000]
%% 1.Multi-view learning
ParaM1.alpha1 = 1e-2;
ParaM1.alpha2 = 1e-1;
ParaM1.alpha3 = 1e3;
ParaM1.d = 3000;
% [MultiM, Theta, vTrain, indx] = mutiview_feature_learning(xTr, yTr, ParaM1);
% name = [num2str(par) 'multiview_learning.mat'];
% save(name,'MultiM', 'Theta', 'vTrain', 'indx');
load('1000multiview_learning.mat');

%% 2.Semantic mapping
ParaM2.beta1 = 1e-3;
ParaM2.beta2 = 1e-4;
load('/Users/lou/Desktop/corel5k_tagVector.mat');
tagV = tagVector';
tagV = tagV./repmat(sqrt(sum(tagV.^2,1)),[size(tagV,1),1]);
[W_mvsmp, b] = semantic_mapping(vTrain, yTr, tagV, ParaM2);

%% 3.Tag inference
ParaM3.alpha2 = ParaM1.alpha2;
ParaM3.gamma1 = 1e-3;
ParaM3.gamma2 = 0;
ParaM3.gamma3 = 0;
ParaM3.gamma4 = 0;
ParaM3.sigma = 1;
ParaM3.kNN = 50;
[predTe] = tag_inference(xTe, MultiM, indx, Theta, W_mvsmp, b, tagV, ParaM3, yTr);

%% 4.Evaluate
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);

end
end