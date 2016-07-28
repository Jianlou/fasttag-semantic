function [W, prec, rec, f1, retrieved] = mv_lowrank_regression(xTr, yTr, xTe, yTe, topK, valIdx)

[xTrain, xTest, xTeSem] = multiview_feature(xTr, xTe, yTr);

clear xTr xTe

for beta = [1e-2]
% corel5k 1e-2 esp 5e-4
tic
[W,b] = linear_analy(xTrain, yTr, beta);
% [W] = linear_analy_self(xTr, yTr, xTe, zeros(size(yTe)), yTe, beta, valIdx, topK);
predTe = W*xTest + b*ones(size(xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end

for beta1 = [1e-2]
    beta2 =0;
tic
[W,b] = lowrank_analy(xTrain, yTr, beta1, beta2);
predTe = W*xTest + b*ones(size(xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end
end