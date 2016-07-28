function [W, prec, rec, f1, retrieved] = mv_multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx)

[xTrain, xTest, xTeSem] = multiview_feature(xTr, xTe, yTr);

clear xTr xTe

for beta = [1e-2]
tic
[W,b] = linear_analy(xTrain, yTr, beta);
predTe = W*xTest + b*ones(size(xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end

for beta1 = [5e-3]
    beta2 =1e-3;
tic
[W,b] = multitask_analy(xTrain, yTr, beta1, beta2);
predTe = W*xTest + b*ones(size(xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end
end