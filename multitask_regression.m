function [W, prec, rec, f1, retrieved] = multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx)

xTrain = [];
xTest = [];
for i=1:length(xTr)
    xTrain = [xTrain;xTr{i}];
    xTest = [xTest;xTe{i}];
end

D = size(xTrain,1);
d = 5000;
P = randn(D,d);
[Q, ~] = qr(P,0);
P = Q;

clear xTr xTe

for beta = [1e-2]
tic
[W,b] = linear_analy(P'*xTrain, yTr, beta);
predTe = W*P'*xTest + b*ones(size(P'*xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end

for beta1 = [5e-3]
    beta2 =0;
tic
[W,b] = multitask_analy(P'*xTrain, yTr, beta1, beta2);
predTe = W*P'*xTest + b*ones(size(P'*xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end
end