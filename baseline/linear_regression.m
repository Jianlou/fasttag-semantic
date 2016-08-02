function [W, prec, rec, f1, retrieved] = linear_regression(xTr, yTr, xTe, yTe, topK, valIdx)

xTrain = [];
xTest = [];
for i=1:length(xTr)
    xTrain = [xTrain;xTr{i}];
    xTest = [xTest;xTe{i}];
end

[D,N] = size(xTrain);
d = 3000;
RR = randn(N,d);
temp = xTrain*RR;
[P, ~] = qr(temp,0);

clear xTr xTe
for beta = [1e-2]
% corel5k 1e-2 esp 5e-4
tic
[W,b] = linear_analy(P'*xTrain, yTr, beta);
% [W] = linear_analy_self(xTr, yTr, xTe, zeros(size(yTe)), yTe, beta, valIdx, topK);
predTe = W*P'*xTest + b*ones(size(P'*xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end
end
