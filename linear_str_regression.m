function [W, prec, rec, f1, retrieved] = linear_str_regression(xTr, yTr, xTe, yTe, topK, valIdx)

for beta1 = [1e-2];
    lambda1 = 1;
    lambda2 = 1;
    lambda3 = 1;
    alpha1 = 1e-2;
    alpha2 = 1e-2;
    d = 3000;
tic
[W,b,Z] = linear_str_analy(xTr, yTr, xTe, alpha1, alpha2, beta1, lambda1, lambda2, lambda3, d);
predTe = W*Z + b*ones(size(Z,2),1)';
% predTe = W*Z;
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end
end