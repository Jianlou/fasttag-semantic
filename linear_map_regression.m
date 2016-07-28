function [W, prec, rec, f1, retrieved] = linear_map_regression(xTr, yTr, xTe, yTe, topK, valIdx)

for beta3 = [1e-2];
    beta3
    beta2 = 1;
    beta1 = 1e-2;
% corel5k 1e-2 esp 5e-4
tic
[W,b,P] = linear_map_analy(xTr, yTr, beta1, beta2, beta3);
Z = P'*xTe;
predTe = W*Z + b*ones(size(Z,2),1)';
% predTe = W*Z;
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end
end