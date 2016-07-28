function [W, prec, rec, f1, retrieved] = multitask_linear_regression(xTr, yTr, xTe, yTe, topK)
%%
%
% 
% 1/2||Y-XW||_F^2 + beta*||W||1,2
% Y(n x r), X(n x d), W(d x r)

%%
for beta1 = [1e-2];
    beta2 = 1;
    beta3 = 1e-2;
    % corel5k 3e-3
tic
[W,b,P] = multitask_admm(xTr, yTr, beta1, beta2, beta3);
Z = P'*xTe;
predTe = W*Z + b*ones(size(Z,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nMultitaskLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc

end

end