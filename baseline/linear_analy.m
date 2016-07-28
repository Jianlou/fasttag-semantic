function [W,b] = linear_analy(X, Y, lambda)

[d n] = size(X);
r = size(Y,1);

Wpos = 1./max(1, sum(Y>0, 2));
% Wpos = ones(r,1);
weights = (sum(bsxfun(@times, (Y>0), Wpos), 1));
iW = spdiags([ones(d, 1)], 0, d, d);
weight = spdiags(weights', 0, n, n);
temp = weight*(eye(n,n)-(1/n)*ones(n,n));

Sx = (X*temp*temp'*X');
Sxy = (Y*temp*temp'*X');

W = Sxy/(Sx + lambda*iW);
b = (1/n)*(Y-W*X)*weight*ones(size(X,2),1);

fit_loss = norm(Y*weight-W*X*weight-b*ones(size(X,2),1)','fro');
fprintf('\n fit loss is %f \n', fit_loss);

end