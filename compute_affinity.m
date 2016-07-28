function [M] = compute_affinity(X, K)

[d n] = size(X);
sigma = 15;

XiXi = repmat(diag(X'*X),[1 n]);
XjXj = repmat(diag(X'*X)',[n 1]);
XiXj = X'*X;
M = exp(-(XiXi+XjXj-2*XiXj)/(sigma));
[Msort, Midx] = sort(M,1,'descend');
kNearestN = Msort(K,:);
Threshold = repmat(kNearestN, [n 1]);
M(M<Threshold) =0;
M(find((M.*M')==0))=0;
end