function [ W ,b] = group_multitask_manifold_admm(X, Y, X_all , lambda, lambda2, lambda3, Idx, kNN)
%MULTITASK_ADMM Summary of this function goes here
%   Detailed explanation goes here


[d n] = size(X);
r = size(Y,1);

%% Initialize
W = zeros(r, d);
% W = randn(r, d);
Z = zeros(r, d);
% Z = W;
Q = zeros(r, d);
I = spdiags([ones(d, 1)], 0, d, d);
MU = 1e-2;

RHO = 10;
EPSILON = 1e-3;
MAXIT = 80;
iteration = 0;

%% compute affinity matrix
Maffinity = compute_affinity(X_all, kNN);
Daffinity = diag(sum(Maffinity,2));
Laffinity = Daffinity - Maffinity;
%%

Wpos = 1./max(1, sum(Y>0, 2));
weights = (sum(bsxfun(@times, (Y>0), Wpos), 1));
weight = spdiags(weights', 0, n, n);
temp = weight*(eye(n,n)-(1/n)*ones(n,n));

Sx =( X*temp*temp'*X');


% Wpos = 1./max(1, sum(Y(Idx(1,k):Idx(2,k),:)>0, 2));
% weights = max(bsxfun(@times, (Y(Idx(1,k):Idx(2,k),:)>0), Wpos), [], 1).^1;
% Sx = X*spdiags(weights', 0, n, n)*X';  
Sxy = (Y*temp*temp'*X');

[dd nn] = size(X_all);
Sxlx = (X_all*Laffinity*X_all');

while true
 iteration = iteration +1;
 W = (Sxy - Q + MU*Z)/(Sx + lambda2*I + MU*I + lambda3*Sxlx);
 b = (1/n)*(Y-W*X)*weight*ones(size(X,2),1);
 
 for k=1:size(Idx,2)
     U(Idx(1,k):Idx(2,k),:) = W(Idx(1,k):Idx(2,k),:) + Q(Idx(1,k):Idx(2,k),:)/MU;
     LAMBDA1 = lambda/MU;
     U_NORM = max(sqrt(sum(U(Idx(1,k):Idx(2,k),:).^2,1)) - LAMBDA1,zeros(1,size(U(Idx(1,k):Idx(2,k),:),2)))./(sqrt(sum(U(Idx(1,k):Idx(2,k),:).^2,1))+10e-20);
     Z_prev(Idx(1,k):Idx(2,k),:) = Z(Idx(1,k):Idx(2,k),:);
     Z(Idx(1,k):Idx(2,k),:) = repmat(U_NORM,[size(U(Idx(1,k):Idx(2,k),:),1),1]).*U(Idx(1,k):Idx(2,k),:);
 end

 Q = Q + MU*(W-Z);
 
 primal_residual = norm(W-Z,'fro');
 dual_residual = norm(MU*(Z-Z_prev),'fro');
 
 if mod(iteration,8)==0
     fit_loss = norm(Y*weight-W*X*weight-b*ones(size(X,2),1)','fro');
     strct_loss = sum(sqrt(sum(Z.^2,1)));
     trace_term = trace(W*Sxlx*W');
     orig_loss = 0.5*fit_loss^2 + lambda*strct_loss + 0.5*lambda2*norm(W,'fro')^2 + 0.5*lambda3*trace_term;
     fprintf('\n Orig_loss is %f, fit loss is %f, strct loss is %f, trace loss is %f, Priaml residual is %f, Dual residual is %f, iteration is %d\n', orig_loss, fit_loss, strct_loss, trace_term, primal_residual, dual_residual, iteration);
 end
 if ((primal_residual < EPSILON)&&(dual_residual < EPSILON))||(iteration > MAXIT)
     break;
 end
 
 if primal_residual>RHO*dual_residual
     MU = 2*MU;
%      fprintf('\n Mu is %f \n ', MU);
 elseif dual_residual>RHO*primal_residual
     MU = MU/2;
%      fprintf('\n Mu is %f \n ', MU);
 else
     MU = MU;
 end
 
end

end
