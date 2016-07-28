function [ W ,b] = group_multitask_admm(X, Y, lambda, Idx)
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
MU = 1e-1;

RHO = 10;
EPSILON = 1e-3;
MAXIT = 80;
iteration = 0;

%%

Wpos = 1./max(1, sum(Y>0, 2));
weights = (sum(bsxfun(@times, (Y>0), Wpos), 1));
weight = spdiags(weights', 0, n, n);
temp = weight*(eye(n,n)-(1/n)*ones(n,n));

Sx = X*temp*temp'*X';

for k=1:size(Idx,2)

MU = 1e-2; 
iteration = 0;

% Wpos = 1./max(1, sum(Y(Idx(1,k):Idx(2,k),:)>0, 2));
% weights = max(bsxfun(@times, (Y(Idx(1,k):Idx(2,k),:)>0), Wpos), [], 1).^1;
% Sx = X*spdiags(weights', 0, n, n)*X';  
Sxy = Y(Idx(1,k):Idx(2,k),:)*temp*temp'*X';

while true
 iteration = iteration +1;
 W(Idx(1,k):Idx(2,k),:) = (Sxy - Q(Idx(1,k):Idx(2,k),:) + MU*Z(Idx(1,k):Idx(2,k),:))/(Sx + 0*I + MU*I);
 b(Idx(1,k):Idx(2,k),:) = (1/n)*(Y(Idx(1,k):Idx(2,k),:)-W(Idx(1,k):Idx(2,k),:)*X)*weight*ones(size(X,2),1);
 U = W(Idx(1,k):Idx(2,k),:) + Q(Idx(1,k):Idx(2,k),:)/MU;
 LAMBDA1 = lambda/MU;
 U_NORM = max(sqrt(sum(U.^2,1)) - LAMBDA1,zeros(1,size(U,2)))./(sqrt(sum(U.^2,1))+10e-20);
 Z_prev = Z(Idx(1,k):Idx(2,k),:);
 Z(Idx(1,k):Idx(2,k),:) = repmat(U_NORM,[size(U,1),1]).*U;
 Q(Idx(1,k):Idx(2,k),:) = Q(Idx(1,k):Idx(2,k),:) + MU*(W(Idx(1,k):Idx(2,k),:)-Z(Idx(1,k):Idx(2,k),:));
 
 primal_residual = norm(W(Idx(1,k):Idx(2,k),:)-Z(Idx(1,k):Idx(2,k),:),'fro');
 dual_residual = norm(MU*(Z(Idx(1,k):Idx(2,k),:)-Z_prev),'fro');
 
 if mod(iteration,8)==0
     fit_loss = norm(Y(Idx(1,k):Idx(2,k),:)*weight-W(Idx(1,k):Idx(2,k),:)*X*weight-b(Idx(1,k):Idx(2,k),:)*ones(size(X,2),1)','fro');
     strct_loss = sum(sqrt(sum(Z(Idx(1,k):Idx(2,k),:).^2,1)));
     orig_loss = 0.5*fit_loss^2 + lambda*strct_loss;
     fprintf('\n Orig_loss is %f, fit loss is %f, strct loss is %f, Priaml residual is %f, Dual residual is %f, iteration is %d\n', orig_loss, fit_loss, strct_loss, primal_residual, dual_residual, iteration);
 end
 if ((primal_residual < EPSILON)&&(dual_residual < EPSILON))||(iteration > MAXIT)
     break;
     clear U U_NORM Z_prev
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

end
