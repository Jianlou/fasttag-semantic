function [W,b] = group_multitask_analy(X, Y, lambda1, lambda2,Idx)

[d n] = size(X);
r = size(Y,1);

Wpos = 1./max(1, sum(Y>0, 2));
% Wpos = ones(r,1);
h = (sum(bsxfun(@times, (Y>0), Wpos), 1))';
H = spdiags(h, 0, n, n);
g = h.*h;
G = H.*H;

A = H - (g*h')/(h'*h);

MU = 1e-2;
RHO = 10;
W = zeros(r,d);
Z = zeros(r,d);
Q = zeros(r,d);

Y_A_A_X = Y*A*A'*X';
X_A_A_X = X*A*A'*X';

for iter=1:80
    % OPTIMIZE W
    W = (Y_A_A_X + MU*Z -Q)/(X_A_A_X + lambda2*eye(d,d) + MU*eye(d,d));
    
    % optimize Z
 for k=1:size(Idx,2)
    UU(Idx(1,k):Idx(2,k),:) = W(Idx(1,k):Idx(2,k),:) + Q(Idx(1,k):Idx(2,k),:)/MU;
     LAMBDA1 = lambda1/MU;
     UU_NORM = max(sqrt(sum(UU(Idx(1,k):Idx(2,k),:).^2,1)) - LAMBDA1,zeros(1,size(UU(Idx(1,k):Idx(2,k),:),2)))./(sqrt(sum(UU(Idx(1,k):Idx(2,k),:).^2,1))+10e-20);
     Z_prev(Idx(1,k):Idx(2,k),:) = Z(Idx(1,k):Idx(2,k),:);
     Z(Idx(1,k):Idx(2,k),:) = repmat(UU_NORM,[size(UU(Idx(1,k):Idx(2,k),:),1),1]).*UU(Idx(1,k):Idx(2,k),:);
     clear UU_NORM;
 end
    
    % OPTIMIZE Q
    Q = Q + MU*(W-Z);
    
    if mod(iter,5)==0
        fit = 0.5*norm(Y*A-W*X*A,'fro')^2;
        loss1 = lambda1*sum(sqrt(sum(Z.^2,1)));
        fprintf('\n fit loss is %f, rank is %f, iteration is %d\n',fit,loss1,iter);
    end

    primal_residual = norm(W-Z,'fro');
    dual_residual = norm(MU*(Z-Z_prev),'fro');
     if primal_residual>RHO*dual_residual
         MU = 2*MU;
         fprintf('\n Mu is %f \n ', MU);
     elseif dual_residual>RHO*primal_residual
         MU = MU/2;
         fprintf('\n Mu is %f \n ', MU);
     else
         MU = MU;
     end
    
end

b = (Y*g-W*X*g)/(h'*h);

end