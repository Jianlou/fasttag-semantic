function [W,b] = lowrank_analy(X, Y, lambda1, lambda2)

[d n] = size(X);
r = size(Y,1);

Wpos = 1./max(1, sum(Y>0, 2));
% Wpos = ones(r,1);
h = (sum(bsxfun(@times, (Y>0), Wpos), 1))';
H = spdiags(h, 0, n, n);
g = h.*h;
G = H.*H;

A = H - (g*h')/(h'*h);

MU = 1e-3;
RHO = 10;
W = zeros(r,d);
Z = zeros(r,d);
Q = zeros(r,d);

Y_A_A_X = Y*A*A'*X';
X_A_A_X = X*A*A'*X';

for iter=1:30
    % OPTIMIZE W
    W = (Y_A_A_X + MU*Z -Q)/(X_A_A_X + lambda2*eye(d,d) + MU*eye(d,d));
    
    % optimize Z
    UU = W + Q/MU;
    [U,S,V] = svd(UU);
    S = sign(S).*max(abs(S)-lambda1/MU,0);
    Z_prev = Z;
    Z = U*S*V';
    
    % OPTIMIZE Q
    Q = Q + MU*(W-Z);
    
    if mod(iter,5)==0
        fit = 0.5*norm(Y*A-W*X*A,'fro')^2;
        rank1 = rank(Z);
        fprintf('\n fit loss is %f, rank is %f, iteration is %d\n',fit,rank1,iter);
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
