function [ W, b,P] = multitask_admm(X, Y, lambda1, lambda2, lambda3)
%MULTITASK_ADMM Summary of this function goes here
%   Detailed explanation goes here

d = 3000;

[D N] = size(X);
r = size(Y,1);

%% Initialize
W = zeros(r, d);
U = zeros(r, d);

P = randn(D,d);
[P1, ~] = qr(P,0);
P = P1;

ZZ = zeros(r, d);
% Z = W;
Q = zeros(r, d);

MU = 1e-2;

RHO = 10;
EPSILON = 1e-3;
MAXIT = 80;
iteration = 0;

Wpos = 1./max(1, sum(Y>0, 2));
% Wpos = ones(r,1);
weights = (sum(bsxfun(@times, (Y>0), Wpos), 1));
h = weights';
H = spdiags(h, 0, N, N);
g = h.*h;
G = H.*H;
A = H - (g*h')/(h'*h);

YA = Y*A;
XA =X*A;
XAAX = X*A*A'*X';

while true
 iteration = iteration +1;
 %% ZZ
 ZZ = (W'*W+lambda2*eye(d,d))\(W'*YA+lambda2*P'*XA);
 
 %% P
 P = (lambda2*XAAX + lambda3*eye(D,D))\(lambda2*XA*ZZ');
 
 %% W b 
 W = (YA*ZZ' - Q + MU*U)/(ZZ*ZZ' + MU*eye(d,d));
 b = ((Y-W*(P'*X))*g)/(h'*h);
 
 %% U
 UU = W + Q/MU;
 LAMBDA1 = lambda1/MU;
 UU_NORM = max(sqrt(sum(UU.^2,1)) - LAMBDA1,zeros(1,size(UU,2)))./(sqrt(sum(UU.^2,1))+10e-20);
 U_prev = U;
 U = repmat(UU_NORM,[size(UU,1),1]).*UU;
 
 %% Q
 Q = Q + MU*(W-U);
 
primal_residual = norm(W-U,'fro');
dual_residual = norm(MU*(U-U_prev),'fro');
 if mod(iteration,8)==0
     
    loss1 = 0.5*norm(YA-W*ZZ,'fro')^2
    loss2 = loss1 + lambda1*sum(sqrt(sum(U.^2,1))) + 0.5*lambda2*norm(ZZ-P'*XA,'fro')^2 + 0.5*lambda3*norm(P,'fro')^2

     fprintf('\n Orig_loss is %f, fit loss is %f, Priaml residual is %f, Dual residual is %f, iteration is %d\n', loss1, loss2, primal_residual, dual_residual, iteration);
 end
 
 if ((primal_residual < EPSILON)&&(dual_residual < EPSILON))||(iteration > MAXIT)
     break;
 end
 
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



end

