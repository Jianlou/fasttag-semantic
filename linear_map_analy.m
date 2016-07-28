function [W,b,P] = linear_map_analy(X, Y, lambda1, lambda2,lambda3)

d = 3000;
[D N] = size(X);
r = size(Y,1);

Wpos = 1./max(1, sum(Y>0, 2));
% Wpos = ones(r,1);
h = (sum(bsxfun(@times, (Y>0), Wpos), 1))';
H = spdiags(h, 0, N, N);
g = h.*h;
G = H.*H;
A = H - (g*h')/(h'*h);

%% initialize 
P = randn(D,d);
[Q, ~] = qr(P,0);
P = Q;
ZZ = P'*X*A;
ZZ = zeros(size(ZZ));
W = zeros(r,d);
iter = 0;

while 1


%% optimize ZZ
ZZ = (W'*W+lambda2*eye(d,d))\((W'*Y + lambda2*P'*X)*A);
    
%% optimize P
P = (lambda2 * X*A*A'*X' + lambda3 * eye(D,D))\(lambda2 * X*A*ZZ');

iter =iter +1;
%% optimize W,b
W = (Y*A*ZZ')/(ZZ*ZZ' + lambda1*eye(d,d));
b = ((Y-W*(P'*X))*g)/(h'*h);

loss1 = 0.5*norm(Y*A-W*ZZ,'fro')^2
loss2 = loss1 + 0.5*lambda1*norm(W,'fro')^2 + 0.5*lambda2*norm(ZZ-P'*X*A,'fro')^2 + 0.5*lambda3*norm(P,'fro')^2


if iter > 15
    break;
end

end

end