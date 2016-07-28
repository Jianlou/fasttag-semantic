function [W,b,Zte] = linear_str_analy(xTr, yTr, xTe, alpha1, alpha2, beta1, lambda1, lambda2, lambda3, d)

X = [];
XTE = [];
for i=1:length(xTr)
    X = [X;xTr{i}];
    XTE = [XTE;xTe{i}];
    dv(i) = size(xTr{i},1);
end
Y = yTr;
r = size(Y,1);
[D,N] = size(X);

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
Z = P'*X;
W = zeros(r, d);
U = zeros(D,d);
MU = 1e-2;

YmultiA = Y*A;
AmultiA = A*A';
%% optimize
for i=1:5

% ZZ

ZZ = (W' * W + lambda2 * eye(d,d))\(W' * YmultiA + lambda2 * Z * A);

% V

V = (lambda1 * U' * U + (lambda1 * alpha2 + lambda3) * eye(d,d))\(lambda1 * U' * X + lambda3 * Z);

% W

W = (YmultiA * ZZ')/(ZZ * ZZ' + beta1 * eye(d,d));
b = ((Y-W*Z)*g)/(h'*h);

% U
dv_1=[1,dv(1:length(xTr)-1)];
iStart = 0;
iEnd = 0;
for idx=1:length(xTr)       
    iStart = iStart + dv_1(idx);
    iEnd = iEnd + dv(idx);
    
    S = zeros(dv(idx),d);
    O = S;
    Xtmp = X(iStart:iEnd,:);
    
    for iter=1:20
        % U
        U(iStart:iEnd,:) = (Xtmp*V' + MU*S - O)/(V*V' + MU*eye(d,d));
        
        % S
         UU = U(iStart:iEnd,:) + O/MU;
         LAMBDA1 = alpha1/MU;
         UU_NORM = max(sqrt(sum(UU.^2,1)) - LAMBDA1,zeros(1,size(UU,2)))./(sqrt(sum(UU.^2,1))+10e-20);
         S_prev = S;
         S = repmat(UU_NORM,[size(UU,1),1]).*UU;
         
         % O
         O = O + MU*(U(iStart:iEnd,:)-S);
         
        primal_residual = norm(U(iStart:iEnd,:)-S,'fro');
        dual_residual = norm(MU*(S-S_prev),'fro');
        if primal_residual>10*dual_residual
            MU = 2*MU;
            fprintf('\n Mu is %f \n ', MU);
        elseif dual_residual>10*primal_residual
            MU = MU/2;
            fprintf('\n Mu is %f \n ', MU);
        else
            MU = MU;
        end   
    end
    clear UU UU_NORM S_prev S O Xtmp
end
% U = (X * V')/(V * V' + alpha1 * eye(d,d));

% Z

Z = (lambda2 * ZZ * A' + lambda3 * V)/(lambda2 * AmultiA + lambda3 * eye(N,N));

% loss

if mod(i,5)==0
    
    loss1 = norm(YmultiA - W * ZZ, 'fro')^2

    loss2 = loss1 + beta1 * norm(W, 'fro')^2 + lambda1 * (norm(X - U * V, 'fro')^2 + alpha1 * norm(U, 'fro')^2 + alpha2 * norm(V,'fro')^2) + lambda2 *norm(ZZ-Z*A,'fro')^2 + lambda3 * norm(V-Z,'fro')^2
    
end


end


Zte = (U' * U + alpha2 * eye(d,d))\(U' * XTE);

end