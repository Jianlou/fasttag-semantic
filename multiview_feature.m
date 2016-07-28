function [zTrain, zTest, xTeSem] = multiview_feature(xTr, xTe, yTr)

d = 3000;
theta1 = 1;
theta2 = 0.5;
lambda1 = 1e-2;
lambda2 = 1e-2;
% lambda3 = 1e-1;

xTrain = [];
xTest = [];
indx(1) = 1;
for i=1:length(xTr)
    xTrain = [xTrain;xTr{i}];
    xTest = [xTest;xTe{i}];
    if (i)<=length(xTr)
        indx(i+1) = indx(i) + size(xTr{i},1);
    end
end

load('/Users/lou/Desktop/corel5k_tagVector.mat');
for i=1:size(xTrain,2)
    xTrSem(:,i) = sum(tagVector(find(yTr(:,i)==1),:),1)';
end
xTrSem = xTrSem./(repmat(sqrt(sum(xTrSem.^2,1)),[size(xTrSem,1) 1])+1e-20);
xTrain = [xTrain;xTrSem];
indx(end+1) = indx(end) + size(xTrSem,1);

[D N] = size(xTrain);
D1 = size(xTrSem,1);
D2 = D - D1;

P = randn(D2,d);
[Q, ~] = qr(P,0);
P = Q;

% Wpos = 1./max(1, sum(yTr>0, 2));
% h = (sum(bsxfun(@times, (yTr>0), Wpos), 1))';
% H = spdiags(h, 0, N, N);

% init Z
Z = P'*xTrain(1:D2,:);
% ZZ = Z*H;

W1 = zeros(D1,d);
W2 = zeros(D2,d);
V1 = zeros(D1,d);
V2 = zeros(D2,d);
Q1 = zeros(D1,d);
Q2 = zeros(D2,d);
W = [W2;W1]; 
Q = [Q2;Q1];
V = [V2;V1];

RHO =10;
MU =1e-1;




for iter = 1:10

    % opt W
    W1 = (xTrain(indx(end-1):end,:)*Z' + MU*V1 - Q1)/(Z*Z' + MU*eye(d,d));
    W2 = (xTrain(1:indx(end-1)-1,:)*Z' + MU*V2 - Q2)/(Z*Z' + MU*eye(d,d));
    W = [W2;W1];
    
    % OPT V  
    for k=1:length(indx)-1
        UU(indx(k):indx(k+1)-1,:) = W(indx(k):indx(k+1)-1,:) + Q(indx(k):indx(k+1)-1,:)/MU;
        LAMBDA1 = lambda1/MU;
        UU_NORM = max(sqrt(sum(UU(indx(k):indx(k+1)-1,:).^2,1)) - LAMBDA1,zeros(1,size(UU(indx(k):indx(k+1)-1,:),2)))./(sqrt(sum(UU(indx(k):indx(k+1)-1,:).^2,1))+10e-20);
        V_prev(indx(k):indx(k+1)-1,:) = V(indx(k):indx(k+1)-1,:);
        V(indx(k):indx(k+1)-1,:) = repmat(UU_NORM,[size(UU(indx(k):indx(k+1)-1,:),1),1]).*UU(indx(k):indx(k+1)-1,:);
        clear UU_NORM;
    end
    V1 = V(D+1-D1:D,:);
    V2 = V(1:D-D1,:);
    
    % Q
    Q1 = Q1 + MU*(W1-V1);
    Q2 = Q2 + MU*(W2-V2);
    Q = [Q2;Q1];
    
    % ZZ Z
    Z = (theta1*W1'*W1 + theta2*W2'*W2 + lambda2*eye(d,d))\(theta1*W1'*xTrain(indx(end-1):end,:) + theta2*W2'*xTrain(1:indx(end-1)-1,:));
%     Z = (lambda3*ZZ*H')/(lambda2*eye(N,N) + lambda3*H*H');
    
    primal_residual = norm(W-V,'fro');
    dual_residual = norm(MU*(V-V_prev),'fro');
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

zTrain = Z;
zTest = (theta2*W2'*W2 + lambda2*eye(d,d))\(theta2*W2'*xTest(1:indx(end-1)-1,:));
xTeSem = W1 * zTest;


end