function [MultiM, vTrain, indx, normP] = mutiview_feature_learning_n(xTr, yTr, ParaM)

%% load feature
xTrain = [];
indx(1) = 1;
for i=1:length(xTr)
    xTrain = [xTrain;xTr{i}];
    if (i)<=length(xTr)
        indx(i+1) = indx(i) + size(xTr{i},1);
    end
end

%% load param
[D N] = size(xTrain);
[R] = size(yTr,1);
alpha1 = ParaM.alpha1;
alpha2 = ParaM.alpha2;
alpha3 = ParaM.alpha3;
d = ParaM.d;
kNN = ParaM.kNN;
partN = length(indx)-1;

%% init 
% 
Xmean = mean(xTrain,2);
% xTrain = xTrain - repmat(Xmean,[1 N]);
Xstd = std(xTrain')' + 1e-20;
% xTrain = xTrain./repmat(Xstd, [1 N]);
% [U1,S1,V1] = svd(cov(xTrain'));
% P = U1(:,1:d);
% V = P'*xTrain;
% clear P;
normP.Xmean = Xmean;
normP.Xstd = Xstd;
RR = randn(N,d);
temp = xTrain*RR;
[P, ~] = qr(temp,0);
V = P'*xTrain;
%V = randn(d,N);
clear P;
Z = V;
Q2 = zeros(size(V));

U = zeros(D,d);
P = U;
Q1 = zeros(size(U));

MU1 = 1e-2;
MU2 = 1e-2;
RHO = 10;

%% calculate laplacian matrix
for i = 1:partN
    [Similarity{i}] = calSimilarity(xTr{i},kNN);
    Dimilarity{i} = diag(sum(Similarity{i},1));
    LaplacianM{i} = Dimilarity{i}-Similarity{i};
end
Similarity{partN+1} = yTr'*yTr;
Ynorm = sqrt(sum(yTr.^2,1));
Similarity{partN+1} = Similarity{partN+1}./(repmat(Ynorm,[N 1]).*repmat(Ynorm', [1 N])+1e-20);
[tt,pp] = sort(Similarity{partN+1},'descend');
Similarity{partN+1}(find(Similarity{partN+1}<repmat(tt(kNN,:),[N 1]))) = 0;
Dimilarity{partN+1} = diag(sum(Similarity{partN+1},1));
LaplacianM{partN+1} = (Dimilarity{partN+1}-Similarity{partN+1});
L = zeros(N,N);
for i = 1:partN+1
    L = L + LaplacianM{i};
end
    
%% optimize
for iter =1:30
    iter
    
    % opt U P
    U = (xTrain*V' + MU1*P - Q1)/(V*V' + MU1*eye(d,d));
    for i=1:partN
        UI = U(indx(i):indx(i+1)-1,:);
        PI = P(indx(i):indx(i+1)-1,:);
        QI = Q1(indx(i):indx(i+1)-1,:);
        
        UU = UI + QI/MU1;
        LAMBDA = alpha1/MU1;
        UU_NORM = max(sqrt(sum(UU.^2,1)) - LAMBDA,zeros(1,size(UU,2)))./(sqrt(sum(UU.^2,1))+10e-20);
        P_prev(indx(i):indx(i+1)-1,:) = PI;
        PI = repmat(UU_NORM,[size(UU,1),1]).*UU;
                
        P(indx(i):indx(i+1)-1,:) = PI;
        
        clear UU_NORM UU UI PI QI;
    end
    
    % OPT V Z
    V = (U'*U + (alpha2 + MU2)*eye(d,d))\(U'*xTrain + MU2*Z - Q2);
    Z_prev = Z;
    Z = (Q2 + MU2*V)/(alpha3*L + MU2*eye(N,N));
    
    % opt Q1 Q2
    Q1 = Q1 + MU1*(U - P);
    Q2 = Q2 + MU2*(V - Z);
    
     if mod(iter,5)==0
        loss_part1 = 0.5*norm(xTrain - U*V,'fro')^2;
        loss_part2 = 0;
        for j=1:partN
            loss_part2 = loss_part2 + alpha1*sum(sqrt(sum(P(indx(j):indx(j+1)-1,:).^2,1)));
        end
        loss_part3 = 0.5*alpha2*norm(V,'fro')^2;
        loss_part4 = 0.5*alpha3*trace(Z*L*Z');
        loss = loss_part1 + loss_part2 + loss_part3 + loss_part4
     end
    
    % convergence
    primal_residual = norm(U-P,'fro');
    dual_residual = norm(MU1*(P-P_prev),'fro');
     if primal_residual>RHO*dual_residual
         MU1 = 2*MU1;
         fprintf('\n Mu1 is %f \n ', MU1);
     elseif dual_residual>RHO*primal_residual
         MU1 = MU1/2;
         fprintf('\n Mu1 is %f \n ', MU1);
     else
         MU1 = MU1;
     end
    primal_residual = norm(V-Z,'fro');
    dual_residual = norm(MU2*(Z-Z_prev),'fro');
     if primal_residual>RHO*dual_residual
         MU2 = 2*MU2;
         fprintf('\n Mu2 is %f \n ', MU2);
     elseif dual_residual>RHO*primal_residual
         MU2 = MU2/2;
         fprintf('\n Mu2 is %f \n ', MU2);
     else
         MU2 = MU2;
     end

end
MultiM = U;
vTrain = V;
end