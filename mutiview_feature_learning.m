function [MultiM, Theta, vTrain, indx] = mutiview_feature_learning(xTr, yTr, ParaM)

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
partN = length(indx)-1;

%% init 
% 
P = randn(D,d);
[Q, ~] = qr(P,0);
P = Q;
V = P'*xTrain;
clear P Q;
% RR = randn(N,d);
% temp = xTrain*RR;
% [P, ~] = qr(temp,0);
% V = P'*xTrain;
% clear P;
Theta = (1/partN)*ones(partN,1);
U = zeros(D,d);
Q = U;
P = U;
MU = 1e-2;
RHO = 10;

%% optimize
for iter =1:30
    iter
    
    % opt U P Q
    for i=1:partN
        XI = xTrain(indx(i):indx(i+1)-1,:);
        UI = U(indx(i):indx(i+1)-1,:);
        PI = P(indx(i):indx(i+1)-1,:);
        QI = Q(indx(i):indx(i+1)-1,:);
        
        UI = (XI*V' + MU*PI - QI)/(V*V' + MU*eye(d,d));
        
        UU = UI + QI/MU;
        LAMBDA = alpha1/MU;
        UU_NORM = max(sqrt(sum(UU.^2,1)) - LAMBDA,zeros(1,size(UU,2)))./(sqrt(sum(UU.^2,1))+10e-20);
        P_prev(indx(i):indx(i+1)-1,:) = PI;
        PI = repmat(UU_NORM,[size(UU,1),1]).*UU;
        
        QI = QI + MU*(UI - PI);
        
        U(indx(i):indx(i+1)-1,:) = UI;
        P(indx(i):indx(i+1)-1,:) = PI;
        Q(indx(i):indx(i+1)-1,:) = QI;
        
        clear UU_NORM UU UI PI QI XI;
    end
    
    % OPT V
    XXXX = [];
    UUUU = [];
    for j=1:partN
       XXXX = [XXXX;sqrt(Theta(j))*xTrain(indx(j):indx(j+1)-1,:)]; 
       UUUU = [UUUU;sqrt(Theta(j))*U(indx(j):indx(j+1)-1,:)];
    end
    V = (UUUU'*UUUU + alpha2*eye(d,d))\(UUUU'*XXXX);
    
    % opt Theta
    for j=1:partN
        qq(j) = 0.5*norm(xTrain(indx(j):indx(j+1)-1,:) - U(indx(j):indx(j+1)-1,:)*V,'fro')^2 ...
                + alpha1*sum(sqrt(sum(U(indx(j):indx(j+1)-1,:).^2,1)));
    end
    Theta = quadprog(alpha3*eye(partN,partN),qq,[],[],ones(1,partN),1,zeros(partN,1),[]);
    
    % convergence
    primal_residual = norm(U-P,'fro');
    dual_residual = norm(MU*(P-P_prev),'fro');
     if primal_residual>RHO*dual_residual
         MU = 2*MU;
         fprintf('\n Mu is %f \n ', MU);
     elseif dual_residual>RHO*primal_residual
         MU = MU/2;
         fprintf('\n Mu is %f \n ', MU);
     else
         MU = MU;
     end
     
     if mod(iter,5)==0
        loss_part1 = 0.5*norm(XXXX - UUUU*V,'fro')^2;
        loss_part2 = 0;
        for j=1:partN
            loss_part2 = loss_part2 + alpha1*Theta(j)*sum(sqrt(sum(U(indx(j):indx(j+1)-1,:).^2,1)));
        end
        loss_part3 = 0.5*alpha2*norm(V,'fro')^2;
        loss_part4 = 0.5*alpha3*norm(Theta)^2;
        loss = loss_part1 + loss_part2 + loss_part3 + loss_part4
     end

end
MultiM = U;
vTrain = V;
end