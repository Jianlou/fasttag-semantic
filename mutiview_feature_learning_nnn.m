function [MultiM, vFeature, indx, normP, trteindx] = mutiview_feature_learning_nnn(xTr, yTr, xTe, ParaM)

%% 1. load feature
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
XX = [xTrain, xTest];
trteindx(1) = 1;
trteindx(2) = trteindx(1)+size(xTrain,2);
trteindx(3) = trteindx(2) + size(xTest,2);
clear xTrain xTest;

%% 2. load param
[D N] = size(XX);
[r] = size(yTr,1);
alpha1 = ParaM.alpha1;
alpha2 = ParaM.alpha2;
alpha3 = ParaM.alpha3;
d = ParaM.d;
kNN = ParaM.kNN;
typeProj = ParaM.typeProj;
typeLap = ParaM.typeLap;
partN = length(indx)-1;
yTe = zeros(r,trteindx(3) - trteindx(2));

%% 3. calculate laplacian matrix L
for i = 1:partN
    [Similarity{i}] = calSimilarity([xTr{i}, xTe{i}],kNN);
end
Similarity{partN+1} = [yTr, yTe]'*[yTr, yTe];
Ynorm = sqrt(sum([yTr, yTe].^2,1));
Similarity{partN+1} = Similarity{partN+1}./(repmat(Ynorm,[N 1]).*repmat(Ynorm', [1 N])+1e-20);
if typeLap == 1
    for i = 1:partN
        Similarity{i} = Similarity{i}.*Similarity{partN+1};
        [tt,pp] = sort(Similarity{i},'descend');
        Similarity{i}(find(Similarity{i}<repmat(tt(kNN,:),[N 1]))) = 0;
        Similarity{i}(find((Similarity{i}-Similarity{i}')~=0))=0;
    end 
    [tt,pp] = sort(Similarity{partN+1},'descend');
    Similarity{partN+1}(find(Similarity{partN+1}<repmat(tt(kNN,:),[N 1]))) = 0;
    Similarity{partN+1}(find((Similarity{partN+1}-Similarity{partN+1}')~=0))=0;
end
for i = 1:partN+1
    Dimilarity{i} = diag(sum(Similarity{i},1));
    LaplacianM{i} = (Dimilarity{i}-Similarity{i});
end
L = zeros(N,N);
weightL = ones(partN+1,1)/(partN+1);
weightL(partN+1) = 1;
for i = 1:partN+1
    L = L + weightL(i)*LaplacianM{i};
end
L = L + 1e-15*eye(N,N);

%% 4. init 
% V_0 V_N1
if typeProj==1
    Xmean = mean(XX,2);
    Xstd = std(XX')' + 1e-20;
    normP.Xmean = Xmean;
    normP.Xstd = Xstd;
    RR = randn(N,d);
    temp = XX*RR;
    [P, ~] = qr(temp,0);
    V_0 = P'*XX;
    clear P temp RR;
    V_N1 = V_0;
elseif typeProj==2
    Xmean = mean(XX,2);
    Xstd = std(XX')' + 1e-20;
    normP.Xmean = Xmean;
    normP.Xstd = Xstd;
    XX = XX - repmat(normP.Xmean,[1 N]);
    XX = XX./repmat(normP.Xstd, [1 N]);
    [UUU,SSS,VVV] = svd(XX);
    P = UUU(:,1:d);
    V_0 = P'*XX;
    clear P UUU SSS VVV;
    V_N1 = V_0;
end
% xTrain = xTrain - repmat(Xmean,[1 N]);
% xTrain = xTrain./repmat(Xstd, [1 N]);
% [U1,S1,V1] = svd(cov(xTrain'));
% P = U1(:,1:d);
% V = P'*xTrain;
% clear P;
%V = randn(d,N);

% U_0 U_N1
U_0 = zeros(D,d);
U_N1 = U_0;

% veta vt_u_0 vt_u_n1 vt_v_0 vt_v_n1 vtau_u_n1 vtau_v_n1
veta = 2;
vt_u_0 = 1;
vt_u_n1 = 1;
vt_v_0 = 1;
vt_v_n1 = 1;
vtau_u_n1 = 1e-2;
vtau_v_n1 = 1e-2;

%% 5. optimize
for iter =0:30 
    % opt U
    % 1.
    U_HAT = U_0 + ((vt_u_n1 - 1)/(vt_u_0))*(U_0 - U_N1);
    % 2.
    vtau_hat = vtau_u_n1;
    nablaf_U_hat = U_HAT * V_0 * V_0' - XX * V_0';
    f_U_HAT = 0.5*norm(XX - U_HAT*V_0,'fro')^2;
    nablaf_U_hat_norm = norm(nablaf_U_hat,'fro')^2;
    while (1)
        G_U = U_HAT - nablaf_U_hat/vtau_hat;
        S_vtau_hat_G_U_21 = 0;
        for i=1:partN
            G_U_norm(i,:) = sqrt(sum(G_U(indx(i):indx(i+1)-1,:).^2,1));
            S_vtau_hat_G_U(indx(i):indx(i+1)-1,:) = repmat(max((G_U_norm(i,:) - alpha1/vtau_hat)./G_U_norm(i,:),0),[indx(i+1)-indx(i),1]).*G_U(indx(i):indx(i+1)-1,:);
            S_vtau_hat_G_U_21 = S_vtau_hat_G_U_21 + sum(sqrt(sum(S_vtau_hat_G_U(indx(i):indx(i+1)-1,:).^2,1)));
        end
        F_S_vtau_hat_G_U = 0.5*norm(XX - S_vtau_hat_G_U*V_0,'fro')^2 + alpha1*S_vtau_hat_G_U_21;
        Q_S_vtau_hat_G_U = 0.5 * vtau_hat * norm(S_vtau_hat_G_U - G_U,'fro')^2 + alpha1*S_vtau_hat_G_U_21 + f_U_HAT - 1/(2*vtau_hat) * nablaf_U_hat_norm;
        if F_S_vtau_hat_G_U <= Q_S_vtau_hat_G_U
            vtau_u_0 = vtau_hat;
            break;
        else
           vtau_hat =  veta*vtau_hat;
        end     
    end
    vtau_u_n1 = vtau_u_0;
    % 3.
    U_1 = S_vtau_hat_G_U;
    U_N1 = U_0;
    U_0 = U_1;
    % 4.
    vt_u_1 = (1+sqrt(1+4*vt_u_0^2))/(2);
    vt_u_n1 = vt_u_0;
    vt_u_0 = vt_u_1;
    F_S_vtau_hat_G_U
    
    % V
    % 1.
    V_HAT = V_0 + ((vt_v_n1 - 1)/(vt_v_0))*(V_0 - V_N1);
    % 2.
    vtau_hat = vtau_v_n1;
    nablaf_V_hat = -U_0' * XX +  U_0' * U_0 * V_HAT + alpha2 * V_HAT + alpha3 * V_HAT * L;
    f_V_HAT = 0.5*norm(XX - U_0*V_HAT,'fro')^2 + 0.5*alpha2*norm(V_HAT,'fro')^2 + 0.5*alpha3*trace(V_HAT*L*V_HAT');
    nablaf_V_hat_norm = norm(nablaf_V_hat,'fro')^2;
    while (1)
        G_V = V_HAT - nablaf_V_hat/vtau_hat;
        S_vtau_hat_G_V = G_V;
        F_S_vtau_hat_G_V = 0.5*norm(XX - U_0*S_vtau_hat_G_V,'fro')^2 + 0.5*alpha2*norm(S_vtau_hat_G_V,'fro')^2 + 0.5*alpha3*trace(S_vtau_hat_G_V*L*S_vtau_hat_G_V');
        Q_S_vtau_hat_G_V = 0.5 * vtau_hat * norm(S_vtau_hat_G_V - G_V,'fro')^2 + f_V_HAT - 1/(2*vtau_hat) * nablaf_V_hat_norm;
        if F_S_vtau_hat_G_V <= Q_S_vtau_hat_G_V
            vtau_v_0 = vtau_hat;
            break;
        else
           vtau_hat =  veta*vtau_hat;
        end     
    end
    vtau_v_n1 = vtau_v_0;
    % 3.
    V_1 = S_vtau_hat_G_V;
    V_N1 = V_0;
    V_0 = V_1;
    % 4.
    vt_v_1 = (1+sqrt(1+4*vt_v_0^2))/(2);
    vt_v_n1 = vt_v_0;
    vt_v_0 = vt_v_1;
    F_S_vtau_hat_G_V
    
    
end
MultiM = U_0;
vFeature = V_0;
end