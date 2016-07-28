function [vTest] = mutiview_feature_learning_test(xTe, MultiM, indx, normP, ParaM)

%% 1. load feature
xTest = [];
for i=1:length(xTe)
    xTest = [xTest;xTe{i}];
end

%% 2. load param
[D N] = size(xTest);
alpha2 = ParaM.alpha2;
alpha3 = ParaM.alpha3;
d = size(MultiM,2);
kNN = ParaM.kNN;
typeProj = ParaM.typeProj;
partN = length(indx)-1;

%% 3. calculate laplacian matrix L
for i = 1:partN
    [Similarity{i}] = calSimilarity(xTe{i},kNN);
end
for i = 1:partN
    Dimilarity{i} = diag(sum(Similarity{i},1));
    LaplacianM{i} = (Dimilarity{i}-Similarity{i});
    LaplacianM{i} = LaplacianM{i} + 1e-10 * eye(N);
end
L = zeros(N,N);
weightL = ones(partN,1)/(partN);
for i = 1:partN
    L = L + weightL(i)*LaplacianM{i};
end

%% init 
% V_0 V_N1
if typeProj==1
    RR = randn(N,d);
    temp = xTest*RR;
    [P, ~] = qr(temp,0);
    V_0 = P'*xTest;
    clear P temp RR;
    V_N1 = V_0;
end

% U
U_0 = MultiM;

% veta vt_v_0 vt_v_n1 vtau_v_n1
veta = 2;
vt_v_0 = 1;
vt_v_n1 = 1;
vtau_v_n1 = 1e-2;

    
%% 4. optimize
for iter =1:30    
    % V
    % 1.
    V_HAT = V_0 + ((vt_v_n1 - 1)/(vt_v_0))*(V_0 - V_N1);
    % 2.
    vtau_hat = vtau_v_n1;
    nablaf_V_hat = -U_0' * xTest +  U_0' * U_0 * V_HAT + alpha2 * V_HAT + alpha3 * V_HAT * L;
    f_V_HAT = 0.5*norm(xTest - U_0*V_HAT,'fro')^2 + 0.5*alpha2*norm(V_HAT,'fro')^2 + 0.5*alpha3*trace(V_HAT*L*V_HAT');
    nablaf_V_hat_norm = norm(nablaf_V_hat,'fro')^2;
    while (1)
        G_V = V_HAT - nablaf_V_hat/vtau_hat;
        S_vtau_hat_G_V = G_V;
        F_S_vtau_hat_G_V = 0.5*norm(xTest - U_0*S_vtau_hat_G_V,'fro')^2 + 0.5*alpha2*norm(S_vtau_hat_G_V,'fro')^2 + 0.5*alpha3*trace(S_vtau_hat_G_V*L*S_vtau_hat_G_V');
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
    F_S_vtau_hat_G_V;
end
vTest = V_0;
end