function [Y] = semantic_mapping_test(vTest, yTr, W, tagVector, ParaM)

%% 1. load parameter
beta2 = ParaM.beta2;
beta4 = ParaM.beta4;
beta5 = ParaM.beta5;
kNN = ParaM.kNN;
[d N] = size(vTest);
[dv, r] = size(tagVector);

%% 2. Init
% Laplacian matrix L^(mv)
Sim = calSimilarity(vTest,kNN);
DiM = diag(sum(Sim,1));
LMV = DiM - Sim;
LMV = LMV + 1e-10 * eye(N);
% Laplacian matrix L^(tag)
SimF = yTr*yTr';
Fnorm = sqrt(sum(yTr.^2,2));
SimF = SimF./(repmat(Fnorm, [1 r]).*repmat(Fnorm', [r 1])+1e-20);
DimF = diag(sum(SimF,1));
LTAG = DimF - SimF;
LTAG = LTAG + 1e-10 * eye(r);
% veta vt_y_0 vt_y_n1 vtau_y_n1
veta = 2;
vt_y_0 = 1;
vt_y_n1 = 1;
vtau_y_n1 = 1e-2;

%% 3. OPT
% init M Y_0 Y_N1
    W_0 = W;
    Y_0 = zeros(r,N);
    Y_N1 = Y_0;
    DtimesD = tagVector'*tagVector;
    for iter = 1:30
        
        % Y
        % 1.
        Y_HAT = Y_0 + ((vt_y_n1 - 1)/(vt_y_0))*(Y_0 - Y_N1);
        % 2.
        vtau_hat = vtau_y_n1;
        WtimesV = W_0*vTest;
        nablaf_Y_hat = DtimesD*Y_HAT - tagVector'*WtimesV + beta4*Y_HAT*LMV + beta5*LTAG*Y_HAT;
        f_Y_HAT = 0.5*norm(tagVector*Y_HAT - WtimesV,'fro')^2 + 0.5*beta4*trace(Y_HAT*LMV*Y_HAT') + 0.5*beta5*trace(Y_HAT'*LTAG*Y_HAT);
        nablaf_Y_hat_norm = norm(nablaf_Y_hat,'fro')^2;
        while (1)
            G_Y = Y_HAT - nablaf_Y_hat/vtau_hat;
            S_vtau_hat_G_Y = sign(G_Y).* max(abs(G_Y)-beta2/vtau_hat,0);
            F_S_vtau_hat_G_Y = 0.5*norm(tagVector*S_vtau_hat_G_Y-WtimesV,'fro')^2 + beta2*sum(sum(abs(S_vtau_hat_G_Y)))  + 0.5*beta4*trace(S_vtau_hat_G_Y*LMV*S_vtau_hat_G_Y') + 0.5*beta5*trace(S_vtau_hat_G_Y'*LTAG*S_vtau_hat_G_Y);
            Q_S_vtau_hat_G_Y = 0.5 * vtau_hat * norm(S_vtau_hat_G_Y - G_Y,'fro')^2 + beta2*sum(sum(abs(S_vtau_hat_G_Y))) + f_Y_HAT - 1/(2*vtau_hat) * nablaf_Y_hat_norm;
            if F_S_vtau_hat_G_Y <= Q_S_vtau_hat_G_Y
                vtau_y_0 = vtau_hat;
                break;
            else
               vtau_hat =  veta*vtau_hat;
            end     
        end
        vtau_y_n1 = vtau_y_0;
        % 3.
        Y_1 = S_vtau_hat_G_Y;
        Y_N1 = Y_0;
        Y_0 = Y_1;
        % 4.
        vt_y_1 = (1+sqrt(1+4*vt_y_0^2))/(2);
        vt_y_n1 = vt_y_0;
        vt_y_0 = vt_y_1;
        F_S_vtau_hat_G_Y;
        
    end
    
    Y = Y_0;
end