function [predTe] = semantic_mapping_nnn(vFeature, yTr, tagVector, trteindx, ParaM)
% tagVector = eye(size(tagVector,2));
%% 1. load parameter
beta1 = ParaM.beta1;
beta2 = ParaM.beta2;
beta3 = ParaM.beta3;
beta4 = ParaM.beta4;
beta5 = ParaM.beta5;
bsnum = ParaM.bs;
wlabel = ParaM.wlabel; 
kNN = ParaM.kNN;
[d N] = size(vFeature);
[dv, r] = size(tagVector);

%% 2. Init
% F
yTe = zeros(r,trteindx(3)-trteindx(2));
F = [yTr, yTe];
% Laplacian matrix L^(mv)
Sim = calSimilarity(vFeature,kNN);
[tt, pp] = sort(Sim,'descend');
Sim(find(Sim<repmat(tt(kNN,:),[N 1])))=0;
Sim(find((Sim-Sim')~=0))=0;
DiM = diag(sum(Sim,1));
LMV = DiM - Sim;
LMV = LMV + 1e-15*eye(N,N);
% Laplacian matrix L^(tag)
% Wpos = 1./max(1, sum(F>0, 2));
% h = (sum(bsxfun(@times, (F>0), Wpos), 1))';
% %     h_hat = ones(N,1);
% H = spdiags(h, 0, N, N);
SimF = F*F';
Fnorm = sqrt(sum((F).^2,2));
SimF = SimF./(repmat(Fnorm, [1 r]).*repmat(Fnorm', [r 1])+1e-20);
DimF = diag(sum(SimF,1));
LTAG = DimF - SimF;
LTAG = LTAG + 1e-15*eye(r,r);
% veta vt_y_0 vt_y_n1 vtau_y_n1
M = zeros(size(F));
veta = 2;
vt_y_0 = 1;
vt_y_n1 = 1;
vtau_y_n1 = 1e-2;
%threshhold
threshhold = 0.5;

%% 3. OPT
for i = 1:bsnum
    % H
    Wpos = 1./max(1, sum(F>0, 2));
    h = (sum(bsxfun(@times, (F>0), Wpos), 1))';
    h_hat = h;
    h_hat(trteindx(2):trteindx(3)-1) = 1;
%     h_hat = ones(N,1);
    H = spdiags(h, 0, N, N);
    H_HAT = spdiags(h_hat, 0, N, N);
    % init M Y_0 Y_N1
    M(find(F ~= 0)) = 1;
    Y_0 = zeros(size(F));
    Y_N1 = Y_0;
    VtimesV = vFeature*H*H'*vFeature';
    DtimesD = tagVector'*tagVector;
    for iter = 1:30
        % W
        if wlabel==1
            W_0 = (tagVector*Y_0*H*H'*vFeature')/(VtimesV + beta1*eye(d,d));
        end
        F_S_vtau_hat_G_W = 0.5*norm((tagVector*Y_0 - W_0*vFeature)*H,'fro')^2 + 0.5*beta1*norm(W_0,'fro')^2;
        
        % Y
        % 1.
        Y_HAT = Y_0 + ((vt_y_n1 - 1)/(vt_y_0))*(Y_0 - Y_N1);
        % 2.
        vtau_hat = vtau_y_n1;
        WtimesV = W_0*vFeature;
        nablaf_Y_hat = DtimesD*Y_HAT*H_HAT*H_HAT' - tagVector'*WtimesV*H_HAT*H_HAT' + beta3*M.*(Y_HAT - F) + beta4*Y_HAT*LMV + beta5*LTAG*Y_HAT;
        f_Y_HAT = 0.5*norm((tagVector*Y_HAT - WtimesV)*H_HAT,'fro')^2 + 0.5*beta3*norm(M.*(Y_HAT-F),'fro')^2 + 0.5*beta4*trace(Y_HAT*LMV*Y_HAT') + 0.5*beta5*trace(Y_HAT'*LTAG*Y_HAT);
        nablaf_Y_hat_norm = norm(nablaf_Y_hat,'fro')^2;
        while (1)
            G_Y = Y_HAT - nablaf_Y_hat/vtau_hat;
            S_vtau_hat_G_Y = sign(G_Y).* max(abs(G_Y)-beta2/vtau_hat,0);
            F_S_vtau_hat_G_Y = 0.5*norm((tagVector*S_vtau_hat_G_Y-WtimesV)*H_HAT,'fro')^2 + beta2*sum(sum(abs(S_vtau_hat_G_Y))) + 0.5*beta3*norm(M.*(S_vtau_hat_G_Y-F),'fro')^2 + 0.5*beta4*trace(S_vtau_hat_G_Y*LMV*S_vtau_hat_G_Y') + 0.5*beta5*trace(S_vtau_hat_G_Y'*LTAG*S_vtau_hat_G_Y);
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
    Y_temp = Y_0.*(1-M);
    [maxV,setIndx] = max(Y_temp,[],1);
%     setIndx(find(rand(1,N)>threshhold)) = 0;
%     for j = 1:length(setIndx)
%         if setIndx(j) ~= 0
%             F(setIndx(j),j) = 1;
%         end
%     end
    for j = 1:length(setIndx)
        if maxV(j) > threshhold
            F(setIndx(j),j) = 1;
        end
    end
end
predTe = Y_0(:,trteindx(2):trteindx(3)-1);
end