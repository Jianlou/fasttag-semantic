function [predTe] = tag_inference_nn(xTe, MultiM, indx, W_mvsmp, tagVector, ParaM, yTr, normP)

%% load parameter
alpha2 = ParaM.alpha2;
alpha3 = ParaM.alpha3;
beta2 = ParaM.beta2;
beta3 = ParaM.beta3;
beta4 = ParaM.beta4;
beta5 = ParaM.beta5;
kNN = ParaM.kNN;
typeProj = ParaM.typeProj;

[N] = size(xTe{1},2);
[d] = size(MultiM,2);
partN = length(xTe);
[dv, r] = size(tagVector);

% init L
for i=1:partN
    [Similarity{i}] = calSimilarity(xTe{i},kNN);
    Dimilarity{i} = diag(sum(Similarity{i},1));
    LaplacianM{i} = (Dimilarity{i}-Similarity{i});
end
L = zeros(N,N);
weightL = ones(partN,1)/(partN);
for i = 1:partN
    L = L + weightL(i)*LaplacianM{i};
end

%% inference visual representation
XXXX = [];
UUUU = MultiM;
for j=1:partN
   XXXX = [XXXX;xTe{j}]; 
end
if typeProj==2
    XXXX = XXXX - repmat(normP.Xmean, [1 N]);
    XXXX = XXXX./repmat(normP.Xstd, [1 N]);
end
vTest = (UUUU'*UUUU + alpha2*eye(d,d))\(UUUU'*XXXX);

%% inference tag
PreDY = W_mvsmp*vTest;% + b*ones(1,N);
% parameter.mode = 2;
% parameter.lambda = gamma1;
% parameter.lambda2 = gamma2;
% predTe = mexLasso(PreDY, tagVector, parameter);
Y = zeros(r,N);
O = Y;
P = Y;
Q1 = Y;
Q2 = Y;
MU1 = 1e-2;
MU2 = 1e-2;
RHO = 10;
DtD = tagVector'*tagVector;
DtX = tagVector'*PreDY;
ViVj = vTest'*vTest;
% Vnorm = sqrt(sum(vTest.^2,1));
% VsV = VsV./(repmat(Vnorm,[N,1]).*repmat(Vnorm',[1 N]));
ViVi = repmat(diag(ViVj)',[N,1]);
VjVj = repmat(diag(ViVj), [1, N]);
VsV = exp(-(ViVi + VjVj -2*ViVj)/sigma);
[tt pp] = sort(VsV,1,'descend');
theshold = tt(5,:);
VsV(find(VsV<repmat(theshold,[N,1]))) = 0;
VsV(find(VsV<repmat(theshold',[1,N]))) = 0;
DiagVsV = diag(sum(VsV,1));
L = DiagVsV - VsV;


YsY = yTr*yTr';
Ynorm = sqrt(sum(yTr.^2,2));
YsY = YsY./(repmat(Ynorm,[1,r]).*repmat(Ynorm',[r 1]));
% YiYi = repmat(diag(YiYj),[1,r]);
% YjYj = repmat(diag(YiYj)', [r, 1]);
% YsY = exp(-(YiYi + YjYj -2*YiYj)/sigma);
[tt pp] = sort(YsY,1,'descend');
theshold = tt(10,:);
YsY(find(YsY<repmat(theshold,[r,1]))) = 0;
YsY(find(YsY<repmat(theshold',[1,r]))) = 0;
DiagYsY = diag(sum(YsY,1));
M = DiagYsY - YsY;


for iter=1:80
    iter
   % opt Y
   Y = (DtD + (gamma2 + MU1 + MU2)*eye(r,r) + gamma4*M)\(DtX + MU1*O + MU2*P -Q1 -Q2);
   
   % opt O
   O_prev = O;
   tmp = Y + Q1/MU1;
   O = sign(tmp).*max(abs(tmp)-(gamma1/MU1),0);
   
   % opt P
   P_prev = P;
   P = (Q2 + MU2*Y)/(gamma3*L + MU2*eye(N,N));
   
   % opt Q1 Q2
   Q1 = Q1 + MU1*(Y-O);
   Q2 = Q2 + MU2*(Y-P);
   
    primal_residual = norm(Y-O,'fro');
    dual_residual = norm(MU1*(O-O_prev),'fro');
     if primal_residual>RHO*dual_residual
         MU1 = 2*MU1;
         fprintf('\n Mu1 is %f \n ', MU1);
     elseif dual_residual>RHO*primal_residual
         MU1 = MU1/2;
         fprintf('\n Mu1 is %f \n ', MU1);
     else
         MU1 = MU1;
     end
     
    primal_residual = norm(Y-P,'fro');
    dual_residual = norm(MU2*(P-P_prev),'fro');
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
predTe = Y;

end