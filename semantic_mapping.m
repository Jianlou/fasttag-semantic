function [W_mvsmp, b] = semantic_mapping(vTrain, yTr, tagVector, ParaM)

%% load parameter
beta1 = ParaM.beta1;
beta2 = ParaM.beta2;
[D N] = size(vTrain);
[d, r] = size(tagVector);

%% Init
Wpos = 1./max(1, sum(yTr>0, 2));
h = (sum(bsxfun(@times, (yTr>0), Wpos), 1))';
H = spdiags(h, 0, N, N);
g = h.*h;
G = H.*H;
A = H - (g*h')/(h'*h);
yTr = yTr./(repmat(sqrt(sum(yTr.^2,1)),[r,1])+1e-20);
YsY = (yTr*H)'*(yTr*H);
Ynorm = sqrt(sum((yTr*H).^2,1));
YsY = YsY./(repmat(Ynorm',[1 N]).*repmat(Ynorm,[N 1]) + 1e-20);
YdY = diag(sum(YsY,1));
L = YdY - YsY;

%% optimize
W_mvsmp = (tagVector*yTr*H*H'*vTrain')/(vTrain*H*H'*vTrain' + beta1*eye(D,D) + 2*beta2*vTrain*H*L*H'*vTrain');
b = (tagVector*yTr*g - W_mvsmp*vTrain*g)/(h'*h);
end