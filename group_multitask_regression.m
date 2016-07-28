function [W, prec, rec, f1, retrieved] = group_multitask_regression(xTr, yTr, xTe, yTe, topK, valIdx)

xTrain = [];
xTest = [];
for i=1:length(xTr)
    xTrain = [xTrain;xTr{i}];
    xTest = [xTest;xTe{i}];
end

D = size(xTrain,1);
d = 5000;
P = randn(D,d);
[Q, ~] = qr(P,0);
P = Q;

clear xTr xTe

for beta = [1e-2]
tic
[W,b] = linear_analy(P'*xTrain, yTr, beta);
predTe = W*P'*xTest + b*ones(size(P'*xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc
end

%% load tag
% load('/media/pris/Study/paperPrj/wordembedding/corel5k_tagVector.mat');

%% tag clustering
for K = [10]
    
opts = statset('Display','final');
[KMIDX, KMC] = kmeans(yTr, K, 'Distance', 'cosine','Replicates',100);

yTrTmp = yTr;
yTeTmp = yTe;
Idx(1,1) = 1;
Idx(2,1) = 1;
for i =1:K
    IDX = find(KMIDX == i);
    Idx(2,i) = length(IDX) + Idx(1,i) -1 ;
    yTr(Idx(1,i):Idx(2,i),:) = yTrTmp(IDX,:);
    yTe(Idx(1,i):Idx(2,i),:) = yTeTmp(IDX,:);
    if (Idx(2,i) + 1) <= size(yTr,1);
        Idx(1,i+1) = Idx(2,i) + 1;
    end
    clear IDX;
end


for beta1 = [5e-3]
    beta2 =0;
tic
[W,b] = group_multitask_analy(P'*xTrain, yTr, beta1, beta2,Idx);
predTe = W*P'*xTest + b*ones(size(P'*xTest,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc
end

end

end