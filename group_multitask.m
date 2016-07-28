function [W, prec, rec, f1, retrieved] = group_multitask(xTr, yTr, xTe, yTe, topK)


%% parameters

for K = [1];
%corel5k 30 esp 30

%% tag clustering
opts = statset('Display','final');
[KMIDX, KMC] = kmeans(yTr, K, 'Distance', 'cosine','Replicates',50);

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
%%
beta3 = [0];
    
beta2 = [0];
%beta2 corel5k 5e-4
for beta = [1e-3];
%beta corel5k 1e-3 esp 3e-5
kNN = 300;
tic
% [W,b] = group_multitask_admm2(xTr, yTr, beta, beta2, Idx);
[W,b] = group_multitask_manifold_admm(xTr, yTr, [xTr,xTe], beta, beta2, beta3, Idx, kNN);
predTe = W*xTe + b*ones(size(xTe,2),1)';
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nGroupMultitask :: Prec = %f, Rec = %f, F1 = %f, N+ = %d K = %d \n',  prec, rec, f1, retrieved,K);
toc
end

end
end