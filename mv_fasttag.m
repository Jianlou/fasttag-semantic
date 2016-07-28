function [W, prec, rec, f1, retrieved] = fasttag(xTr, yTr, xTe, yTe, topK, valIdx)

[xTrain, xTest, xTeSem] = multiview_feature(xTr, xTe, yTr);

xTrain = [xTrain; ones(1, size(xTrain, 2))];
xTest = [xTest; ones(1, size(xTest, 2))];

[d, nTr] = size(xTrain);

[hyperparams] = multiHyperTunning(xTrain, yTr, xTest, yTe, topK);

K = size(yTr, 1);
iW = spdiags([ones(d-1, 1); 0], 0, d, d);
Wpos = 1./max(1, sum(yTr>0, 2));
	
tic
W = zeros(K, d);
for optIter = 1:size(hyperparams, 2)
	tagIdx = hyperparams(optIter).tagIdx;
	beta = hyperparams(optIter).beta;
	noise = hyperparams(optIter).noise;
	alpha = hyperparams(optIter).alpha;
	layers = hyperparams(optIter).layers;

	instanceIdx = sum(yTr(tagIdx, :)>0, 1)>0;
	weights = sum(bsxfun(@times, yTr(tagIdx, instanceIdx)>0, Wpos(tagIdx)), 1);
        fprintf('\n optIter = %d, tagIdx = %d, instanceIdx = %d\n\n', optIter, length(tagIdx), sum(instanceIdx));

	Sx = xTrain(:, instanceIdx)*spdiags(weights', 0, length(weights), length(weights))*xTrain(:, instanceIdx)';
	invSx = spdiags(weights', 0, length(weights), length(weights))*xTrain(:, instanceIdx)'/(Sx+beta*iW);

	[Ms, Ws] = optBW(yTr(tagIdx, instanceIdx), yTr(:, instanceIdx), alpha, noise, layers, weights, invSx, xTrain(:, instanceIdx));
	
	W(tagIdx, :) = Ws{layers};	
	predTe = W*xTest;
	[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
	fprintf('FastTag :: Beta = %e, Noise = %f, Layer = %d, Alpha = %e, Prec = %f, Rec = %f, F1 = %f, N+ = %d\n', beta, noise, layers, alpha, prec, rec, f1, retrieved);
end
toc