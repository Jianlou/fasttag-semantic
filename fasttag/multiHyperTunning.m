function [hyperparams] = multiHyperTunning(xTr, yTr, xVal, yVal, topK)

fprintf('\nhyper-parameter tuning ... ')
threshold = 0.0;
thresholdPos = 0;
maxLayer = 3;
% regularization for W, might need to tune it when experiment with other datasets
beta = 1e-2;

d = size(xTr, 1);
K = size(yTr, 1);
iW = spdiags([ones(d-1, 1); 0], 0, d, d);
% iW = spdiags([ones(d, 1)], 0, d, d);

bestF1 = 0;
bestL = zeros(size(yTr)); 
bestRetrievedIdxF1 = false(K, 1);
bestWF1 = zeros(K, d);

optIter = 1;

Wpos = 1./max(1, sum(yTr>0, 2));
while true
	improved = false;

	predVal = bestWF1*xVal;

	tagIdx = find(bestRetrievedIdxF1==0);
	instanceIdx = sum(yTr(tagIdx, :)>0, 1)>0;

	weights = sum(bsxfun(@times, yTr(tagIdx, instanceIdx)>0, Wpos(tagIdx)),1);

        fprintf('\n optIter = %d, tagIdx = %d, instanceIdx = %d, thresholdPos = %f\n\n', optIter, length(tagIdx), sum(instanceIdx), thresholdPos);
    SxTr = xTr(:, instanceIdx)*spdiags(weights', 0, length(weights), length(weights))*xTr(:, instanceIdx)';
%     for beta = [1e-1 1e-2]
	invTr = spdiags(weights', 0, length(weights), length(weights))*xTr(:, instanceIdx)'/(SxTr+beta*iW);
    for alpha = [1e0, 1e1, 1e2, 1e3, 1e4]
		for noise = 0:0.1:0.9
			[Ms, Ws] = optBW(yTr(tagIdx, instanceIdx), yTr(:, instanceIdx), alpha, noise, maxLayer, weights, invTr, xTr(:, instanceIdx));
			for layer = 1:size(Ws, 2)
				L = Ms{layer};
				if (mean(L(yTr(tagIdx, instanceIdx)>0)) < thresholdPos)
					continue;
				end
				W = Ws{layer};
                                predVal(tagIdx, :) = W*xVal;
              	                [precVal, recVal, f1Val, retrievedVal, f1Ind, precInd, recInd]= evaluate(yVal, predVal, topK);
                                if f1Val > bestF1
                                        fprintf('beta = %e, alpha = %e, noise = %f, layer = %d, precVal = %f, recVal = %f, f1Val = %f, retrievedVal = %d\n', beta, alpha, noise, layer,  precVal, recVal, f1Val, retrievedVal);
					bestL = L;
                                        bestF1 = f1Val;
                                        bestWF1(tagIdx, :) = W;
					bestRetrievedIdxF1 = (f1Ind > threshold);
					
					hyperparams(optIter).tagIdx = tagIdx;
					hyperparams(optIter).beta = beta;
					hyperparams(optIter).noise = noise;
					hyperparams(optIter).alpha = alpha;
					hyperparams(optIter).layers = layer;
						
					improved = true;
			                    end
            end
                end
%     end
    end
	if ~improved
		break;
	end
	thresholdPos = mean(bestL(yTr(tagIdx, instanceIdx)>0));
	optIter = optIter + 1;
end
