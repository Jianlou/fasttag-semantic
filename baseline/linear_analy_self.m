function [ W ] = linear_analy_self( X, Y, X1, Y1, Y2, lambda, valIdx, topK)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% 
% xTr = [X(:, ~valIdx)];
% yTr = [Y(:, ~valIdx)];
% xVa = X(:,valIdx);
% yVa = Y(:,valIdx);
xTr = [X, X1];
yTr = [Y, Y1];
xVa = X1;
yVa = Y2;
bestF1 = 0;
threshold = 0.4;
rho = 1.0;
iter =0;

[d n] = size(xTr);
r = size(yTr,1);

% Wpos = 1./max(1, sum(yTr>0, 2));
% weights = max(bsxfun(@times, (yTr>0), Wpos), [], 1).^1;
% iW = spdiags([ones(d-1, 1); 0], 0, d, d);
% 
% Sx = xTr*spdiags(weights', 0, n, n)*xTr';
while iter<5
    iter = iter +1;
    
    Wpos = 1./max(1, sum(yTr>0, 2));
%     Wpos = ones(size(yTr,1),1);
    weights = max(bsxfun(@times, (yTr>0), Wpos), [], 1).^1;
    iW = spdiags([ones(d-1, 1); 0], 0, d, d);

    Sx = xTr*spdiags(weights', 0, n, n)*xTr';
    Sxy = yTr*spdiags(weights', 0, n, n)*xTr';
    Wpre = Sxy/(Sx + lambda*iW);
    predVa = Wpre*xVa;
    [prec, rec, f1, retrieved] = evaluate(yVa, predVa, topK);
    fprintf('Prec is %f, Rec is %f, F1 is %f, N+ is %d \n',prec, rec, f1, retrieved);
%     if f1<=bestF1
%         break;
%     else
%         bestF1 = f1;
%     end
    yNew = tanh(Wpre*X1);
%     yTr(find(and(yTr == 1, yNew < threshold))) = 0;
%     yNew(yTr ==1 ) = 0;
%     [maxV, maxIdx] = max(yNew, [], 1);
%     for i=1:size(yTr,2)
%         if maxV(i) > threshold
%             yTr(maxIdx(i),i) = 1;
%         end
%     end
    yTr(:,(size(X,2)+1):(size(X,2) + size(X1,2))) = yNew;
    threshold = rho*threshold;
    W = Wpre;
%     yTr(find(and(yNew>threshold,yTr < 0.5))) = 1;
end
end

