function [Ms, Ws] = optBW(Lfreq, L, alpha, noise, maxLayer, weights, invTr, Xb)

lambda = 1e-6;
maxIter = 10;
tol = 1e-2;

d = size(Xb, 1);
[r, n] = size(Lfreq);
M = L;

[B0s] = optB(Lfreq, L, weights,  noise, maxLayer, lambda);

weights = spdiags(weights', 0, length(weights), length(weights));

for ii = 1:maxLayer
	k = size(M, 1);
	iB = eye(k+1);
	iB(end, end) = 0;

	Mb = [M; ones(1, n)];
	weightedMb = weights*Mb';
	Sl = Mb*weightedMb;
	q = ones(k+1, 1)*(1-noise);
	q(end) = 1;

%     Q = Sl.*(q*q');
%     Q(1:k+2:end) = q.*diag(Sl);

    weightedMb = sqrt(weights)*Mb';
    Q = diag(sum(weightedMb,1));
        
        if ii == 1
                P = Lfreq*weightedMb.*repmat(q', r, 1);
        else
                P = Sl(1:end-1, :).*repmat(q', k, 1);
        end

	B = B0s{ii};
%     B = [eye(k),ones(k,1)];
%     B = eye(size(B));
    
	prevB = B;
	prevW = rand(r, d);

	Td = Mb*invTr;
	for iter = 1:maxIter
		W = B*Td;
		pred = W*Xb;
		B = (alpha*P + pred*weightedMb)/(alpha*Q + alpha*lambda*iB + Sl);
        
%         B = eye(size(B));
        
		optcondW = norm(W-prevW, 'fro')/norm(prevW, 'fro');
                optcondB = norm(B-prevB, 'fro')/norm(prevB, 'fro');
		if optcondW < tol & optcondB < tol
                        break;
                end
                prevW = W;
                prevB = B;
        end
	M = tanh(B*Mb);
	Ws{ii} = W;
	Ms{ii} = M;
end
