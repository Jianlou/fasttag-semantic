function [Bs] = optB(Lfreq, L, weights, noise, maxLayers, lambda)

[r, n] = size(Lfreq);
M = L;

weights = spdiags(weights', 0, length(weights), length(weights));
for ii = 1:maxLayers

	k = size(M, 1);
	iB = eye(k+1);
	iB(end, end) = 0;
	Mb = [M; ones(1, n)];

	weightedMb = weights*Mb';
	Sl = Mb*weightedMb;	
	q = ones(k+1, 1)*(1-noise);
	q(end) = 1;

	Q = Sl.*(q*q');
	Q(1:k+2:end) = q.*diag(Sl);
    
%     weightedMb = sqrt(weights)*Mb';
%     Q = diag(sum(weightedMb,1));

	if ii == 1
		P = Lfreq*weightedMb.*repmat(q', r, 1);
	else
		P = Sl(1:end-1, :).*repmat(q', k, 1);
	end
	B = P/(Q+lambda*iB);
%     B = eye(size(B));
    
        Bs{ii} = B;
	M = tanh(B*Mb);

end

