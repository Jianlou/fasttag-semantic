function [X] = rpSVD(dataFolder, dimen);


featNames = {'DenseHue.hvecs', 'DenseHueV3H1.hvecs', 'DenseSift.hvecs', 'DenseSiftV3H1.hvecs', 'Gist.fvec', 'HarrisHue.hvecs', 'HarrisHueV3H1.hvecs', 'HarrisSift.hvecs', 'HarrisSiftV3H1.hvecs', 'Hsv.hvecs32', 'HsvV3H1.hvecs32', 'Lab.hvecs32', 'LabV3H1.hvecs32', 'Rgb.hvecs32', 'RgbV3H1.hvecs32'};
	
tic;
X = {};
iter = 0;
for feat = featNames
    iter = iter+1;
        feat = char(feat);
	matchStr = regexp(ls(dataFolder), ['\w*_train_', feat], 'match');
        tmp = vec_read(strcat(dataFolder, matchStr{1}));
	combine = double(tmp);
	[nTr, d0] = size(tmp);

	matchStr = regexp(ls(dataFolder), ['\w*_test_', feat], 'match');
	tmp = vec_read(strcat(dataFolder, matchStr{1}));
	nTe = size(tmp, 1);%chi-squre
	combine = [combine; double(tmp)];

	if ~strcmp(feat, 'Gist.fvec')
		combine = spdiags(sum(abs(combine), 2)+eps, 0, size(combine, 1), size(combine, 1))*combine;
	end	
        if strcmp(feat, 'Gist.fvec')
		PhiX = combine';
        elseif strcmp(feat, 'Hsv.hvecs32')||strcmp(feat, 'HsvV3H1.hvecs32')||strcmp(feat, 'Lab.hvecs32')||strcmp(feat, 'LabV3H1.hvecs32')||strcmp(feat, 'Rgb.hvecs32')||strcmp(feat, 'RgbV3H1.hvecs32')
                PhiX = homogeneous_feature_map(combine', 1, 1.2, 'intersection', 1, 1);
        else
           PhiX = homogeneous_feature_map(combine', 1, 0.6, 'chi-square', 1, 1);
%            PhiX = homogeneous_feature_map(combine', 1, 0.6, 'intersection', 1, 1);
        end

	homoDim = size(PhiX, 1);

	if size(PhiX, 1) <= dimen
%         X{iter} = PhiX - repmat(mean(PhiX,2),[1 size(PhiX,2)]);
%         X{iter} = PhiX./(repmat(std(PhiX')', [1 size(PhiX,2)])+1e-20);
         X{iter} = PhiX;
		fprintf('feature = %s, dimen = %d, homoDim = %d, RPDim = %d\n', feat, d0, size(PhiX, 1), size(PhiX, 1));
	else
		[d, n] = size(PhiX);
		R = randn(n, dimen);
        Y = PhiX*R;
        [Q, ~] = qr(Y, 0);
		PhiX = Q'*PhiX;
%         X{iter} = PhiX - repmat(mean(PhiX,2),[1 size(PhiX,2)]);
%         X{iter} = PhiX./(repmat(std(PhiX')', [1 size(PhiX,2)])+1e-20);
        X{iter} = PhiX;
		fprintf('feature = %s, dimen = %d, homoDim = %d, RPDim = %d\n', feat, d0, homoDim, dimen);
	end
end
toc;
