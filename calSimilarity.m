function [S] = calSimilarity(X,kNN)

    [D,N] = size(X);
    XIXJ = X'*X;
    XIXI = repmat(diag(XIXJ),[1 N]);
    XJXJ = repmat(diag(XIXJ)',[N 1]);
    Dist2 = XIXI + XJXJ - 2*XIXJ;
    [tt, pp] = sort(Dist2,'ascend');
    Delta = repmat(sqrt(tt(kNN,:)),[N 1]).*repmat(sqrt(tt(kNN,:))',[1 N]);
    S = exp(-Dist2./(Delta+1e-20));
%     [tt, pp] = sort(S,'descend');
%     S(find(S<repmat(tt(kNN,:),[N 1])))=0;
%     S(find((S-S')~=0))=0;
end