%--------------------------------------------------------------------------
% This function takes the D x N data matrix with columns indicating
% different data points and project the D dimensional data into the r
% dimensional space. Different types of projections are possible:
% (1) Projection using PCA
% (2) Projection using random projections with iid elements from N(0,1/r)
% (3) Projection using random projections with iid elements from symmetric
% bernoulli distribution: +1/sqrt(r),-1/sqrt(r) elements with same probability
% X: D x N data matrix of N data points
% r: dimension of the space to project the data to
% type: type of projection, {'PCA','NormalProj','BernoulliProj'}
% Xp: r x N data matrix of N projectred data points
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------


function Xp = DataProjection(X,r,type)

if r == 0
    Xp = X;  
else
    if (nargin < 3)
        type = 'NormalProj';
    end
    D = size(X,1);

    if ( strcmp(type , 'PCA') )

        [U,S,V] = svd(X',0);
        Xp = U(:,1:r)';

    elseif ( strcmp(type , 'NormalProj') )

        np = normrnd(0,1/sqrt(r),r*D,1);
        PrN = reshape(np,r,D);
        Xp = PrN * X;

    elseif( strcmp(type , 'BernoulliProj') )

        bp = rand(r*D,1);
        Bp = 1/sqrt(r) .* (bp >= .5) - 1/sqrt(r) .* (bp < .5);
        PrB = reshape(Bp,r,D);
        Xp = PrB * X;

    end
end