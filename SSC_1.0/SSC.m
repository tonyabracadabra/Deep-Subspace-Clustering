%--------------------------------------------------------------------------
% This is the main function for running SSC. 
% Load the DxN matrix X representing N data points in the D dim. space 
% living in a union of n low-dim. subspaces.
% The projection step onto the r-dimensional space is arbitrary and can 
% be skipped. In the case of using projection there are different types of 
% projections possible: 'NormalProj', 'BernoulliProj', 'PCA'. Please refer 
% to DataProjection.m for more information.
%--------------------------------------------------------------------------
% X: DxN matrix of N points in D-dim. space living in n low-dim. subspaces
% s: groundtruth for the segmentation
% n: number of subspaces
% r: dimension of the projection e.g. r = d*n (d: max subspace dim.)
% Cst: 1 if using the constraint sum(c)=1 in Lasso, else 0
% OptM: optimization method {'L1Perfect','L1Noise','Lasso','L1ED'}, see 
% SparseCoefRecovery.m for more information
% lambda: regularization parameter for 'Lasso' typically in [0.001,0.01] 
% or the noise level for 'L1Noise'. See SparseCoefRecovery.m for more 
% information.
% K: number of largest coefficients to pick in order to build the
% similarity graph, typically K = max{subspace dimensions} 
% Missrate: vector of misclassification rates
%--------------------------------------------------------------------------
% In order to run the code CVX package must be installed in Matlab. It can 
% be downlaoded from http://cvxr.com/cvx/download
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------

clc, clear all, close all
D = 30; %Dimension of ambient space
n = 2; %Number of subspaces
d1 = 1; d2 = 1; %d1 and d2: dimension of subspace 1 and 2
N1 = 20; N2 = 20; %N1 and N2: number of points in subspace 1 and 2
X1 = randn(D,d1) * randn(d1,N1); %Generating N1 points in a d1 dim. subspace
X2 = randn(D,d2) * randn(d2,N2); %Generating N2 points in a d2 dim. subspace
X = [X1 X2];
s = [1*ones(1,N1) 2*ones(1,N2)]; %Generating the ground-truth for evaluating clustering results
r = 0; %Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
Cst = 0; %Enter 1 to use the additional affine constraint sum(c) == 1
OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
lambda = 0.001; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
K = max(d1,d2); %Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients
if Cst == 1
    K = max(d1,d2) + 1; %For affine subspaces, the number of coefficients to pick is dimension + 1 
end

Xp = DataProjection(X,r,'NormalProj');
CMat = SparseCoefRecovery(Xp,Cst,OptM,lambda);
[CMatC,sc,OutlierIndx,Fail] = OutlierDetection(CMat,s);
if (Fail == 0)
    CKSym = BuildAdjacency(CMatC,K);
    [Grps , SingVals, LapKernel] = SpectralClustering(CKSym,n);
    Missrate = Misclassification(Grps,sc);
    save Lasso_001.mat CMat CKSym Missrate SingVals LapKernel Fail
else
    save Lasso_001.mat CMat Fail
end