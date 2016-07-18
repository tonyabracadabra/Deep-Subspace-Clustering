%--------------------------------------------------------------------------
% This function takes the coefficient matrix resulted from sparse
% representation using \ell_1 minimization. If a point cannot be written as
% a linear combination of other points, it should be an outlier. The
% function detects the indices of outliers and modifies the coefficient
% matrix and the ground-truth accordingly.
% CMat: NxN coefficient matrix
% s: Nx1 ground-truth vector
% CMatC: coefficient matrix after eliminating Nans
% sc: ground-truth after eliminating outliers
% OutlierIndx: indices of outliers in {1,2,...,N}
% Fail: 1 if number of inliers is less than number of groups, 0 otherwise
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------


function [CMatC,sc,OutlierIndx,Fail] = OutlierDetection(CMat,s)

n = max(s);
N = size(CMat,2);
NanIndx = [];
FailCnt = 0;
Fail = 0;

for i = 1:N
    c = CMat(:,i);
    if( sum(isnan(c)) >= 1 )
        NanIndx = [NanIndx ; i];
        FailCnt = FailCnt + 1;
    end
end

sc = s;
sc(NanIndx) = [];
CMatC = CMat;
CMatC(NanIndx,:) = [];
CMatC(:,NanIndx) = [];
OutlierIndx = NanIndx;

if ( FailCnt > N - n )
    CMatC = [];
    sc = [];
    Fail = 1;
end