%--------------------------------------------------------------------------
% This function takes the D x N matrix of N data points and write every
% point as a sparse linear combination of other points.
% Xp: D x N matrix of N data points
% cst: 1 if using the affine constraint sum(c)=1, else 0
% Opt: type of optimization, {'L1Perfect','L1Noisy','Lasso','L1ED'}
% lambda: regularizartion parameter of LASSO, typically between 0.001 and 
% 0.1 or the noise level for 'L1Noise'
% CMat: N x N matrix of coefficients, column i correspond to the sparse
% coefficients of data point in column i of Xp
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------


function CMat = SparseCoefRecovery(Xp,cst,Opt,lambda)

if (nargin < 2)
    cst = 0;
end
if (nargin < 3)
    Opt = 'Lasso';
end
if (nargin < 4)
    lambda = 0.001;
end

D = size(Xp,1);
N = size(Xp,2);

for i = 1:N
    
    y = Xp(:,i);
    if i == 1
        Y = Xp(:,i+1:end);
    elseif ( (i > 1) && (i < N) )
        Y = [Xp(:,1:i-1) Xp(:,i+1:N)];        
    else
        Y = Xp(:,1:N-1);
    end
    
    % L1 optimization using CVX
    if cst == 1
        if ( strcmp(Opt , 'Lasso') )
            cvx_begin;
            cvx_precision high
            variable c(N-1,1);
            minimize( norm(c,1) + lambda * norm(Y * c  - y) );
            subject to
            sum(c) == 1;
            cvx_end;
        elseif ( strcmp(Opt , 'L1Perfect') )
            cvx_begin;
            cvx_precision high
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            Y * c  == y;
            sum(c) == 1;
            cvx_end;
        elseif ( strcmp(Opt , 'L1Noisy') )
            cvx_begin;
            cvx_precision high
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            norm( Y * c  - y ) <= lambda;
            sum(c) == 1;
            cvx_end;
        elseif ( strcmp(Opt , 'L1ED') )
            cvx_begin;
            cvx_precision high
            variable c(N-1+D,1);
            minimize( norm(c,1) );
            subject to
            [Y eye(D)] * c  == y;
            sum(c(1:N-1)) == 1;
            cvx_end;
        end
    else
        if ( strcmp(Opt , 'Lasso') )
            cvx_begin;
            cvx_precision high
            variable c(N-1,1);
            minimize( norm(c,1) + lambda * norm(Y * c  - y) );
            cvx_end;
        elseif ( strcmp(Opt , 'L1Perfect') )
            cvx_begin;
            cvx_precision high
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            Y * c  == y;
            cvx_end;
        elseif ( strcmp(Opt , 'L1Noisy') )
            cvx_begin;
            cvx_precision high
            variable c(N-1,1);
            minimize( norm(c,1) );
            subject to
            norm( Y * c  - y ) <= lambda;
            cvx_end;
        elseif ( strcmp(Opt , 'L1ED') )
            cvx_begin;
            cvx_precision high
            variable c(N-1+D,1);
            minimize( norm(c,1) );
            subject to
            [Y eye(D)] * c  == y;
            cvx_end;
        end
    end
    
    % place 0's in the diagonals of the coefficient matrix
    if i == 1   
        CMat(1,1) = 0;
        CMat(2:N,1) = c(1:N-1);       
    elseif ( (i > 1) && (i < N) )
        CMat(1:i-1,i) = c(1:i-1);
        CMat(i,i) = 0;
        CMat(i+1:N,i) = c(i:N-1);
    else
        CMat(1:N-1,N) = c(1:N-1);
        CMat(N,N) = 0;
    end

end