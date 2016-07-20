import cvxpy as cp
import numpy as np
import blitzl1

def sparseCoefRecovery(X, l=0.001):
	d, n = X.shape
	C = np.zeros((n, n))

	for i in xrange(n):
		if i % 100 == 0:
			print "Processed for " + str(i) + "samples"

		A = np.delete(X, (i), axis=1)
		b = X[:, i]

		prob = blitzl1.LogRegProblem(A, b)
		lammax = prob.compute_lambda_max()
		sol = prob.solve(l * lammax)

		c_val = sol.x

		if i > 1:
			C[:i-1,i] = c_val[:i-1]
		if i < n:
			C[i+1:n,i] = c_val[i:n]
		C[i,i] = 0

	return C