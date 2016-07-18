import cvxpy as cp
import numpy as np


def sparseCoefRecovery(X, Opt, l=0.001):
	d, n = X.shape
	C = np.zeros((n, n))
	# Minimize(sum_squares(X - X*C))

	for i in xrange(n):
		print i

		y = X[:, i]
		Y = np.delete(X, (i), axis=1)

		c = cp.Variable(n-1,1)
		if Opt == 'Lasso':
			objective = cp.Minimize(cp.norm(c,1) + l*cp.norm(Y*c -y))
			constraints = []

		elif Opt =='L1Perfect':
			objective = cp.Minimize(cp.norm(c, 1))
			constraints = [Y * c  == y]

		prob = cp.Problem(objective, constraints)
		prob.solve()

		c_val = np.array(c.value)[:,0]

		if i > 1:
			C[:i-1,i] = c_val[:i-1]
		if i < n:
			C[i+1:n,i] = c_val[i:n]
		C[i,i] = 0

	return C
