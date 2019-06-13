#!/usr/bin/python
# ---------------------------------------------------------------------------
# Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46)
# cplex Version 12.8.0.0
# ---------------------------------------------------------------------------

import numpy as np
import cplex
import math


def BoolProd(C, R):
    """
    :param C:           (0,1) matrix of size n x k
    :param R:           (0,1) matrix of size k x m
    :return X:          (0,1) matrix of size n x m such that x_ij = V_l ( c_il * r_lj ), the Boolean matrix product
    """

    if C.shape[1] != R.shape[0]:
        raise ValueError('Incompatible Matrix Sizes')

    X = np.dot(C,R)
    X[X > 0] = 1
    return X

def GenerateX(n,m,Kappa,Seed,Sigma=0.5,Mu=0):

    """
    :param n:           number of rows of X
    :param m:           number of columns of X
    :param Kappa:       Boolean rank of X before noise is introduced
    :param Seed:        seed passed onto random number generator
    :param Sigma:       sparsity of X, as the probability of a random entry x_ij is zero
    :param Mu:          noise in X, as the probability of a random entry x_ij is perturbed

    :return X:          nxm (0,1)-matrix that is at most Mu*n*m Hamming distance away from an nxm Boolean rank Kappa matrix
    """

    # Random Number Generation Seed
    np.random.seed(Seed)

    # Sparsity
    q = 1 - np.sqrt(1 - Sigma**(1 / Kappa))     # probability of 0's in a Bernoulli trial
    p = 1 - q                                   # probability of 1's in a Bernoulli trial

    # Initial X with Boolean rank Kappa
    C = np.random.binomial(1, p, [n, Kappa])    # generating nxKappa C with probability p for a 1
    R = np.random.binomial(1, p, [Kappa, m])
    X = BoolProd(C, R)

    # Introducing Noise
    for noise in range(math.floor(Mu*n*m)):     # at most Mu * n * m entries perturbed in X
        i = np.random.choice(n)
        j = np.random.choice(m)                 # if same i, j comes up, noise is less than Mu

        X[i,j] = math.fabs(X[i,j] - 1)

    return X

def Preprocess(X):
    """
    :param X:                           (0,1) matrix to be preprocessed by removing row and column duplicates
                                        and zero rows and columns

    :return X_out:                      preprocessed (0,1)-matrix containing unique rows,columns and no zero rows,columns
    :return rowCounts,                  row counts/weights associated with each unique row of X_out
    :return colCounts,                  column counts/weights associated with each unique row of X_out
    :return uniqueRowIdx,               indices of unique rows to build back original matrix
    :return zeroRowIdx,                 index of last zero row
    :return uniqueColIdx,               indices of unique columns to build back original matrix
    :return zeroColIdx                  index of last zero column
    """

    #operations on rows

    #delete row duplicates but record their indices and counts
    X_uniqueRows, uniqueRowIdx, rowCounts = np.unique(X,
                                                      return_inverse = True,
                                                      return_counts = True,
                                                      axis = 0)

    zeroRowTrue = np.all(X_uniqueRows == 0, axis = 1)                 #find the row of all zeros if it exist

    if zeroRowTrue.sum() > 0:
        X_uniqueRows_noZeroRow = X_uniqueRows[~zeroRowTrue]           #delete that row of zeros
        rowCounts = rowCounts[~zeroRowTrue]                           #delete the count of zero rows
        zeroRowIdx = np.where(zeroRowTrue)[0][0]                      #get index of zero row
    else:
        X_uniqueRows_noZeroRow = X_uniqueRows
        zeroRowIdx = None



    #operations on columns

    #delete column duplicates but record their indices and counts
    X_uniqueCols, uniqueColIdx, colCounts = np.unique(X_uniqueRows_noZeroRow,
                                                      return_inverse = True,
                                                      return_counts = True,
                                                      axis = 1)

    zeroColTrue = np.all(X_uniqueCols == 0, axis = 0)                 #find the col of all zeros if it exists

    if zeroColTrue.sum() > 0:
        X_uniqueCols_noZeroCol = np.transpose(np.transpose(X_uniqueCols)[~zeroColTrue])
                                                                      #delete that col of zeros
        colCounts = colCounts[~zeroColTrue]                           #delete the count of zero columns
        zeroColIdx = np.where(zeroColTrue)[0][0]                      #store the index of zero column
    else:
        X_uniqueCols_noZeroCol = X_uniqueCols
        zeroColIdx = None

    X_out = X_uniqueCols_noZeroCol

    return X_out, rowCounts, colCounts, uniqueRowIdx, zeroRowIdx, uniqueColIdx, zeroColIdx

def UnPreprocess(X_in, uniqueRowIdx, zeroRowIdx, uniqueColIdx, zeroColIdx):
    """
    :param X_in:                (0,1)-matrix to be extended with duplicate and zero rows and columns
    :param uniqueRowIdx:        indices of row duplicates
    :param zeroRowIdx:          index of zero row
    :param uniqueColIdx:        indices of column duplicates
    :param zeroColIdx:          index of zero column

    :return X_out:              (0,1)-matrix containing duplicate and zero rows and columns
    """

    #operations on columns
    [n_tmp, m_tmp] = X_in.shape
    # if there was a zero column removed add it back
    if zeroColIdx is None:
        X_wZeroCol = X_in
    else:
        X_wZeroCol = np.concatenate((X_in[:,0:zeroColIdx],
                                     np.zeros([n_tmp,1],dtype = int),
                                     X_in[:,zeroColIdx:m_tmp]),
                                    axis = 1)
    #create duplicates of columns
    X = X_wZeroCol[:,uniqueColIdx]


    #operations on rows
    [n_tmp, m_tmp] = X.shape
    #if there was a zero row removed add it back
    if zeroRowIdx is None:
        X_wZeroRow = X
    else:
        X_wZeroRow = np.concatenate((X[0:zeroRowIdx,:],
                                     np.zeros([1,m_tmp],dtype = int),
                                     X[zeroRowIdx:n_tmp,:]),
                                    axis = 0)
    #create duplicates of rows
    X_out = X_wZeroRow[uniqueRowIdx]

    return X_out

def PostPreprocess(C_in,R_in, uniqueRowIdx, zeroRowIdx, uniqueColIdx, zeroColIdx):
    """
    :param C_in:                (0,1)-matrix to be extended with duplicate and zero columns
    :param R_in:                (0,1)-matrix to be extended with duplicate and zero rows
    :param uniqueRowIdx:        indices of row duplicates
    :param zeroRowIdx:          index of zero row
    :param uniqueColIdx:        indices of column duplicates
    :param zeroColIdx:          index of zero column

    :return C:              (0,1)-matrix containing duplicate and zero columns
    :return R:              (0,1)-matrix containing duplicate and zero rows
    """

    #operations on columns
    [n_tmp, m_tmp] = R_in.shape
    # if there was a zero column removed add it back
    if zeroColIdx is None:
        R_wZeroCol = R_in
    else:
        R_wZeroCol = np.concatenate((R_in[:,0:zeroColIdx],
                                     np.zeros([n_tmp,1],dtype = int),
                                     R_in[:,zeroColIdx:m_tmp]),
                                    axis = 1)
    #create duplicates of columns
    R = R_wZeroCol[:,uniqueColIdx]


    #operations on rows
    [n_tmp, m_tmp] = C_in.shape
    #if there was a zero row removed add it back
    if zeroRowIdx is None:
        C_wZeroRow = C_in
    else:
        C_wZeroRow = np.concatenate((C_in[0:zeroRowIdx,:],
                                     np.zeros([1,m_tmp],dtype = int),
                                     C_in[zeroRowIdx:n_tmp,:]),
                                    axis = 0)
    #create duplicates of rows
    C = C_wZeroRow[uniqueRowIdx]

    return C,R

def CPLEX_MODEL_BoolRank(X):

    """
    Builds Cplex object

    :param X:               nxm (0,1) matrix
    :return cpx:            Cplex object which containts (IP) to compute the Boolean rank of X

    k = min(n,m)

    (IP): min sum_l d_l

    s.t.    sum_l y_ilj >= 1                (i,j) s.t. x_ij=1
            y_ilj = 0                       (i,j) s.t. x_ij=0
            y_ilj <= c_il                   l \in [k], (i,j) s.t. x_ij=1
            y_ilj <= r_lj                   l \in [k], (i,j) s.t. x_ij=1
            c_il + r_lj <= 1                l \in [k], (i,j) s.t. x_ij=0
            c_il <= d_l                     i \in [n], l \in [k]
            r_lj <= d_l                     l \in [k], j \in [m]

            y_ilj \in [0,1]                 i \in [n], l \in [k], j \in [m]
            c_il, r_lj, d_l \in {0,1}       i \in [n], l \in [k], j \in [m]

    """

    n,m = X.shape

    k = min(n,m)

    idx0 = np.transpose(np.where(X == 0))
    idx1 = np.transpose(np.where(X == 1))

    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)

    # c_i,l
    cpx.variables.add(obj = [0 for i in range(n) for l in range(k)],
                      lb = [0 for i in range(n) for l in range(k)],
                      ub = [1 for i in range(n) for l in range(k)],
                      types = [cpx.variables.type.binary for i in range(n) for l in range(k)],
                      names = ["c_"+str(i)+","+str(l) for i in range(n) for l in range(k)])

    # r_l,j
    cpx.variables.add(obj = [0 for l in range(k) for j in range(m)],
                      lb = [0 for l in range(k) for j in range(m)],
                      ub = [1 for l in range(k) for j in range(m)],
                      types = [cpx.variables.type.binary for l in range(k) for j in range(m)],
                      names = ["r_"+str(l)+","+str(j) for l in range(k) for j in range(m)])

    # y_i,l,j
    cpx.variables.add(obj = [0 for i in range(n) for l in range(k) for j in range(m)],
                      lb = [0 for i in range(n) for l in range(k) for j in range(m)],
                      ub = [1 for i in range(n) for l in range(k) for j in range(m)],
                      types = [cpx.variables.type.continuous for i in range(n) for l in range(k) for j in range(m)],
                      names = ["y_"+str(i)+","+str(l)+","+str(j) for i in range(n) for l in range(k) for j in range(m)])

    # d_l
    cpx.variables.add(obj = [1 for l in range(k)],
                      lb = [0 for l in range(k)],
                      ub = [1 for l in range(k)],
                      types = [cpx.variables.type.binary for l in range(k)],
                      names = ["d_"+str(l) for l in range(k)])


    #CONSTRAINTS


    # sum_l y_ilj => 1
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = [ "y_"+str(ij[0])+","+str(l)+","+str(ij[1]) for l in range(k)],
                             val = [-1 for l in range(k)])
            for ij in idx1
        ],
        rhs = [-1 for ij in idx1],
        senses = ["L" for ij in idx1],
        names = [
            "sum_l y_"+str(ij[0])+",l,"+str(ij[1])+" <= 1" for ij in idx1
        ]
    )

    # y_i,l,j = 0
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["y_"+str(ij[0])+","+str(l)+","+str(ij[1])],
                             val = [1])
            for l in range(k) for ij in idx0
        ],
        rhs = [0 for l in range(k) for ij in idx0],
        senses = ["E" for l in range(k) for ij in idx0],
        names = [
            "y_"+str(ij[0])+","+str(l)+","+str(ij[1])+" = 0"
            for l in range(k) for ij in idx0
        ]
    )

    # y_i,l,j <= c_i,l
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["y_"+str(ij[0])+","+str(l)+","+str(ij[1]), "c_"+str(ij[0])+","+str(l)],
                             val = [1, -1])
            for l in range(k) for ij in idx1
        ],
        rhs = [0 for l in range(k) for ij in idx1],
        senses = ["L" for l in range(k) for ij in idx1],
        names = [
            "y_"+str(ij[0])+","+str(l)+","+str(ij[1])+" <= c_"+str(ij[0])+","+str(l)
            for l in range(k) for ij in idx1
        ]
    )

    # y_i,l,j <= r_l,j
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["y_"+str(ij[0])+","+str(l)+","+str(ij[1]), "r_"+str(l)+","+str(ij[1])],
                             val = [1, -1])
            for l in range(k) for ij in idx1
        ],
        rhs = [0 for l in range(k) for ij in idx1],
        senses = ["L" for l in range(k) for ij in idx1],
        names = [
            "y_"+str(ij[0])+","+str(l)+","+str(ij[1])+" <= r_"+str(l)+","+str(ij[1])
            for l in range(k) for ij in idx1
        ]
    )

    # c_i,l + r_l,j <= 1
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["c_"+str(ij[0])+","+str(l), "r_"+str(l)+","+str(ij[1])],
                             val = [1, 1])
            for l in range(k) for ij in idx0
        ],
        rhs = [1 for l in range(k) for ij in idx0],
        senses = ["L" for l in range(k) for ij in idx0],
        names = [
            "c_"+str(ij[0])+","+str(l)+" + "+"r_"+str(l)+","+str(ij[1])+" <= 1" for l in range(k) for ij in idx0
        ]
    )

    # c_i,l <= d_l
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["c_"+str(i)+","+str(l), "d_"+str(l)],
                             val = [1, -1])
            for i in range(n) for l in range(k)
        ],
        rhs = [0 for i in range(n) for l in range(k)],
        senses = ["L" for i in range(n) for l in range(k)],
        names = [
            "c_"+str(i)+","+str(l)+" <= d_"+str(l) for i in range(n) for l in range(k)
        ]
    )

    # r_l,j <= d_l
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["r_"+str(l)+","+str(j), "d_"+str(l)],
                             val = [1, -1])
            for j in range(m) for l in range(k)
        ],
        rhs = [0 for j in range(m) for l in range(k)],
        senses = ["L" for j in range(m) for l in range(k)],
        names = [
            "r_"+str(l)+","+str(j)+" <= d_"+str(l) for j in range(m) for l in range(k)
        ]
    )

    # PARTIAL SYMMETRY BREAKING
    # d_(l-1) => d_l
    cpx.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind = ["d_"+str(l-1), "d_"+str(l)],
                             val = [1, -1])
            for l in range(1,k)
        ],
        rhs = [0 for l in range(1,k)],
        senses = ["G" for l in range(1,k)],
        names = [
            "d_"+str(l-1)+" >= d_"+str(l) for l in range(1,k)
        ]
    )

    return cpx

def Get_BooleanRank(X_in, Epsilon=0.00000001):
    """
    - Preprocesses X_in
    - Solves Cplex object
    - Unpreprocessess output from Cplex

    :param X_in:        NxM (0,1) matrix
    :param Epsilon:     numerical precision tolerance

    :return rank:       Boolean rank of X_in
    :return A:          left (0,1) factor matrix of X_in, of dimension N x rank
    :return B:          right (0,1) factor matrix of X_in, of dimension rank x M
    """

    if X_in.dtype != np.dtype(int):
        raise ValueError('Input matrix is not integer numpy array')

    if (X_in == 0).sum() + (X_in==1).sum() != X_in.size:
        raise ValueError('Input matrix is not a (0,1) numpy array')


    X, rowWeights, colWeights, uniqueRowIdx, zeroRowIdx, uniqueColIdx, zeroColIdx = Preprocess(X_in)
    n,m = X.shape
    k = min(n,m)

    cpx = CPLEX_MODEL_BoolRank(X)

    #apply inbuilt aggressive symmetry breaking
    cpx.parameters.preprocessing.symmetry.set(5)

    cpx.solve()

    rank_FLOAT = cpx.solution.get_objective_value()

    # numerical precision checking I.

    if np.abs( int(rank_FLOAT) - rank_FLOAT) > Epsilon:
        raise ValueError('Cplex returned a non-integer rank')
    else:
        rank = int(rank_FLOAT)


    solution = cpx.solution.get_values()
    C_w_zeros_FLOAT = np.array(solution[ : n*k]).reshape(n,k)
    R_w_zeros_FLOAT = np.array(solution[n*k : n*k + k*m]).reshape(k,m)

    # numerical precision checking II.

    C_0_idx = np.abs(C_w_zeros_FLOAT - 0) < Epsilon
    C_1_idx = np.abs(C_w_zeros_FLOAT - 1) < Epsilon
    R_0_idx = np.abs(R_w_zeros_FLOAT - 0) < Epsilon
    R_1_idx = np.abs(R_w_zeros_FLOAT - 1) < Epsilon

    if (C_0_idx.sum() + C_1_idx.sum() != C_w_zeros_FLOAT.size) or (R_0_idx.sum() + R_1_idx.sum() != R_w_zeros_FLOAT.size):
        raise ValueError('Cplex returned non-integer factor matrices')

    C_w_zeros = np.zeros(C_w_zeros_FLOAT.shape, dtype=int)
    R_w_zeros = np.zeros(R_w_zeros_FLOAT.shape, dtype=int)

    C_w_zeros[C_1_idx] = 1
    R_w_zeros[R_1_idx] = 1

    C, R = PostPreprocess(C_w_zeros, R_w_zeros, uniqueRowIdx, zeroRowIdx, uniqueColIdx, zeroColIdx)

    A = C[:, C.sum(axis=0) > 0]
    B = R[R.sum(axis=1) > 0, :]

    return rank, A, B


#TEST I.
for i in range(33):

    n = 17
    m = 15
    Kappa = 7
    Seed = i*17
    X_in = GenerateX(n, m, Kappa, Seed)

    rank, A, B = Get_BooleanRank(X_in)


    if rank > Kappa:
        print('wrong rank')
        break

    Z = BoolProd(A,B)

    if not np.array_equal(X_in,Z):
        print('wrong factorisation')
        break

#TEST II.
np.random.seed(0)
X_in = np.random.randint(0,2,[11,13])

rank, A, B = Get_BooleanRank(X_in)

Z = BoolProd(A, B)

if not np.array_equal(X_in, Z):
    print('wrong factorisation')
