
import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy
# OWA 算子https://wenku.baidu.com/view/f0964320753231126edb6f1aff00bed5b9f373d1.html
# https://link.springer.com/chapter/10.1007/978-3-030-04663-7_3 FUZZY

class LinearWeights():
    """
    `(4/10, 3/10, 2/10, 1/10)`
    Also known as *additive* weights.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with linearly decreasing weights.
    """

    def __call__(self, k: int):
        return np.flip(2 * np.arange(1, k + 1) / (k * (k + 1)))

class ExponentialWeights():
    """
    `(8/15, 4/15, 2/15, 1/15)`
    Exponentially decreasing weights with parametrisable base.
    Parameters
    ----------
    base: float
        Exponential base. Should be larger than 1.
    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with exponentially decreasing weights with base `b`.
    Notes
    -----
    With base 2, weights rapidly approach 0, meaning:
    - the resulting weight vector is not very useful, and quickly becomes insensitive to increasing `k`,
    - using large values for `k` will produce weights that are so small as to cause computational wonkiness.
    These issues are exacerbated for larger bases, so bases only slightly larger than 1 may be most useful.
    """

    base: float = 2

    def __call__(self, k: int):
        w = np.flip(self.base ** np.arange(k))
        return w / np.sum(w)


class ReciprocallyLinearWeights():
    """
    W invadd L
    `(12/25, 12/50, 12/75, 12/100)`
    Also known as *inverse additive* weights.

    Returns
    -------
    f: int -> np.array
        Function that takes a positive integer `k` and returns a weight vector of length `k`
        with reciprocally linearly decreasing weights.
    """

    def __call__(self, k: int):
        return 1 / (np.arange(1, k + 1) * np.sum(1 / np.arange(1, k + 1)))


class MinkowskiSize:
    """
    Family of vector size measures of the form
    `(x1**p + x2**p + ... + xm**p)**(1/p)` (if `unrooted = False`), or
    `(x1**p + x2**p + ... + xm**p)` (if `unrooted = True`),
    for `0 < p < ∞`, and their limits in 0 and ∞.

    For `p = 0`, the rooted variant evaluates to ∞ if there is more than one non-zero coefficient,
    to 0 if all coefficients are zero, and to the only non-zero coefficient otherwise.
    The unrooted variant is equal to the number of non-zero coefficients.

    For `p = ∞`, the rooted variant is the maximum of all coefficients.
    The unrooted variant evaluates to ∞ if there is at least one coefficient larger than 1,
    and to the number of coefficients equal to 1 otherwise.

    Parameters
    ----------
    p: float = 1
        Exponent to use. Must be in `[0, ∞]`.

    unrooted: bool = False
        Whether to omit the root `**(1/p)` from the formula.
        For `p = 0`, this gives Hamming size.
        For `p = 2`, this gives squared Euclidean size.

    scale_by_dimensionality: bool = False
        If `True`, values are scaled linearly such that the vector `[1, 1, ..., 1]` has size 1.
        This can be used to ensure that the range of dissimilarity values in the unit hypercube is `[0, 1]`,
        which can be useful when working with features scaled to `[0, 1]`.

    Notes
    -----
    The most used parameter combinations have their own name.

    * Hamming size is unrooted `p = 0`.
    * The Boscovich norm is `p = 1`. Also known as cityblock, Manhattan or Taxicab norm.
    * The Euclidean norm is rooted `p = 2`. Also known as Pythagorean norm.
    * Squared Euclidean size is unrooted `p = 2`.
    * The Chebishev norm is rooted `p = ∞`. Also known as chessboard or maximum norm.
    """

    p: float
    unrooted: bool = False
    scale_by_dimensionality: bool = False

    def __post_init__(self):
        if self.p < 0:
            raise ValueError('`p` must be in `[0, ∞]`')

    def __call__(self, u, axis=-1):
        if self.p == 0:
            if self.unrooted:
                result = np.count_nonzero(u, axis=axis)
            else:
                result = np.where(np.count_nonzero(u, axis=axis) <= 1, np.sum(np.abs(u), axis=axis), np.inf)
        elif self.p == 1:
            result = np.sum(np.abs(u), axis=axis)
        elif self.p == np.inf:
            if self.unrooted:
                result = np.sum(np.where(np.abs(u) < 1, 0, np.where(np.abs(u) > 1, np.inf, 1)), axis=axis)
            else:
                result = np.max(u, axis=axis)
        else:
            result = np.sum(np.abs(u) ** self.p, axis=axis)
            if not self.unrooted:
                result = result**(1/self.p)
        if self.scale_by_dimensionality and self.p < np.inf:
            if self.unrooted:
                result = result / u.shape[axis]
            else:
                result = result / (u.shape[axis]**(1/self.p))
        return result

from typing import Callable

def resolve_kresolve_k(k: float or Callable[[int], float] or None, n: int, k_max: int = None):
    """
    Helper method to obtain a valid number of neighbours
    from a parameter `k` given `n` target records,
    where `k` may be defined in terms of `n`.

    Parameters
    ----------
    k: float or (int -> float) or None
        Parameter value to resolve. Can be a float,
        a callable that takes `n` and returns a float,
        or None.

    n: int
        The input for `k` if `k` is callable.

    k_max: int = None
        The maximum allowed value for `k`.
        If None, this is equal to `n`.

    Returns
    -------
    k: int
       If `k` is a float in [1, k_max]: `k`;
       If `k` is None: `k_max`;
       If `k` is callable, the output of `k` applied to `n`,
       rounded to the nearest integer in `[1, k_max]`.

    Raises
    ------
    ValueError
        If `k` is a float not in [1, k_max].

    """
    if k_max is None:
        k_max = n
    if callable(k):
        k = k(n)
    elif k is None:
        k = k_max
    elif not 1 <= k <= k_max:
        raise ValueError(f'{k} is too many nearest neighbours, number has to be between 1 and {k_max}.')
    return min(max(1, round(k)), k_max)


def _weighted_mean(a, weights, axis, type):
    if weights is None:
        return np.take(a, -1, axis=axis)
    w = weights(a.shape[axis])
    w = np.reshape(w, [-1] + ((len(a.shape) - axis - 1) % len(a.shape)) * [1])
    if type == 'arithmetic':
        return np.sum(w * a, axis=axis)
    if type == 'geometric':
        return np.exp(np.sum(w * np.log(a), axis=axis))
    if type == 'harmonic':
        return 1 / np.sum(w / a, axis=axis)


def greatest(a, k: int, axis: int = -1):
    """
    Returns the `k` greatest values of `a` along the specified axis, in order.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k: int
        Number of values to return.
        Should be a positive integer not larger than `a` along `axis`.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    greatest_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if k == a.shape[axis]:
        return np.flip(np.sort(a, axis=axis), axis=axis)
    a = np.partition(a, -k, axis=axis)
    take_this = np.arange(-k % a.shape[axis], a.shape[axis])
    a = np.take(a, take_this, axis=axis)
    a = np.flip(np.sort(a, axis=axis), axis=axis)
    return a


def soft_max(a, weights, k: int or None, axis=-1, type: str = 'arithmetic'):
    r"""
    Calculates the soft maximum of an array.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    weights : (k -> np.array) or None
        Weights to apply to the `k` selected values. If None, the `k`\ th value is returned.

    k: int or None
        Number of greatest values from which the soft maximum is calculated.
        Should be either a positive integer not larger than `a` along `axis`,
        or None, which is interpreted as the size of `a` along `axis`.

    axis : int, default=-1
        The axis along which the soft maximum is calculated.

    type : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
        Determines the type of weighted average.

    Returns
    -------
    soft_max_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed. If `a` is a 0-d array, a scalar is returned.
    """
    if k is None:
        k = a.shape[axis]
    a = greatest(a, k, axis=axis)
    return _weighted_mean(a, weights, axis=axis, type=type)




def least(a, k: int, axis: int = -1):
    """
    Returns the `k` least values of `a` along the specified axis, in order.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    k: int
        Number of values to return.
        Should be a positive integer not larger than `a` along `axis`.

    axis : int, default=-1
        The axis along which values are selected.

    Returns
    -------
    least_along_axis : ndarray
        An array with the same shape as `a`, with the specified axis reduced according to the value of `k`.
    """
    if k == a.shape[axis]:
        return np.sort(a, axis=axis)
    a = np.partition(a, k - 1, axis=axis)
    take_this = np.arange(k)
    a = np.take(a, take_this, axis=axis)
    a = np.sort(a, axis=axis)
    return a

def soft_min(a, weights, k: int or None, axis=-1, type: str = 'arithmetic'):
    r"""
    Calculates the soft minimum of an array.

    Parameters
    ----------
    a : ndarray
        Input array of values.

    weights : (k -> np.array) or None
        Weights to apply to the `k` selected values. If None, the `k`\ th value is returned.

    k: int or None
        Number of least values from which the soft minimum is calculated.
        Should be either a positive integer not larger than `a` along `axis`,
        or None, which is interpreted as the size of `a` along `axis`.

    axis : int, default=-1
        The axis along which the soft minimum is calculated.

    type : str {'arithmetic', 'geometric', 'harmonic', }, default='arithmetic'
        Determines the type of weighted average.

    Returns
    -------
    soft_min_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed. If `a` is a 0-d array, a scalar is returned.
    """
    if k is None:
        k = a.shape[axis]
    a = least(a, k, axis=axis)
    return _weighted_mean(a, weights, axis=axis, type=type)


def _Q_1(neighbours, R, R_d, owa_weights):
    # (neighbours_min, R_min, self.model_R_d_min,self.owa_weights)
    tmp2 = R_d[neighbours, :]
    tmp = R[..., None]

    vals = np.minimum(1 - R[..., None] + R_d[neighbours, :] - 1, 1)
    #vals = np.minimum(1 - R[..., None] + R_d[neighbours, :] - 1, 1)

    # R 是距离越近越小
    # 所以x越相似 vals 越大
    # y越相似 vals越大
    return soft_min(vals, owa_weights, k=None, axis=1)


def _Q_2(neighbours, R, R_d, owa_weights):
    vals = np.maximum(R[..., None] + R_d[neighbours, :] - 1, 0)
    # x越相似 val 越小
    # y越相似 val 越大
    return soft_max(vals, owa_weights, k=None, axis=1)


def _Q_11(neighbours, R, R_d, owa_weights):
    # (neighbours_min, R_min, self.model_R_d_min,self.owa_weights)
    tmp2 = R_d[neighbours, :]
    tmp = R[..., None]

    vals = np.minimum(1 - R[..., None] + R_d[neighbours, :] - 1, 1)
    #vals = np.minimum(1 - R[..., None] + R_d[neighbours, :] - 1, 1)

    # R 是距离越近越小
    # 所以x越相似 vals 越大
    # y越相似 vals越大
    return soft_min(vals, owa_weights, k=None, axis=0)


def _Q_22(neighbours, R, R_d, owa_weights):
    vals = np.maximum(R[..., None] + R_d[neighbours, :] - 1, 0)
    # x越相似 val 越小
    # y越相似 val 越大
    return soft_max(vals, owa_weights, k=None, axis=0)

def _Q_3(neighbours, R, R_d, owa_weights, Y):
    tmp = R[..., None]
    vals = np.minimum(1 - R[..., None] + R_d[neighbours, :] - 1, 1)
    # R 是距离越近越小
    # 所以x越相似 vals 越大
    # y越相似 vals越大
    return soft_min(vals, owa_weights, k=None, axis=1)

def Q_sum(Q, Y):
    t0 = Q[..., None]
    tt_sum = 0
    t11 = []
    t12 = []
    for t0_i in t0:
        indexes = np.where(np.array(t0_i) == True)[0]
        t_index = Y[indexes]
        t_sum = np.sum(t_index, axis=0)
        t11.append(t_sum)
        #tt_sum += t_sum
        t12.append(t_sum / indexes.size )

    #t1 = np.minimum(Y, t0)
    #t11 = np.sum(t1)

    t2 = np.sum(Q, axis=-1, keepdims=True)
    t3 = t11 / t2
    return t3


class FRONEC():
    def __init__(self, Q_type = 2, R_d_type = 1,
                 k = 10, owa_weights = ReciprocallyLinearWeights(),
                 dissimilarity = 'boscovich'):
        self.Q_type = Q_type
        self.R_d_type = R_d_type
        self.k = k
        self.owa_weights = owa_weights
        #self.dissimilarity = MinkowskiSize(p=1, unrooted = False, scale_by_dimensionality=True)

    def _R_d_1(self, Y):
        tmp = Y[:, None, :]
        tmp2 = (tmp == Y) # 遍历所有样本 看是否与当前样本的label一致
        tmp3 = np.sum(tmp2, axis=-1)  # 相加  哦？ 那这个就是 label 相似性吧！ 不用管 对应式子9
        return np.sum(Y[:, None, :] == Y, axis=-1)

    def _R_d_2(self, Y):
        p = np.sum(Y, axis=0) / len(Y)

        both = np.minimum(Y[:, None, :], Y)
        either = np.maximum(Y[:, None, :], Y)
        neither = 1 - either
        xeither = either - both

        numerator = both * (1 - p) + neither * p
        divisor = numerator + xeither * 0.5
        return np.sum(numerator, axis=-1) / np.sum(divisor, axis=-1)

    def _R_d_3(self, Y):
        p = np.sum(Y, axis=0) / len(Y)

        numeratorInfo = {}
        divisorInfo = {}
        for i in range(0, len(Y)):
            numeratorInfo[i] = {}
            divisorInfo[i] = {}

        for i in range(0, len(Y) - 1):
            for j in range(i+1, len(Y)):
                t1 = Y[i]
                t2 = Y[j]

                both = np.minimum(t1, t2)
                either = np.maximum(t1,t2)
                neither = 1 - either
                xeither = either - both

                numerator = both * (1 - p) + neither * p  # 只需要这两个
                divisor = numerator + xeither * 0.5
                numeratorInfo[i][j] =  np.sum(numerator)
                numeratorInfo[j][i] = np.sum(numerator)

                divisorInfo[i][j] = np.sum(divisor)
                divisorInfo[j][i] = np.sum(divisor)
        result = []

        for i in range(0, len(Y)):
            numeratorX = numeratorInfo[i]
            divisorX = divisorInfo[i]
            t = np.zeros(len(Y))
            t[i] = 1.
            for k in numeratorX.keys():
                numeratorY = numeratorX[k]
                divisorY = divisorX[k]
                t[k] = round(numeratorY / divisorY, 8)
            result.append(t)

        return np.array(result)

    def _R_d_4(self, Y, p_max, Y_max):
        p = np.sum(Y, axis=0) / len(Y) #here Y is min probabily
        #p += p_max
        nYmin = len(Y); nYmax = len(Y_max)
        nYmin1 = np.sum(Y, axis=0); nYmax2 = np.sum(Y_max, axis=0); nY = np.sum(self.Y, axis=0)
        pAll =  np.sum(self.Y, axis=0) / len(self.Y)
        nL = np.sum([nYmin1, nYmax2], axis=0).tolist() # 每个label总共有几个

        b = []
        for i in range(len(nL)):
            if nL[i] == 0:
                continue
            else:
                b.append(nL[i])
        mini = np.min(b)  # 最小值
        for i in range(len(nL)):
            if nL[i] == 0:
                nL[i] = mini
                print (str(i) + ' has been modified')

        t1 = [a/b for a,b in zip(nYmin1,nL)]
        #t2 = [a/b for a,b in zip(nYmax2,nL)] # 对应part去除
        #p = (nYmin / (nYmin + nYmax)) * p_max -  (nYmax / (nYmin + nYmax)) * p
        #t11 = [a*b for a,b in zip(t2,p_max)]
        #t22 = [a*b for a,b in zip(t1,p)]
        #p = np.array([a-b for a,b in zip(t11,t22)])
        #p = np.array([a-b for a,b in zip(t2,t1)])

        t111 = [a*b for a,b in zip(t1,pAll)]
        #t222 = [a*b for a,b in zip(t2,pAll)]
        #p = pAll
        #p = np.array([a-b for a,b in zip(t222,t111)])
        p = np.array([a - b for a, b in zip(pAll, t111)])

        #p /= 2
        numeratorInfo = {}
        divisorInfo = {}
        for i in range(0, len(Y)):
            numeratorInfo[i] = {}
            divisorInfo[i] = {}

        for i in range(0, len(Y) - 1):
            for j in range(i+1, len(Y)):
                t1 = Y[i]
                t2 = Y[j]

                both = np.minimum(t1, t2)
                either = np.maximum(t1,t2)
                neither = 1 - either
                xeither = either - both

                numerator = both * (1 - p) + neither * p  # 只需要这两个
                divisor = numerator + xeither * 0.5
                numeratorInfo[i][j] =  np.sum(numerator)
                numeratorInfo[j][i] = np.sum(numerator)

                divisorInfo[i][j] = np.sum(divisor)
                divisorInfo[j][i] = np.sum(divisor)
        result = []

        for i in range(0, len(Y)):
            numeratorX = numeratorInfo[i]
            divisorX = divisorInfo[i]
            t = np.zeros(len(Y))
            t[i] = 1.
            for k in numeratorX.keys():
                numeratorY = numeratorX[k]
                divisorY = divisorX[k]
                t[k] = round(numeratorY / divisorY, 8)
            result.append(t)

        return np.array(result)  # samples * sampels 即 当前样本 和 所有其他样本的label相似度

    def _R_d_5(self, Y, p_max):

        #p = pAll
        p = np.sum(Y, axis=0) / len(Y) #here Y is min probabily

        #p /= 2
        numeratorInfo = {}
        divisorInfo = {}
        for i in range(0, len(Y)):
            numeratorInfo[i] = {}
            divisorInfo[i] = {}

        for i in range(0, len(Y) - 1):
            for j in range(i+1, len(Y)):
                t1 = Y[i]
                t2 = Y[j]

                both = np.minimum(t1, t2)
                either = np.maximum(t1,t2)
                neither = 1 - either
                xeither = either - both

                numerator = both * (1 - p) + neither * p  # 只需要这两个
                divisor = numerator + xeither * 0.5
                numeratorInfo[i][j] =  np.sum(numerator)
                numeratorInfo[j][i] = np.sum(numerator)

                divisorInfo[i][j] = np.sum(divisor)
                divisorInfo[j][i] = np.sum(divisor)
        result = []

        for i in range(0, len(Y)):
            numeratorX = numeratorInfo[i]
            divisorX = divisorInfo[i]
            t = np.zeros(len(Y))
            t[i] = 1.
            for k in numeratorX.keys():
                numeratorY = numeratorX[k]
                divisorY = divisorX[k]
                t[k] = round(numeratorY / divisorY, 8)
            result.append(t)

        return np.array(result)  # samples * sampels 即 当前样本 和 所有其他样本的label相似度


    def construct_minmax3(self, X, Y, min_set):
        #self.model_R_d = self._R_d_2(Y)
        self.Y = Y

        self.nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(X)  # 若数据少一些 是否可以满足 0102
        z = 1
        self.X_min = [X[i]for i in min_set]
        self.Y_min = np.array([Y[i]for i in min_set])

        #self.X_min = X
        #self.Y_min = Y

        #self.X_max = [X[i] for i in range(0, len(X)) if i not in min_set]
        self.Y_max = np.array([Y[i] for i in range(0, len(Y)) if i not in min_set])
        p_max = np.sum(self.Y_max, axis=0) / len(self.Y_max)
        p_min = np.sum(self.Y_min, axis=0) / len(self.Y_min)

        #print ('p_max ' + str(len(self.Y_max)))
        #print ('p_min ' + str(len(self.Y_min)))

        self.model_R_d_min = self._R_d_4(self.Y_min, p_max, self.Y_max)
        #t2 = self._R_d_2(self.Y_min)
        self.nbrs_min = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(self.X_min)  # 若数据少一些 是否可以满足 0102

        z = 1

    def _query_1(self, X):
        distances_min, neighbours_min = self.nbrs_min.kneighbors(X)
        #R_min = np.maximum(1 - distances_min, 0) # 样本距离的相反数 距离越近 R越小
        R_min = np.maximum(1 - distances_min, 0) # 样本距离的相反数 距离越近 R越小 #todo 在这里进行cluster 加权

        Q_min = _Q_1(neighbours_min, R_min, self.model_R_d_min,self.owa_weights) # 和其它样本的相似性 L

        Q_min_2 = _Q_2(neighbours_min, R_min, self.model_R_d_min,self.owa_weights) # 和其它样本的相似性 U
        Q_min = (Q_min + Q_min_2) / 2.

        Q_max_min = np.max(Q_min, axis=-1, keepdims=True) # 最大的那个值 L中最大的值


        Q_min = Q_min >= Q_max_min

        tmp_min = Q_sum(Q_min, self.Y_min)
        tmp = tmp_min  / 1.0
        return tmp

    def _query_2(self, X, x_depence, min_to_allindx, label_cluster_min_matrix):
        distances_min, neighbours_min = self.nbrs_min.kneighbors(X)
        distances_min_new_list = []
        neighbours_min_new_list = []
        R_min = []
        for i in range(0, len(x_depence)):
            sample_be_generated = x_depence[i]
            distances_min_new = []
            neighbours_min_new = []
            if sample_be_generated in label_cluster_min_matrix:
                neighbours_min_ori = neighbours_min[i]  # 邻居是哪个
                distances_min_ori = distances_min[i]

                depend_labels = label_cluster_min_matrix[sample_be_generated] # 在min里面邻居是哪个
                #for j in range(0, len(neighbours_min[i])):
                nei_j = neighbours_min[i]

                for nn in range(0, len(nei_j)):
                    if nei_j[nn] in depend_labels:
                        neighbours_min_new.append(nei_j[nn])
                        distances_min_new.append(distances_min_ori[nn])

                if len(neighbours_min_ori) == 0:
                    distances_min_new = distances_min[i]
                    neighbours_min_new = neighbours_min[i]
            else:
                distances_min_new = distances_min[i]
                neighbours_min_new = neighbours_min[i]

            distances_min_new_list.append(distances_min_new)
            neighbours_min_new_list.append(neighbours_min_new)
            Rss = np.maximum(1 - np.array(distances_min_new), 0)
            R_min.append(Rss)

        # need modify neighbours_min R_min
        distances_min = distances_min_new_list
        neighbours_min = neighbours_min_new_list
        Q_min = []
        Q_min_2 = []
        Q_minS = []
        # todo why not line by line first
        for i in range(0, len(X)):
            Q_min_tmp = _Q_11(neighbours_min[i], R_min[i], self.model_R_d_min, self.owa_weights)
            Q_min_2_tmp = _Q_22(neighbours_min[i], R_min[i], self.model_R_d_min, self.owa_weights)
            Q_min = (Q_min_tmp + Q_min_2_tmp) / 2.
            Q_max_min = np.max(Q_min, axis=-1, keepdims=True)
            Q_min = Q_min >= Q_max_min
            Q_minS.append(Q_min)
        #Q_min = _Q_1(neighbours_min, R_min, self.model_R_d_min,self.owa_weights) # 和其它样本的相似性 L

        #Q_min_2 = _Q_2(neighbours_min, R_min, self.model_R_d_min,self.owa_weights) # 和其它样本的相似性 U
        #Q_min = (Q_min + Q_min_2) / 2.

        #Q_max_min = np.max(Q_min, axis=-1, keepdims=True) # 最大的那个值 L中最大的值


        #Q_min = Q_min >= Q_max_min

        tmp_min = Q_sum(np.array(Q_minS), self.Y_min)
        tmp = tmp_min  / 1.0
        return tmp

