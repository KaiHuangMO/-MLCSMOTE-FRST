
from fronec import FRONEC
import numpy as np


def FR3(X_mlsmote, Y_mlsmote, y_new_ori, min_set, X_train, y_train,threshold=.5, knei=10):
    # COMPUTE MEAN IR CVIR HERE
    import copy
    X_mlsmote1 = copy.deepcopy(X_mlsmote)
    Y_mlsmote1 = copy.deepcopy(Y_mlsmote)

    XF = copy.deepcopy(X_train)
    YF = copy.deepcopy(y_train)

    X_mlsmoteOri = X_mlsmote[len(XF):]
    Y_mlsmoteOri = Y_mlsmote[len(XF):]

    print('min_set' + str(len(min_set)))

    cls = FRONEC(k=knei)
    cls.construct_minmax3(XF, YF, min_set)

    result = cls._query_1(X_mlsmote[len(XF):])
    X_mlsmote = X_mlsmoteOri

    Y_mlsmote_fronex = []
    import copy
    for i in range(0, len(X_mlsmote)):
        ythis = result[i]

        Y_mlsmote_this = list(np.zeros(len(YF[0])))
        Y_mlsmote_this2 = copy.deepcopy(y_new_ori[i])  # 生成依赖的标签

        count = 0
        for j in ythis:

            if j >= threshold :
                Y_mlsmote_this[count] = 1
            elif Y_mlsmote_this2[count] == 1: # and Y_mlsmote_this2[count] == 1:
                Y_mlsmote_this[count] = 1
            else:
                Y_mlsmote_this[count] = 0

            count += 1

        if np.max(Y_mlsmote_this) == 0:  # 如果啥都没有
            Y_mlsmote_this = y_new_ori[i]
        Y_mlsmote_fronex.append(Y_mlsmote_this)

    X_mix2 = np.concatenate([X_mlsmote, XF], axis=0)
    Y_mix2 = np.concatenate([Y_mlsmote_fronex, YF], axis=0)
    return X_mix2, Y_mix2

