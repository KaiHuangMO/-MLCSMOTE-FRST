from skmultilearn.dataset import load_dataset
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
# instantiate the classifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import f1_score, accuracy_score
from skmultilearn.adapt import MLkNN

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import RakelD
from FRSLR import FR3
from LIFTO import LIFT_SAMPLES


testDataset = ['emotions']
def example_accuracy(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_or = np.sum(np.logical_or(gt, predict), axis=1).astype("float32")

    return np.mean(ex_and / (ex_or+0.0001))

for par in range(5, 30, 5):

    for data in testDataset:
        print ('start' + data)
        X, y, _, _ = load_dataset(data, 'train')
        X_testO, y_test, _, _ = load_dataset(data, 'test')

        X = X.toarray()
        y = y.toarray()

        X_testO = X_testO.toarray()
        y_test = y_test.toarray()
        #exit()
        from sklearn.decomposition import PCA
        print("X.shape: " + str((X.shape)))


        print("X_test.shape: " + str((X_testO.shape)))
        print("y.shape: " + str((y.shape)))
        print("Ratio:" + str(int(X.shape[0])/int(X_testO.shape[0])))
        import copy

        macroF1 = {}
        microF1 = {}

        rocs = {}
        aps = {}

        macroF1_C = {}
        microF1_C = {}

        rocs_C = {}
        aps_C = {}

        for jj in range(0, 5):
            print (jj)
            random.seed(jj) # 设置随机种子

            X_mlsmote, Y_mlsmote, y_new_ori, min_set = LIFT_SAMPLES(X,y,X_testO,y_test, perc_gen_instances = .25, K=par, r = .1)
            Y_mlsmote = np.array(Y_mlsmote)


            for kk in range(0, 4):
                br_clf = MLkNN()
                if kk == 0:
                    br_clf = MLkNN()
                if kk == 1:
                    br_clf = BinaryRelevance(
                        classifier=RandomForestClassifier(),
                        require_dense=[False, True]
                    )
                if kk == 2:
                    br_clf = RakelD(
                        base_classifier=RandomForestClassifier(),
                        base_classifier_require_dense=[True, True]
                    )
                    # continue
                if kk == 3:
                    br_clf = ClassifierChain(
                        classifier=RandomForestClassifier(),
                        require_dense=[False, True]
                    )
                X_mlsmote = np.array(X_mlsmote)
                Y_mlsmote = np.array(Y_mlsmote)
                br_clf.fit(X_mlsmote, Y_mlsmote)
                y_pred = br_clf.predict(X_testO)  # .toarray()
                y_t = y_test

                target_names = []
                for k in range(0, len(y_test[0])):
                    target_names.append(str(k))

                if kk not in macroF1:
                    macroF1_C[kk] = [f1_score(y_test, y_pred, average='macro')]
                    microF1_C[kk] = [f1_score(y_test, y_pred, average='micro')]
                    rocs_C[kk] = [example_accuracy(y_test, y_pred.toarray())]

                else:
                    v1 = macroF1_C[kk]; v1.append(f1_score(y_test, y_pred, average='macro')) ; macroF1_C[kk] = v1
                    v2 = microF1_C[kk]; v2.append(f1_score(y_test, y_pred, average='micro')) ; microF1_C[kk] = v2
                    v3 = rocs_C[kk];v3.append(example_accuracy(y_test, y_pred.toarray()));rocs_C[kk] = v3

            X_mix2, Y_mix2 = FR3(X_mlsmote, Y_mlsmote, y_new_ori, min_set, copy.deepcopy(X), copy.deepcopy(y))
            X_mlsmote = X_mix2
            Y_mlsmote = Y_mix2
            for kk in range(0, 4):
                br_clf = MLkNN()
                if kk == 0:
                    br_clf = MLkNN()
                if kk == 1:
                    br_clf = BinaryRelevance(
                        classifier=RandomForestClassifier(),
                        require_dense=[False, True]
                    )
                if kk == 2:
                    br_clf = RakelD(
                        base_classifier=RandomForestClassifier(),
                        base_classifier_require_dense=[True, True]
                    )
                    # continue
                if kk == 3:
                    br_clf = ClassifierChain(
                        classifier=RandomForestClassifier(),
                        require_dense=[False, True]
                    )
                X_mlsmote = np.array(X_mlsmote)
                Y_mlsmote = np.array(Y_mlsmote)
                br_clf.fit(X_mlsmote, Y_mlsmote)
                # predict
                y_pred = br_clf.predict(X_testO)  # .toarray()
                # predict
                y_t = y_test

                target_names = []
                for k in range(0, len(y_test[0])):
                    target_names.append(str(k))
                if kk not in macroF1:
                    macroF1[kk] = [f1_score(y_test, y_pred, average='macro')]
                    microF1[kk] = [f1_score(y_test, y_pred, average='micro')]
                    rocs[kk] = [example_accuracy(y_test, y_pred.toarray())]

                else:
                    v1 = macroF1[kk]; v1.append(f1_score(y_test, y_pred, average='macro')) ; macroF1[kk] = v1
                    v2 = microF1[kk]; v2.append(f1_score(y_test, y_pred, average='micro')) ; microF1[kk] = v2
                    v3 = rocs[kk];v3.append(example_accuracy(y_test, y_pred.toarray()));rocs[kk] = v3
        print ('para' + str(par))
        for kk in range(0, 4):
            print(kk)
            print('macroF1_C std ' + str(np.round(np.mean(macroF1_C[kk]), 4)) + ' ' + str(np.round(np.std(macroF1_C[kk]), 4)))
            print('microF1_C std ' + str(np.round(np.mean(microF1_C[kk]), 4)) + ' ' + str(np.round(np.std(microF1_C[kk]), 4)))
            print ('hamming_C std '  + str(np.round(np.mean(rocs_C[kk]), 4)) + ' ' + str(np.round(np.std(rocs_C[kk]), 4)))


        for kk in range(0, 4):
            print(kk)
            print('macroF1 std ' + str(np.round(np.mean(macroF1[kk]), 4)) + ' ' + str(np.round(np.std(macroF1[kk]), 4)))
            print('microF1 std ' + str(np.round(np.mean(microF1[kk]), 4)) + ' ' + str(np.round(np.std(microF1[kk]), 4)))
            print ('hamming std '  + str(np.round(np.mean(rocs[kk]), 4)) + ' ' + str(np.round(np.std(rocs[kk]), 4)))
