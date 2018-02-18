import sys
import numpy as np
from scipy import linalg
import time

class Ridge:
    #@abstractmethod
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute_A_Xy_s(self, filename, n_samples, n_features):
        A = np.array([[0.0 for i in range(n_features+1)] for j in range(n_features+1)])
        Xy = np.array([[0.0 for i in range(1)] for j in range(n_features+1)])

        f_yX = open(filename, 'r')
        line = f_yX.readline()
        cnt = 0
        while line:
            cnt = cnt + 1
            if cnt%100000 == 0:
                print(cnt)
            words = line.split('\n')[0].split(',')
            values = np.array([float(x) for x in words if x])
            #print(values)

            values_y = values[0]
            values_X = values[1:]
            #print(values_X)
            values_X = np.append(values_X, 1.)
            #print(values_X)
            A = A + np.dot(values_X.reshape(n_features+1,1), values_X.reshape(1,n_features+1))
            Xy = Xy + np.dot(values_X.reshape(n_features+1,1), values_y.reshape(1,1))

            line = f_yX.readline()
        f_yX.close

        return A, Xy

    def compute_A_Xy_z(self, filename, n_samples, n_features):
        A = np.array([[0.0 for i in range(n_features+1)] for j in range(n_features+1)])
        Xy = np.array([[0.0 for i in range(1)] for j in range(n_features+1)])

        CONST1 = 1.0/n_samples

        f_yX = open(filename, 'r')
        line = f_yX.readline()
        cnt = 0
        while line:
            cnt = cnt + 1
            if cnt%100000 == 0:
                print(cnt)
            words = line.split('\n')[0].split(',')
            words_X = words[1:]
            values_y = float(words[0])
            values_X = np.array([int(x) for x in words_X if x])
            values_X = np.append(values_X, n_features)
            #print(values_X)

            for i in range(len(values_X)):
                ii = values_X[i]
                Xy[ii] = Xy[ii] + CONST1*values_y

                for j in range(len(values_X)):
                    jj = values_X[j]
                    A[ii,jj] = A[ii,jj] + CONST1

            line = f_yX.readline()
        f_yX.close

        return A, Xy, CONST1

    def fit(self, filename, filename_coef, n_samples, n_features):
        #A, Xy = self.compute_A_Xy_s(filename, n_samples, n_features)
        A, Xy, CONST1 = self.compute_A_Xy_z(filename, n_samples, n_features)

        mid = time.time()
        print(mid)

        for i in range(n_features+1):
            A[i][i] = A[i][i] + self.alpha*CONST1

        self.coef_ = linalg.solve(A, Xy, sym_pos=True,
                            overwrite_a=True).T

        #np.savetxt(filename_coef, self.coef_, fmt='%.3f', delimiter=',')
        self.savecoef(filename_coef)

        return self.coef_

    def savecoef(self, filename_coef):
        np.savetxt(filename_coef, self.coef_, delimiter=',')

    def loadcoef(self, filename_coef):
        self.coef_ = np.loadtxt(filename_coef, delimiter=',')

    def predict(self, filename, n_samples, n_features, filename_predict):
        f = open(filename, 'r')
        fw = open(filename_predict, 'w')

        line = f.readline()
        cnt = 0
        while line:
            cnt = cnt + 1
            if cnt%100000 == 0:
                print(cnt)
            words_X = line.split('\n')[0].split(',')
            values_X = np.array([int(x) for x in words_X if x])
            values_X = np.append(values_X, n_features)
            #print(values_X)

            value_y = 0.
            for x in values_X:
                #value_y = value_y + self.coef_[0, x]
                value_y = value_y + self.coef_[x]

            fw.write(str(value_y) + ',' + line)

            line = f.readline()

        fw.close()
        f.close()

        return 0.

def train(filename_train, filename_coef, alpha, n_samples, n_features):
    start = time.time()
    print(start)

    ridge = Ridge(alpha)
    coef = ridge.fit(filename_train, filename_coef, n_samples, n_features)

    #ridge.predict('../40_test/test_X.csv', n_samples, n_features, '../40_test/20_log_exp/predict.csv')
    print(coef)
    print(ridge.coef_)

    end = time.time()
    print(end)

    print('training time :', str(end-start))

def main():
    filename_train = sys.argv[1]
    filename_coef = sys.argv[2]
    alpha = float(sys.argv[3])
    n_samples = int(sys.argv[4])
    n_features = int(sys.argv[5])

    train(filename_train, filename_coef, alpha, n_samples, n_features)

if __name__ == '__main__':
    main()
