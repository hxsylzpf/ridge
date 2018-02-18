import ridge as r
import time
import sys

def predict(filename_testX, filename_coef, filename_predict, n_samples, n_features):
    start = time.time()
    print(start)

    alpha = 100.01
    ridge = r.Ridge(alpha)
    #coef = ridge.fit(filename_train, filename_coef, n_samples, n_features)
    ridge.loadcoef(filename_coef)
    print(ridge.coef_)
    ridge.predict(filename_testX, n_samples, n_features, filename_predict)

    end = time.time()
    print(end)

    print('training time :', str(end-start))

def main():
    filename_testX = sys.argv[1]
    filename_coef = sys.argv[2]
    filename_predict = sys.argv[3]
    n_samples = int(sys.argv[4])
    n_features = int(sys.argv[5])

    predict(filename_testX, filename_coef, filename_predict, n_samples, n_features)

if __name__ == '__main__':
    main()

