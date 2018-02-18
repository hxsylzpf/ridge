import ridge as r
import time
import sys

def train(filename_train, filename_coef, alpha, n_samples, n_features):
    start = time.time()
    print(start)

    ridge = r.Ridge(alpha)
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

