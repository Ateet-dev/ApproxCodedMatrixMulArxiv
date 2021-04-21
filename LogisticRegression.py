import numpy as np
from OptCC import AlternateMinMultiply
from scipy.special import comb as nCr
from tensorflow import keras
from sklearn.model_selection import KFold

def softmax(x):
    z = x - np.max(x,axis=0)
    exps = np.exp(z)
    probs = exps / np.sum(exps, axis=0)
    return probs

def LR(m,k,r,filename,x_train,x_test,y_train,y_test,num_classes,seed,rng):
    '''
        Runs logistic regression using uncoded and coded multiplications simultaneously in training step
         on the given dataset.

        Input:
            m: number of splits in matrices A or B.
            k: number of nodes that do not straggle (k=N_succ)
            r: number of failures (n - N_succ)
            filename: file containing encoding and decoding matrices
            x_train,x_test,y_train,y_test: train and test sets
            num_classes: number of labels
            seed: seed for random number generators of batch picker and random failures selection.
            rng: random number generator instance for initialization of W, preferably seeded with 'seed'.

        Returns:
            results: contains accuracies.

    '''
    n = k + r
    p = nCr(n, r,exact=True)
    num_train = x_train.shape[1]
    num_test = x_test.shape[1]
    num_par = x_train.shape[0]
    with open(filename, 'rb') as f:
        alpha = np.load(f)
        beta = np.load(f)
        D = np.load(f)

    E = np.zeros((m ** 2, n))
    for i in range(n):
        E[:, i] = np.kron(alpha[:, i], beta[:, i])

    alpha = alpha.T
    beta = beta.T
    E = E.T
    D = D.T

    rng1 = np.random.RandomState()
    rng1.seed(seed)
    rng2 = np.random.RandomState()
    rng2.seed(seed)

    w = rng.randn(num_classes,num_par)
    w_a1 = w.copy()
    w_a2 = w.copy()
    w_a3 = w.copy()
    eps = np.finfo(np.float64).eps

    # print(np.linalg.norm(w))

    learning_rate = 0.001
    batch_size = 128
    niterations=1000
    print_after = 1001

    flat_eye = np.eye(m).flatten()
    Desired_Matrix = np.kron(np.ones((p, 1)), flat_eye)
    losses = np.linalg.norm(Desired_Matrix - D @ E, axis=1) ** 2
    max_loss_ind = np.argmax(losses)
    min_loss_ind = np.argmin(losses)

    # Uncomment this to run uncoded strategy with 2 failures. Ref OptCC.AlternateMinMultiply
    # min_loss_ind = np.nan

    tot_loss = np.linalg.norm(Desired_Matrix - D @ E) ** 2
    for t in range(niterations+1):
        batch = rng1.randint(0, num_train, batch_size)
        # if t==100:
            # print(np.linalg.norm(batch))
        if (t%print_after==0):
            # For intermediate printing, not required otherwise, same thing is done in else part, without loss computation.
            z = w @ x_train
            probs = softmax(z)
            loss = -np.sum(np.log(probs + eps), where=y_train) / num_train
            probs = probs[:, batch]

            z = AlternateMinMultiply(w_a1, x_train, D, alpha, beta, m, n,min_loss_ind)
            probs_a1 = softmax(z)
            loss_a1 = -np.sum(np.log(probs_a1 + eps), where=y_train) / num_train
            probs_a1 = probs_a1[:, batch]

            z = AlternateMinMultiply(w_a2, x_train, D, alpha, beta, m, n, max_loss_ind)
            probs_a2 = softmax(z)
            loss_a2 = -np.sum(np.log(probs_a2 + eps), where=y_train) / num_train
            probs_a2 = probs_a2[:, batch]

            z = AlternateMinMultiply(w_a3, x_train, D, alpha, beta, m, n, rng2.randint(0,D.shape[0],1)[0])
            probs_a3 = softmax(z)
            loss_a3 = -np.sum(np.log(probs_a3 + eps), where=y_train) / num_train
            probs_a3 = probs_a3[:, batch]

            # if (loss < 0.01):
            #     break
            # if __name__ == '__main__':
            #     print('Iteration',t,'AMM max',loss_a2,'AMM rand',loss_a3)
        else:
            # Uncoded multiplication
            z = w @ x_train[:,batch]
            probs = softmax(z)

            # Coded multiplication with min loss or uncoded multiplication with 2 failures (if min_loss_ind=nan)
            z = AlternateMinMultiply(w_a1, x_train[:, batch], D, alpha, beta, m, n, min_loss_ind)
            probs_a1 = softmax(z)

            # Coded multiplication for worst loss failure pattern
            z = AlternateMinMultiply(w_a2, x_train[:, batch], D, alpha, beta, m, n, max_loss_ind)
            probs_a2 = softmax(z)

            # Coded multiplication for randomly picked failure pattern
            z = AlternateMinMultiply(w_a3, x_train[:, batch], D, alpha, beta, m, n, rng2.randint(0,D.shape[0],1)[0])
            probs_a3 = softmax(z)

        # if (t==100):
        #     print(m,k,rng2.randint(0, D.shape[0], 1)[0])

        gradients = (probs - y_train[:, batch])@ x_train[:, batch].T
        gradients_a1 = AlternateMinMultiply(probs_a1 - y_train[:, batch], x_train[:, batch].T, D, alpha, beta, m, n, min_loss_ind)
        gradients_a2 = AlternateMinMultiply(probs_a2 - y_train[:, batch], x_train[:, batch].T, D, alpha, beta, m, n, max_loss_ind)
        gradients_a3 = AlternateMinMultiply(probs_a3 - y_train[:, batch], x_train[:, batch].T, D, alpha, beta, m, n,rng2.randint(0,D.shape[0],1)[0])

        w = w - learning_rate * gradients
        w_a1 = w_a1 - learning_rate * gradients_a1
        w_a2 = w_a2 - learning_rate * gradients_a2
        w_a3 = w_a3 - learning_rate * gradients_a3

    misclassified = np.sum(np.any((softmax(w@x_train)>0.5)^y_train,axis=0))
    accuracy_train_DM = (num_train-misclassified)/num_train*100

    misclassified = np.sum(np.any((softmax(w@x_test)>0.5)^y_test,axis=0))
    accuracy_test_DM = (num_test-misclassified)/num_test*100

    misclassified = np.sum(np.any((softmax(w_a1@x_train)>0.5)^y_train,axis=0))
    accuracy_train_min = (num_train-misclassified)/num_train*100

    misclassified = np.sum(np.any((softmax(w_a1@x_test)>0.5)^y_test,axis=0))
    accuracy_test_min = (num_test-misclassified)/num_test*100

    misclassified = np.sum(np.any((softmax(w_a2@x_train)>0.5)^y_train,axis=0))
    accuracy_train_max = (num_train-misclassified)/num_train*100

    misclassified = np.sum(np.any((softmax(w_a2@x_test)>0.5)^y_test,axis=0))
    accuracy_test_max = (num_test-misclassified)/num_test*100

    misclassified = np.sum(np.any((softmax(w_a3@x_train)>0.5)^y_train,axis=0))
    accuracy_train_rand = (num_train-misclassified)/num_train*100

    misclassified = np.sum(np.any((softmax(w_a3@x_test)>0.5)^y_test,axis=0))
    accuracy_test_rand = (num_test-misclassified)/num_test*100

    # if __name__ == '__main__':
    #     print('DM', 'Training Accuracy', accuracy_train_DM, 'Testing Accuracy', accuracy_test_DM)
    #     print('AMM min loss', 'Training Accuracy', accuracy_train_min, 'Testing Accuracy', accuracy_test_min)
    #     print('AMM max loss', 'Training Accuracy', accuracy_train_max, 'Testing Accuracy', accuracy_test_max)
    #     print('AMM random loss','Training Accuracy',accuracy_train_rand,'Testing Accuracy',accuracy_test_rand)

    results = np.array([accuracy_train_DM,accuracy_test_DM,accuracy_train_min,accuracy_test_min,accuracy_train_max,accuracy_test_max,accuracy_train_rand,accuracy_test_rand])
    return results

if __name__ == '__main__':

    # the MNIST data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))
    X = np.vstack((x_train, x_test))

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes).astype(bool)
    y_test = keras.utils.to_categorical(y_test, num_classes).astype(bool)
    y = np.vstack((y_train, y_test))

    accuracies = []
    foldno = 1

    # parameters that can be varies
    m = 5
    k = 5
    r = 2
    seed = 1
    n_splits = 10

    rng = np.random.RandomState()
    rng.seed(seed)

    # 10-fold CV
    kf = KFold(n_splits=n_splits, shuffle=True,random_state=rng)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        accuracies_fold = LR(m, k, r, 'Data/arxiv/Chebyshev_5_5_2.npy', X_train.T, X_test.T, Y_train.T, Y_test.T, num_classes,seed,rng)
        accuracies.append(accuracies_fold)

    accuracies = np.array(accuracies)
    acc_mean = np.mean(accuracies,axis=0)
    acc_std = np.std(accuracies, axis=0)
    print(acc_mean)
    print(acc_std)