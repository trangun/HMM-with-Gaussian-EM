#!/usr/bin/env python3
import numpy as np

if not __file__.endswith('_hmm_gaussian.py'):
    print(
        'ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs446/data/em/"  # TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)


def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9 * len(data))
    train_xs = np.asarray(data[:dev_cutoff], dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:], dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs


def init_model(args):
    K = args.cluster_num
    seed = 10000
    num_dim = 2
    if args.cluster_num:
        mus = np.zeros((K, 2))
        np.random.seed(seed)
        mus = np.random.rand(K, 2)
        if not args.tied:
            sigmas = np.array([np.eye(num_dim) for i in range(args.cluster_num)])
        else:
            # sigmas = np.zeros((2,2))
            # Returns a 2-D array with ones on the diagonal and zeros elsewhere.
            sigmas = np.eye(num_dim)

        transitions = np.zeros([args.cluster_num, args.cluster_num])  # transitions[i][j] = probability of moving from cluster i to cluster j
        transitions = np.ones((K, K))
        transitions = transitions/np.sum(transitions, 1)[None].T
        initials = np.ones((K,1))/K  # probability for starting in each state

        # TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        # raise NotImplementedError #remove when random initialization is implemented
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file, 'r') as f:
            for line in f:
                # each line is a cluster, and looks like this:
                # initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float, line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5], vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    # TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    class Model():
        def __init__(self):
            self.mus = mus
            self.sigmas = sigmas
            self.initials = initials
            self.transitions = transitions
            self.alphas = None
            self.betas = None
            self.gamma = None
            self.xi = None
            self.likelihood = 0.0

        def expectation(self, data, args):
            from scipy.stats import multivariate_normal
            K = args.cluster_num
            N = len(data)
            self.alphas, self.likelihood = forward(self, data, args)
            self.betas = backward(self, data, args)
            gamma = np.zeros([N, K])
            xi = np.zeros([N, K, K])
            for i in range(K):
                gamma[:, i] = np.multiply(self.alphas[:, i], self.betas[:, i]) / np.sum(
                    np.multiply(self.alphas, self.betas), axis=1)

            for t in range(1, N):
                for i in range(K):
                    for j in range(K):
                        if args.tied:
                            xi[t, i, j] = (1/self.likelihood[t]) * self.alphas[t-1, i][None].T.dot(self.betas[t, j][None])*\
                                          self.transitions[i, j] * multivariate_normal.pdf(
                                data[t], mean=self.mus[j], cov=self.sigmas)
                        else:
                            xi[t, i, j] = (1/self.likelihood[t]) * self.alphas[t-1, i][None].T.dot(self.betas[t, j][None])*\
                                          self.transitions[i, j] * multivariate_normal.pdf(
                                data[t], mean=self.mus[j], cov=self.sigmas[j])

            self.gamma = gamma
            self.xi = xi
            pass

        def maximization(self, data, args):
            K = args.cluster_num
            N = len(data)
            self.initials = self.gamma[0, :]
            temp_transitions = np.sum(self.xi[1:, :, :], axis=0)
            self.transitions = temp_transitions/np.sum(temp_transitions, axis=1)[None].T

            for t in range(N):
                for i in range(K):

                    # update mu
                    self.mus[i] = self.gamma[:, i].T @ data / np.sum(self.gamma[:, i])

                    deviation = data[:] - self.mus[i]

                    if args.tied:
                        self.sigmas += np.dot(np.multiply(self.gamma[:, i].reshape(-1, 1), deviation).T,
                                              deviation) / np.sum(self.gamma[:, i]) / K
                    else:
                        self.sigmas[i] = np.dot(np.multiply(self.gamma[:, i].reshape(-1, 1), deviation).T,
                                                deviation) / np.sum(self.gamma[:, i])


    model = Model()
    # raise NotImplementedError #remove when model initialization is implemented
    return model


def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log

    # TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    # NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value,
    # and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going
    # to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different
    # than what's in the notes). This was discussed in class on April 3rd.
    # raise NotImplementedError
    K = args.cluster_num
    N = len(data)
    alphas = np.zeros((N, K))
    likelihood = np.zeros(N)

    for k in range(K):
        if args.tied:
            alphas[0, k] = model.initials[k] * multivariate_normal.pdf(
                data[0, :], mean=model.mus[k], cov=model.sigmas)
        else:
            alphas[0, k] = model.initials[k] * multivariate_normal.pdf(
                data[0, :], mean=model.mus[k], cov=model.sigmas[k])
    likelihood[0] = np.sum(alphas[0, :])
    alphas[0, :] /= likelihood[0]

    for t in range(1, N):
        for i in range(K):
            for j in range(K):
                alphas[t, i] += alphas[t - 1, j]* model.transitions[j, i]
            if args.tied:
                alphas[t, i] *= multivariate_normal.pdf(data[t], mean=model.mus[i], cov=model.sigmas)
            else:
                alphas[t, i] *= multivariate_normal.pdf(data[t], mean=model.mus[i], cov=model.sigmas[i])
        likelihood[t] = np.sum(alphas[t, :])
        alphas[t, :] /= likelihood[t]

    return alphas, likelihood


def backward(model, data, args):
    from scipy.stats import multivariate_normal

    K = args.cluster_num
    N = len(data)
    betas = np.zeros((N, K))
    # TODO: Calculate and return backward probabilities (normalized like in forward before)
    # raise NotImplementedError
    ll = model.likelihood
    betas[N - 1, :] = 1

    for t in reversed(range(N - 1)):
        for i in range(K):
            temp_betas = 0
            for j in range(K):
                if args.tied:
                    temp_betas += model.transitions[i, j] * betas[t + 1, j] * multivariate_normal.pdf(
                        data[t + 1], mean=model.mus[j], cov=model.sigmas)
                else:
                    temp_betas += model.transitions[i, j] * betas[t + 1, j] * multivariate_normal.pdf(
                        data[t + 1], mean=model.mus[j], cov=model.sigmas[j])
            betas[t, i] = temp_betas
        betas[t, :] /= ll[t+1]

    return betas


def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    # TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    # raise NotImplementedError #remove when model training is implemented

    while args.iterations:
        model.expectation(train_xs, args)
        model.maximization(train_xs, args)
        dev_ll = float("inf")  # log likelihood
        train_ll = float("inf")
        if not args.nodev:
            if dev_ll < average_log_likelihood(model, dev_xs, args):
                dev_ll = average_log_likelihood(model, dev_xs, args)
        else:
            train_ll = average_log_likelihood(model, train_xs, args)
        args.iterations -= 1

    return model


def average_log_likelihood(model, data, args):
    # TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    # NOTE: yes, this is very simple, because you did most of the work in the forward function above
    _, ll = forward(model, data, args)

    ll = np.sum(np.log(ll))/len(data)
    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll


def extract_parameters(model):
    # TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model.initials
    transitions = model.transitions
    mus = model.mus
    sigmas = model.sigmas
    # raise NotImplementedError #remove when parameter extraction is implemented
    return initials, transitions, mus, sigmas


def experiment(args, train_xs, dev_xs):
    import matplotlib.pyplot as plt

    iterations = 15
    cluster_num = 9
    args.nodev = True

    train_ll = np.zeros([iterations, cluster_num])
    dev_ll = np.zeros([iterations, cluster_num])

    for i in range(0, iterations):
        print('iter: ', i)
        for c in range(2, cluster_num + 1):
            args.iterations = i
            args.cluster_num = c
            model = init_model(args)
            model = train_model(model, train_xs, dev_xs, args)
            train_ll[i][c - 2] = average_log_likelihood(model, train_xs, args)
            dev_ll[i][c - 2] = average_log_likelihood(model, dev_xs, args)

    iterations = list(range(iterations))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(iterations, train_ll[:, 0])
    axs[0, 0].plot(iterations, dev_ll[:, 0])
    axs[0, 0].set_title('cluster_num = 2')

    axs[0, 1].plot(iterations, train_ll[:, 1])
    axs[0, 1].plot(iterations, dev_ll[:, 1])
    axs[0, 1].set_title('cluster_num = 3')

    axs[1, 0].plot(iterations, train_ll[:, 2])
    axs[1, 0].plot(iterations, dev_ll[:, 2])
    axs[1, 0].set_title('cluster_num = 4')

    axs[1, 1].plot(iterations, train_ll[:, 3])
    axs[1, 1].plot(iterations, dev_ll[:, 3])
    axs[1, 1].set_title('cluster_num = 5')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='average_log_likelihood')
        # ax.label_outer()
        ax.legend(['train_ll', 'dev_ll'])
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(iterations, train_ll[:, 4])
    axs[0, 0].plot(iterations, dev_ll[:, 4])
    axs[0, 0].set_title('cluster_num = 6')

    axs[0, 1].plot(iterations, train_ll[:, 5])
    axs[0, 1].plot(iterations, dev_ll[:, 5])
    axs[0, 1].set_title('cluster_num = 7')

    axs[1, 0].plot(iterations, train_ll[:, 6])
    axs[1, 0].plot(iterations, dev_ll[:, 6])
    axs[1, 0].set_title('cluster_num = 8')

    axs[1, 1].plot(iterations, train_ll[:, 7])
    axs[1, 1].plot(iterations, dev_ll[:, 7])
    axs[1, 1].set_title('cluster_num = 9')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='average_log_likelihood')
        # ax.label_outer()
        ax.legend(['train_ll', 'dev_ll'])
    plt.show()
    pass


def main():
    import argparse
    import os
    print('Gaussian')  # Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    init_group.add_argument('--plot', action='store_true', default=False,
                            help='If provided, graphs will be provided')

    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true',
                        help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied', action='store_true',
                        help='If provided, use a single covariance matrix for all clusters.')

    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print(
            'You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)

    if args.plot:
        experiment(args, train_xs, dev_xs)

    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '), transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))


if __name__ == '__main__':
    main()
