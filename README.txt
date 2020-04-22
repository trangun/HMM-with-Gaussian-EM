Trang Nguyen
Machine Learning
Homework 8: HMM with Gaussian EM

--Files
nguyen_hmm_gaussian.py: Main file and also has experiment function. Added extra args.plot to 
represent the experiment.

--Description
This project is an implementation of EM to train an hidden markov model (HMM) to get the max log
likelihood for the current distribution. We then have transitions probability matrix which is 
probability of moving from cluster i to cluster j, initials probability matrix which is the 
probability for starting in each state and emission probability matrix which is the probability
of the current state to the current observation variable. We also have alpha, beta, gamma and xi
which are intermediate variables. Alphas and betas are computed in forward and backward
respectively. Gamma is the probability of being in one state i at time t. Xi is the the 
probability of being in one state i at time t-1 and being in state j at time t when t != 0.

--Method of Initialize Model
Similar to the previous EM algorithm. Added transistion probability matrix and initials.

--Does the HMM model the data better than the original non-sequence model? 
Yes. HMM converges much faster than the non-sequence model. HMM model also can obtain a better 
result after few iterations that pure EM model might not get to, since EM might get stuck at some
local optimal and cannot have higher precision.

--What is the best number of states?
We have a total of 4 graphs. The graphs in nt_cluster2-5.png and nt_cluster6-9.png are the models
with standard, separate covariance setting which we will denote as graph (a) here. The graphs in 
t_cluster2-5.png and t_cluster6-9.png are the models that implement tied covariances. We will 
denote these graphs as graph (b) here. Since graph (b) has singular variance, it will be not a
good option to decide the best state.

From graphs (a) we have, in my opinion, the best number of states is 6. Since after state 6, we 
can reach highest average loglikelihood after few iterations. However, the higher the number of
states is, we might face a problem of overfitting. So 6 is probably the best option here in this 
experiment.
