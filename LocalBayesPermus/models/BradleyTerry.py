from .Base import Base

import numpy as np
import stan
import nest_asyncio
nest_asyncio.apply()

bt_code = """ 
functions {
    
   real calculate_prob(int num_algorithms, int[] permu, vector ratings) {
       real prob = 1;

       for (i in 1:num_algorithms) {
           for (j in (i + 1):num_algorithms) {
               prob = prob * ratings[permu[i]];
           }
       }

       return prob;
   }

   real calculate_constant(int num_algorithms, vector ratings) {
       
      // Heap algorithm to generate every possible permutation

      int c[num_algorithms];
      int permu[num_algorithms];
      int i;
      int aux;
      real result = 0;

      for (k in 1:num_algorithms) {
          c[k] = 1;
          permu[k] = k;
      }

      result = result + calculate_prob(num_algorithms, permu, ratings);

      i = 2;
      while (i <= num_algorithms) {
        if (c[i] < i) {
            if (i % 2 != 0) {
                aux = permu[1];
                permu[1] = permu[i];
                permu[i] = aux;
            } else {
                aux = permu[c[i]];
                permu[c[i]] = permu[i];
                permu[i] = aux;
            }

          result = result + calculate_prob(num_algorithms, permu, ratings);

          c[i] += 1;
          i = 2;
        } else {
            c[i] = 1;
            i += 1;
        }
      }

      return result;
    }
}

data {
    int<lower=1> num_permus;
    int<lower=2> num_algorithms;
    int permus [num_permus, num_algorithms];
    vector[num_permus] weights;
    vector[num_algorithms] alpha;
}

parameters {
    simplex[num_algorithms] ratings;
}

transformed parameters {
  real loglik;
  real rest;
  real constant = calculate_constant(num_algorithms, ratings);

  loglik = 0;
  for (s in 1:num_permus){
    rest = 1;
    for (i in 1:(num_algorithms - 1)) {
      for (j in (i + 1):num_algorithms) {
        rest = rest * ratings[permus[s, i]];
      }
    }

    loglik = loglik + log(weights[s] * rest / constant);
  }
}

model {
    ratings ~ dirichlet(alpha);
    target += loglik;
}
"""#tenemos el código de stan aquí

class BradleyTerry(Base):
  #The Bradley-Terry model.

  def __init__(self, alpha, seed=1, num_chains=1, num_samples=2000):
    Base.__init__(self, stan_model=bt_code, 
                  seed=seed, num_chains=num_chains, num_samples=num_samples)#inicializamos en base (4 parámetros)1->
    self.alpha = alpha

  def get_model_data(self, permus, weights=None):
    num_permus, num_algorithms = permus.shape

    if weights == None:
      weights = [1 for _ in range(num_permus)]

    model_data = {'num_permus': num_permus,
                  'num_algorithms': num_algorithms,
                  'permus': permus,
                  'weights': weights,
                  'alpha': self.alpha}
    return model_data

  def get_samples(self, data):
    result = [(ratings, constant) for ratings, constant in zip(data['ratings'].T, data['constant'].T)]
    return np.array(result)

  def calculate_permu_prob(self, permu, params):
    num_algorithms = len(permu)
    ratings, constant = params
    prob = 1
    for i in range(num_algorithms):
      prob *= ratings[permu[i]] ** (num_algorithms - i - 1)
  
    return prob / constant

  def calculate_better_than_probs(self, permus, weights=None):
    samples = self.sample_posterior(permus, weights)#aquí tenemos sample funcion de base
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms, num_algorithms))

    for i, sample in enumerate(samples):
      params, constant = sample
      for j in range(num_algorithms):
        for k in range(num_algorithms):
          if j != k:
              probs[i, j, k] = params[j] / (params[j] + params[k])

    return probs

