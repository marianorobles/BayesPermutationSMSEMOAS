import os.path
import math
import numpy as np
import itertools
import stan
import nest_asyncio
nest_asyncio.apply()

class Base:
  def __init__(self, stan_model=None, seed=1, num_chains=1, num_samples=2000):#Inicialización viene de bradley terry <-
    self.seed = seed
    self.num_chains = num_chains
    self.num_samples = num_samples
    
    if stan_model != None:
      self.posterior=stan_model
      print(self.posterior)#hecho por mi,, guardas temporalmente el modelo donde queremos la posterior
      #self.posterior = stan.build(program_code=stan_model)#aquí está el error ya que necesito los datos, podemos conseguirlos antes?->

  def sample_posterior(self, permus, weights):

    model_data = self.get_model_data(permus, weights)#esto es lo que deberíamos pasar a la inicialización
    self.posterior=stan.build(program_code=self.posterior,data=model_data)#creado por mi, tal vez eliminar
    fit = self.posterior.sample(chains=self.num_chains, iter=self.num_samples, seed=self.seed)
    samples = self.get_samples(fit)
    return samples


  

  def get_samples(self, data):

    return data['ratings']

  def calculate_permu_prob(self, permu, params):
    return 0

  def sample_uniform_permu(self, permu):

    num_algorithms = len(permu)
    for i in range(num_algorithms):
      idx = np.random.randint(i, num_algorithms)
      permu[i], permu[idx] = permu[idx], permu[i]
    return permu

  def calculate_top_ranking_probs(self, permus, weights=None):

    samples = self.sample_posterior(permus, weights)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms))

    for i, sample in enumerate(samples):
      for permu in itertools.permutations(list(range(num_algorithms))):
        permu = np.array(permu)
        probs[i, permu[0]] += self.calculate_permu_prob(permu, sample)

    return probs

  def calculate_better_than_probs(self, permus, weights=None):

    samples = self.sample_posterior(permus, weights)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms, num_algorithms))

    for i, sample in enumerate(samples):
      for permu in itertools.permutations(list(range(num_algorithms))):
        permu = np.array(permu)
        prob = self.calculate_permu_prob(permu, sample)

        for j in range(num_algorithms):
          for k in range(j + 1, num_algorithms):
            probs[i, permu[j], permu[k]] += prob

    return probs

  def calculate_top_k_probs(self, permus, weights=None):


    samples = self.sample_posterior(permus, weights=None)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms, num_algorithms))

    for i, sample in enumerate(samples):
      for permu in itertools.permutations(list(range(num_algorithms))):
        permu = np.array(permu)
        prob = self.calculate_permu_prob(permu, sample)

        for j in range(num_algorithms):
          for k in range(j, num_algorithms):
            probs[i, permu[j], k] += prob

    return probs
