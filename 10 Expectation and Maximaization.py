import numpy as np

np.random.seed(0)
data = np.hstack((np.random.normal(0, 2, 100), np.random.normal(10, 2, 100)))

# Initialize parameters
mu1, mu2 = 0, 10
sigma1, sigma2 = 1, 1
pi1, pi2 = 0.5, 0.5

def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def expectation(data, mu1, mu2, sigma1, sigma2, pi1, pi2):
    gamma1 = pi1 * gaussian(data, mu1, sigma1)
    gamma2 = pi2 * gaussian(data, mu2, sigma2)
    gamma_sum = gamma1 + gamma2
    return gamma1 / gamma_sum, gamma2 / gamma_sum

def maximization(data, gamma1, gamma2):
    N1 = np.sum(gamma1)
    N2 = np.sum(gamma2)
    mu1 = np.sum(gamma1 * data) / N1
    mu2 = np.sum(gamma2 * data) / N2
    sigma1 = np.sqrt(np.sum(gamma1 * (data - mu1)**2) / N1)
    sigma2 = np.sqrt(np.sum(gamma2 * (data - mu2)**2) / N2)
    pi1 = N1 / len(data)
    pi2 = N2 / len(data)
    return mu1, mu2, sigma1, sigma2, pi1, pi2

# EM Algorithm
for _ in range(10):
    gamma1, gamma2 = expectation(data, mu1, mu2, sigma1, sigma2, pi1, pi2)
    mu1, mu2, sigma1, sigma2, pi1, pi2 = maximization(data, gamma1, gamma2)

print(f"Estimated means: mu1 = {mu1}, mu2 = {mu2}")
print(f"Estimated standard deviations: sigma1 = {sigma1}, sigma2 = {sigma2}")
print(f"Estimated mixing coefficients: pi1 = {pi1}, pi2 = {pi2}")
