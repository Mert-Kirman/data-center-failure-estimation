import numpy as np
from scipy.stats import f, t

def g(x):
    '''
    Function that calculates the number of failures per day
    '''
    return 0.1 * (x[0] ** 2) + 12.5 * (x[1] ** 2) - 7.5 * (x[2] ** 2)

mu = [20, 0.3, 0.8]
sigma = [[4, 0.5, 0.2],
         [0.5, 0.7, 0.2],
         [0.2, 0.2, 0.1]]

def monte_carlo_estimation(sample_size):
    # Sample from the multivariate normal distribution
    samples = np.random.multivariate_normal(mu, sigma, sample_size)
    
    # Compute g(x) for each sample
    g_values = np.array([g(sample) for sample in samples])
    
    # Estimate the expected value
    E_g = np.mean(g_values)
    
    # Calculate 95% confidence interval
    std_error = np.std(g_values, ddof=1) / np.sqrt(sample_size)
    confidence_interval = (E_g - 1.96 * std_error, E_g + 1.96 * std_error)

    sample_variance = np.var(g_values, ddof=1)
    
    return E_g, confidence_interval, sample_variance


# Task 2
print('Results for Task-2:')
sample_sizes = [50, 100, 1000, 10000]
for sample_size in sample_sizes:
    E_g, confidence_interval, _ = monte_carlo_estimation(sample_size)
    print(f"n = {sample_size:5d}, E[g(x)] = {E_g:.4f}, 95% CI = {confidence_interval[0].item():.4f}, {confidence_interval[1].item():.4f}")

# Task 3
print('\nResults for Task-3:')
n_0 = 10000 # Sample size 0
n_1 = 50    # Sample size 1

E_g0, _, s_0 = monte_carlo_estimation(n_0)  # Expected value, _, sample variance
E_g1, _, s_1 = monte_carlo_estimation(n_1)

# Test if variances are different (F-distribution) using alpha = 0.05
alpha = 0.05
df1 = n_0 - 1
df2 = n_1 - 1

f_score = s_0 / s_1     # s_0 and s_1 are already variances, no squaring required
f_upper = f.ppf(1 - alpha/2, df1, df2)
f_lower = f.ppf(alpha/2, df1, df2)

print(f'f_score: {f_score:.4f}, f_lower: {f_lower:.4f}, f_upper: {f_upper:.4f}')
if f_score < f_lower or f_score > f_upper:
    print('Reject H0 null hypothesis, variances are significantly different!')
else:
    print('Failed to reject H0 null hypothesis, var0 == var1')

# Variances unknown but equal, use pooled estimate of variance
df = n_0 + n_1 - 2
pooled_estimate_variance = ((n_0 - 1) * s_0 + (n_1 - 1) * s_1) / df
t_score = (E_g0 - E_g1) / (np.sqrt(pooled_estimate_variance * (1/n_0 + 1/n_1)))

print(f'\nPooled estimate of variance: {pooled_estimate_variance:.4f}')

t_critical = t.ppf(1 - alpha/2, df)
print(f'T-score: {t_score:.4f}, t_critical: {t_critical:.4f}')

if t_score < -t_critical or t_score > t_critical:
    print('Reject H0 null hypothesis, g0 != g1')
else:
    print('Failed to reject H0 null hypothesis, g0 == g1')
