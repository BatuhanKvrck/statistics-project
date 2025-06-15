# Statistics Project - BATUHAN KIVIRCIK - 2021221059
# Student Math Scores Statistical Analysis
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
# ===================================================================
# 1. DATA LOADING AND INITIAL EXAMINATION
# ===================================================================
print("=== STUDENT MATH SCORES ANALYSIS ===")
print()
# Create sample dataset (no CSV file needed)
np.random.seed(42) # For reproducible results
# Generate 1000 student math scores with normal distribution
math_scores_array = np.random.normal(loc=66, scale=15, size=1000)
math_scores_array = np.clip(math_scores_array, 0, 100) # Limit to 0-100 range
# Create other sample columns
genders = np.random.choice(['male', 'female'], size=1000)
ethnicities = np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], size=1000)
parent_education = np.random.choice(['high school', 'some college', 'bachelor\'s', 'master\'s'], size=1000)
lunch = np.random.choice(['standard', 'free/reduced'], size=1000)
test_prep = np.random.choice(['none', 'completed'], size=1000)
# Create DataFrame
df = pd.DataFrame({
 'gender': genders,
 'race/ethnicity': ethnicities,
 'parental level of education': parent_education,
 'lunch': lunch,
 'test preparation course': test_prep,
 'math score': math_scores_array,
 'reading score': np.random.normal(loc=69, scale=14, size=1000),
 'writing score': np.random.normal(loc=68, scale=15, size=1000)
})
# Select main variable for analysis
math_scores = df['math score']
# Basic dataset information
print("=== DATASET OVERVIEW ===")
print(f"Dataset: {df.shape[0]} students x {df.shape[1]} variables")
print(f"Missing data: {df.isnull().sum().sum()} (none)")
print(f"Selected variable: Math Score (range: {math_scores.min():.1f}-{math_scores.max():.1f})")
print()
# Display first few observations
print("First 3 rows:")
print(df.head(3))
print()
# ===================================================================
# 2. DESCRIPTIVE STATISTICS CALCULATION (10 POINTS)
# ===================================================================
print("=== DESCRIPTIVE STATISTICS ===")
# Calculate central tendency measures
mean_score = math_scores.mean() # Arithmetic mean
median_score = math_scores.median() # Median value
# Calculate dispersion measures
variance_score = math_scores.var(ddof=1) # Sample variance (n-1)
std_dev_score = math_scores.std(ddof=1) # Sample standard deviation
# Calculate standard error for sampling distribution
std_error = std_dev_score / math.sqrt(len(math_scores))
# Print calculated descriptive statistics
print(f"Mean: {mean_score:.4f}")
print(f"Median: {median_score:.4f}")
print(f"Variance: {variance_score:.4f}")
print(f"Standard Deviation: {std_dev_score:.4f}")
print(f"Standard Error: {std_error:.4f}")
print()
# ===================================================================
# 3. DATA VISUALIZATION (15 POINTS)
# ===================================================================
print("=== DATA VISUALIZATION ===")
# Create figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# Histogram - Shows frequency distribution
ax1.hist(math_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
# Add mean and median lines (to see distribution symmetry)
ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_score:.2f}')
ax1.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median = {median_score:.2f}')
ax1.set_xlabel('Math Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Math Scores (Histogram)')
ax1.legend()
ax1.grid(True, alpha=0.3)
# Box plot - Shows outliers and quartiles
box_plot = ax2.boxplot(math_scores, patch_artist=True)
box_plot['boxes'][0].set_facecolor('lightblue')
ax2.set_ylabel('Math Score')
ax2.set_title('Math Scores Boxplot (Outlier Detection)')
ax2.grid(True, alpha=0.3)
# Adjust layout and show plots
plt.tight_layout()
plt.show()
# Outlier detection using IQR (Interquartile Range) method
Q1 = math_scores.quantile(0.25) # First quartile (25th percentile)
Q3 = math_scores.quantile(0.75) # Third quartile (75th percentile)
IQR = Q3 - Q1 # Interquartile range
# Determine outlier boundaries (1.5*IQR rule)
lower_bound = Q1 - 1.5 * IQR # Lower boundary
upper_bound = Q3 + 1.5 * IQR # Upper boundary
# Find values outside the boundaries (outliers)
outliers = math_scores[(math_scores < lower_bound) | (math_scores > upper_bound)]
# Print outlier analysis results
print(f"Outlier analysis:")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)}")
print(f"Outlier values: {outliers.tolist()}")
print()
# ===================================================================
# 4. CONFIDENCE INTERVALS (10 POINTS)
# ===================================================================
print("=== CONFIDENCE INTERVALS (95%) ===")
# Define parameters for confidence interval calculations
confidence_level = 0.95 # Confidence level (95%)
alpha = 1 - confidence_level # Significance level (α = 0.05)
n = len(math_scores) # Sample size
df_freedom = n - 1 # Degrees of freedom
# CONFIDENCE INTERVAL FOR MEAN (uses t-distribution because σ is unknown)
t_critical = stats.t.ppf(1 - alpha/2, df_freedom) # Critical t value (two-tailed)
margin_error_mean = t_critical * std_error # Margin of error calculation
ci_mean_lower = mean_score - margin_error_mean # Lower bound
ci_mean_upper = mean_score + margin_error_mean # Upper bound
print(f"95% Confidence Interval for Mean:")
print(f"[{ci_mean_lower:.4f}, {ci_mean_upper:.4f}]")
print(f"Margin of Error: ±{margin_error_mean:.4f}")
print()
# CONFIDENCE INTERVAL FOR VARIANCE (uses chi-square distribution)
chi2_lower = stats.chi2.ppf(alpha/2, df_freedom) # Lower critical chi-square value
chi2_upper = stats.chi2.ppf(1 - alpha/2, df_freedom) # Upper critical chi-square value
# Calculate variance confidence interval
ci_var_lower = (df_freedom * variance_score) / chi2_upper
ci_var_upper = (df_freedom * variance_score) / chi2_lower
print(f"95% Confidence Interval for Variance:")
print(f"[{ci_var_lower:.4f}, {ci_var_upper:.4f}]")
print()
# ===================================================================
# 5. SAMPLE SIZE ESTIMATION (10 POINTS)
# ===================================================================
print("=== SAMPLE SIZE ESTIMATION ===")
# Define parameters for sample size calculation
margin_error_target = 0.1 # Target margin of error (±0.1)
confidence_90 = 0.90 # Confidence level (90%)
z_90 = stats.norm.ppf(1 - (1-confidence_90)/2) # Critical z value for 90%
# Sample size formula: n = (z*σ/E)²
# Using current standard deviation as population estimate
required_sample_size = (z_90 * std_dev_score / margin_error_target) ** 2
# Compare results and evaluate
print(f"For 90% confidence level with ±0.1 margin of error:")
print(f"Required minimum sample size: {math.ceil(required_sample_size)}")
print(f"Our current sample size: {n}")
print(f"Is it sufficient? {'Yes' if n >= required_sample_size else 'No'}")
print()
# ===================================================================
# 6. HYPOTHESIS TESTING (15 POINTS)
# ===================================================================
print("=== HYPOTHESIS TESTING ===")
# Hypothesized population mean to test
hypothesized_mean = 70
# Define hypotheses
print(f"Hypothesis Test: Is math score mean different from 70?")
print(f"H0: μ = 70 (null hypothesis)") # Null hypothesis: no difference
print(f"H1: μ ≠ 70 (alternative hypothesis)") # Alternative hypothesis: there is difference
print(f"Significance level: α = 0.05") # Significance level
print()
# Calculate one-sample t-test statistic
t_statistic = (mean_score - hypothesized_mean) / std_error
# Calculate p-value for two-tailed test
p_value = 2 * stats.t.cdf(-abs(t_statistic), df_freedom)
# Print test results
print(f"Test Results:")
print(f"Sample mean: {mean_score:.4f}")
print(f"Hypothesized mean: {hypothesized_mean}")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Critical t-value (±): {stats.t.ppf(0.975, df_freedom):.4f}")
print()
# Statistical decision making (p-value vs α comparison)
if p_value < 0.05:
 print("Decision: REJECT H0 (p < 0.05)")
 print("Conclusion: Math score mean is statistically significantly different from 70.")
else:
 print("Decision: FAIL TO REJECT H0 (p >= 0.05)")  # DÜZELTME!
 print("Conclusion: Math score mean does not show statistically significant difference from 70.")
print()
print("=== ANALYSIS COMPLETED ===")