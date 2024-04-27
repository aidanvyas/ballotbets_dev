"""
Determine the optimal lambda value and standard deviation.

This module contains functions to determine the optimal lambda value and
standard deviation for the exponential function used to calculate the weighted
average of polling data.
"""


import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def determine_optimal_lambda():
    """
    Determine the optimal lambda value for polling data.

    This function loads polling data from a CSV file, calculates the optimal lambda value
    by evaluating R^2 (coefficient of determination) and MAE (mean absolute error) for a range of lambda values,
    and plots the evaluation metrics. It returns the average of the best lambda values for R^2 and MAE.

    Returns:
        float: The average of the best lambda values for R^2 and MAE.
    """
    # Load the data from the CSV file.
    data = pd.read_csv('raw_polls.csv')

    # Define a range of lambda values to test
    lambda_values = [i / 10000 for i in range(0, 10001)]

    # Initialize variables to store the best lambdas and their corresponding metrics
    best_lambda_r2, best_lambda_mae = None, None
    highest_r2, lowest_mae = float('-inf'), float('inf')

    # Lists to store R^2 and MAE values for each lambda
    r2_values, mae_values = [], []

    # Evaluate each lambda
    for lambda_ in lambda_values:
        predicted_values, actual_values = [], []

        # Group the data by 'race' and perform calculations
        for race, group in data.groupby('race'):
            group['weight'] = np.exp(-lambda_ * group['time_to_election']) * np.sqrt(group['samplesize'])
            group['weight'] /= group['weight'].sum()
            weighted_avg = (group['margin_poll'] * group['weight']).sum()

            predicted_values.append(weighted_avg)
            actual_values.append(group['margin_actual'].mean())

        # Calculate R^2 and MAE for this lambda
        r2 = r2_score(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)

        r2_values.append(r2)
        mae_values.append(mae)

        # Update best lambdas and metrics
        if r2 > highest_r2:
            best_lambda_r2, highest_r2 = lambda_, r2
        if mae < lowest_mae:
            best_lambda_mae, lowest_mae = lambda_, mae

        print(f"Lambda: {lambda_}, R^2: {r2}, MAE: {mae}")

    # Print best lambda values and metrics
    print(f"Best lambda for R^2: {best_lambda_r2}, Highest R^2: {highest_r2}")
    print(f"Best lambda for MAE: {best_lambda_mae}, Lowest MAE: {lowest_mae}")

    # Optional: Plotting the R^2 and MAE values against lambda
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lambda_values, r2_values, marker='o')
    plt.title('R^2 values')
    plt.xlabel('Lambda')
    plt.ylabel('R^2')

    plt.subplot(1, 2, 2)
    plt.plot(lambda_values, mae_values, marker='o')
    plt.title('MAE values')
    plt.xlabel('Lambda')
    plt.ylabel('MAE')

    plt.tight_layout()
    plt.show()

    # Return the average of the best lambda values for R^2 and MAE
    return (best_lambda_r2 + best_lambda_mae) / 2


def determine_optimal_standard_deviation():
    """
    Determine the optimal standard deviation from polling data.

    This function loads polling data from a CSV file, calculates the weighted average
    of the 'margin_poll' for each race using exponential decay based on time to election,
    and computes the standard deviation of the absolute differences between these
    weighted averages and the actual results.

    Returns:
        float: The standard deviation of the absolute differences.
    """
    # Load data from CSV file
    data = pd.read_csv('raw_data/raw_polls.csv')

    results = []

    # Calculate weighted averages and actual results for each race
    for race, group in data.groupby('race'):
        # Apply exponential decay to calculate weights
        group['weight'] = np.exp(-0.0619 * group['time_to_election']) * np.sqrt(group['samplesize'])
        group['weight'] /= group['weight'].sum()  # Normalize the weights

        # Calculate the weighted average of 'margin_poll'
        weighted_avg = (group['margin_poll'] * group['weight']).sum()
        actual_result = group['margin_actual'].mean()  # Get the actual result

        # Store race, weighted average, and actual result
        results.append([race, weighted_avg, actual_result])

    # Create DataFrame from results
    results_df = pd.DataFrame(results, columns=['race', 'weighted_avg', 'actual_result'])

    # Compute the standard deviation of absolute differences
    std_abs_diff = np.abs(results_df['weighted_avg'] - results_df['actual_result']).std()

    return std_abs_diff


if __name__ == "__main__":

    start_time = time.time()
    determine_optimal_lambda()
    print("Determined the optimal lambda.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    determine_optimal_standard_deviation()
    print("Determined the optimal standard deviation.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
