"""
Determine the optimal lambda value and standard deviation.

This module contains functions to determine the optimal lambda value and
standard deviation for the exponential function used to calculate the weighted
average of polling data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
from scipy.stats import norm


def determine_optimal_lambda():
    """
    Determine the optimal lambda value.

    This function loads data from a CSV file, calculates the optimal lambda
    value based on R^2 and MAE metrics, and optionally plots the results.

    Returns:
        float: The average of the best lambdas for R^2 and MAE.
    """
    # Load the data from the CSV file.
    data = pd.read_csv('raw_polls.csv')

    # Define a range of lambda values to test
    lambda_values = [i / 10000 for i in range(0, 10001)]

    # Initialize variables to store the best lambdas, the highest R^2 and the lowest MAE
    best_lambda_r2 = None
    best_lambda_mae = None
    highest_r2 = float('-inf')
    lowest_mae = float('inf')

    # Initialize lists to store the R^2 and MAE values for each lambda
    r2_values = []
    mae_values = []

    # Iterate over the lambda values
    for lambda_ in lambda_values:
        # Initialize lists to store predicted and actual values, and the cycle names
        predicted_values = []
        actual_values = []
        races = []

        # Group the data by 'race'
        for race, group in data.groupby('race'):
            # Calculate the weights for each poll in the cycle
            group['weight'] = np.exp(-lambda_ * group['time_to_election']) * np.sqrt(group['samplesize'])

            # Normalize the weights
            group['weight'] /= group['weight'].sum()

            # Calculate the weighted average of 'margin_poll' for the cycle
            weighted_avg = (group['margin_poll'] * group['weight']).sum()

            # Append the predicted and actual values, and the cycle name to the lists
            predicted_values.append(weighted_avg)
            actual_values.append(group['margin_actual'].mean())
            races.append(race)

        # Calculate the R^2 and MAE for this lambda
        r2 = r2_score(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)

        # Append the R^2 and MAE values to the lists
        r2_values.append(r2)
        mae_values.append(mae)

        # If this R^2 is higher than the highest R^2 we've seen so far,
        # update the best lambda for R^2, the highest R^2
        if r2 > highest_r2:
            best_lambda_r2 = lambda_
            highest_r2 = r2

        # If this MAE is lower than the lowest MAE we've seen so far,
        # update the best lambda for MAE, the lowest MAE
        if mae < lowest_mae:
            best_lambda_mae = lambda_
            lowest_mae = mae

        print(f"Lambda: {lambda_}, R^2: {r2}, MAE: {mae}")

    # Print the best lambdas, the highest R^2 and the lowest MAE
    print(f"Best lambda for R^2: {best_lambda_r2}")
    print(f"Highest R^2: {highest_r2}")
    print(f"Best lambda for MAE: {best_lambda_mae}")
    print(f"Lowest MAE: {lowest_mae}")

    # Optional Plotting
    # Plot the R^2 values
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lambda_values, r2_values, marker='o')
    plt.title('R^2 values')
    plt.xlabel('Lambda')
    plt.ylabel('R^2')

    # Plot the MAE values
    plt.subplot(1, 2, 2)
    plt.plot(lambda_values, mae_values, marker='o')
    plt.title('MAE values')
    plt.xlabel('Lambda')
    plt.ylabel('MAE')

    plt.tight_layout()
    plt.show()

    # Return the average of the best lambdas for R^2 and MAE.
    return (best_lambda_r2 + best_lambda_mae) / 2


def determine_optimal_standard_deviation():
    """
    Determine the optimal standard deviation.

    This function loads data from a CSV file, calculates the weighted average
    of 'margin_poll' for each race, and computes the standard deviation of the
    absolute differences between the weighted averages and actual results.

    Returns:
        float: The standard deviation of the absolute differences.
    """
    data = pd.read_csv('raw_data/raw_polls.csv')

    results = []

    for race, group in data.groupby('race'):
        # Calculate the weights for each poll in the cycle
        group['weight'] = np.exp(-0.0619 * group['time_to_election']) * np.sqrt(group['samplesize'])

        # Normalize the weights
        group['weight'] /= group['weight'].sum()

        # Calculate the weighted average of 'margin_poll' for the day
        weighted_avg = (group['margin_poll'] * group['weight']).sum()

        # Get the actual result
        actual_result = group['margin_actual'].mean()

        # Append the race, weighted average, and actual result to the results
        results.append([race, weighted_avg, actual_result])

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=['race', 'weighted_avg', 'actual_result'])

    # Calculate the absolute differences between 'weighted_avg' and 'actual_result'
    std_abs_diff = np.abs(results_df['weighted_avg'] - results_df['actual_result']).std()

    return std_abs_diff


if __name__ == "__main__":
    # determine_optimal_lambda()
    print(determine_optimal_standard_deviation())
