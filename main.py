"""
Download polling data, process it, and calculate daily weighted averages.

This script downloads polling data, processes it, and calculates daily weighted averages
for the candidates based on an exponential function and sample size.
"""


import pandas as pd
import numpy as np
import requests
from datetime import timedelta
import os
import io
import time
from scipy.stats import norm


# Hard-coded values
LAMBDA = 0.0619
STANDARD_DEVIATION = 5.356


def get_polling_data(url, output_file):
    """
    Download the CSV file from the specified URL and save it locally.

    Args:
        url (str): The URL of the CSV file to download.
        output_file (str): The path to save the downloaded CSV file.
    """

    # Get the polling data from the URL.
    response = requests.get(url)

    # Decode the content of the response.
    content = response.content.decode('utf-8')

    # Read the polling data into a DataFrame.
    polling_data = pd.read_csv(io.StringIO(content))

    # Get the columns that are not related to the candidate names and percentages.
    non_candidate_columns = [col for col in polling_data.columns if col not in ('candidate_name', 'pct')]

    # Drop duplicate rows based on the poll_id column.
    unique_poll_details = polling_data[non_candidate_columns].drop_duplicates('poll_id')

    # Pivot the candidate names and percentages to columns.
    candidate_percentages = polling_data.pivot_table(index='poll_id', columns='candidate_name',
                                                    values='pct', aggfunc='first').reset_index()

    # Merge the unique poll details with the candidate percentages.
    merged_poll_data = pd.merge(unique_poll_details, candidate_percentages, on='poll_id', how='left')

    # Fill missing values with 0.
    merged_poll_data.fillna(0, inplace=True)

    # Rename the columns for consistency.
    merged_poll_data = merged_poll_data[['poll_id', 'display_name', 'state', 'end_date', 'sample_size', 'url'] +
                                        [col for col in candidate_percentages.columns if col != 'poll_id']]

    # Filter out rows where both candidates have 0 percentage.
    merged_poll_data = merged_poll_data[(merged_poll_data['Joe Biden'] != 0) & (merged_poll_data['Donald Trump'] != 0)]

    # Create the output directory if it does not exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the processed polling data to a CSV file.
    merged_poll_data.to_csv(output_file, index=False)


def create_national_polling_averages(input_file, output_file):
    """
    Process the polling data and calculate daily weighted averages for the candidates.

    Args:
        input_file (str): The path to the input CSV file containing the polling data.
        output_file (str): The path to save the output CSV file with daily averages.
    """

    # Read the polling data from the input file.
    polling_data = pd.read_csv(input_file)

    # Only keep the national polls.
    polling_data = polling_data[polling_data['state'] == '0']

    # Convert the 'end_date' column to a datetime object.
    polling_data['end_date'] = pd.to_datetime(polling_data['end_date'])

    # Get the first and last end dates from the polling data.
    first_end_date = polling_data['end_date'].min() + pd.Timedelta(days=1)
    last_end_date = polling_data['end_date'].max() + pd.Timedelta(days=1)

    # Generate a range of dates from the first to the last end date.
    dates = pd.date_range(start=first_end_date, end=last_end_date)

    # Create the output file and write the header.
    with open(output_file, 'w') as file:
        file.write("Date,Joe Biden,Donald Trump,Joe Biden Win Probability,Donald Trump Win Probability\n")

    # Iterate over the dates and calculate the daily weighted averages.
    for day in dates:

        # Only consider polls that have ended before the current date.
        polls = polling_data[polling_data['end_date'] < day].copy()

        # Calculate the number of days between the current date and the end date of the polls.
        polls.loc[:, 'days_diff'] = (day - polls['end_date']).dt.days

        # Calculate the weights for each poll based on the number of days and sample size.
        polls.loc[:, 'weight'] = np.exp(-LAMBDA * polls['days_diff']) * np.sqrt(polls['sample_size'])

        # Normalize the weights.
        polls.loc[:, 'weight'] = polls['weight'] / polls['weight'].sum()

        # Calculate the weighted average for Joe Biden and Donald Trump.
        biden_avg = (polls['Joe Biden'] * polls['weight']).sum()
        trump_avg = (polls['Donald Trump'] * polls['weight']).sum()

        # Calculate the margin between Biden and Trump.
        margin = biden_avg - trump_avg

        # Calculate the z-score based on the margin and standard deviation.
        z_score = margin / STANDARD_DEVIATION

        # Calculate the win probabilities using the cumulative distribution function (CDF) of the standard normal distribution.
        biden_win_prob = norm.cdf(z_score)
        trump_win_prob = 1 - biden_win_prob

        # Open the output file in append mode and write the daily averages and win probabilities.
        with open(output_file, 'a') as file:
            file.write(f"{day.strftime('%Y-%m-%d')},{biden_avg},{trump_avg},{biden_win_prob},{trump_win_prob}\n")


def create_state_polling_averages():
    # Read in the csv files once
    past_results = pd.read_csv('raw_data/raw_past_results.csv')
    national_polling = pd.read_csv('processed_data/president_polls_daily.csv')
    state_polling = pd.read_csv('processed_data/processed_polls.csv')

    # Convert dates to datetime objects immediately
    national_polling['Date'] = pd.to_datetime(national_polling['Date'])
    state_polling['end_date'] = pd.to_datetime(state_polling['end_date'])

    # Filter out national data and find unique states early
    states = past_results.loc[past_results['Location'] != 'National', 'Location'].unique()

    # Determine the date range for averaging
    start_date = national_polling['Date'].min() + timedelta(days=14)
    end_date = national_polling['Date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)

    # Pre-calculate the past national shares and total votes for optimization
    national_past_results = past_results[past_results['Location'] == 'National']
    biden_past_national_share = national_past_results['Biden Share'].values[0]
    trump_past_national_share = national_past_results['Trump Share'].values[0]

    # Prepare DataFrame to hold results
    biden_averages = pd.DataFrame(index=date_range, columns=states)
    trump_averages = pd.DataFrame(index=date_range, columns=states)
    biden_win_probabilities = pd.DataFrame(index=date_range, columns=states)

    # Vectorize where possible
    for state in states:
        state_polls = state_polling[state_polling['state'] == state]
        state_past_results = past_results[past_results['Location'] == state]
        biden_past_share = state_past_results['Biden Share'].values[0]
        trump_past_share = state_past_results['Trump Share'].values[0]

        for date in date_range:
            # Reduce repeated filtering and calculations
            national_polls_to_date = national_polling[national_polling['Date'] <= date]
            current_total = national_polls_to_date.iloc[-1]['Joe Biden'] + national_polls_to_date.iloc[-1]['Donald Trump']

            biden_boost = (national_polls_to_date.iloc[-1]['Joe Biden'] / current_total) / biden_past_national_share
            trump_boost = (national_polls_to_date.iloc[-1]['Donald Trump'] / current_total) / trump_past_national_share

            biden_estimated_share = biden_boost * biden_past_share * current_total
            trump_estimated_share = trump_boost * trump_past_share * current_total

            # Process state-specific data
            state_polls_to_date = state_polls[state_polls['end_date'] <= date].copy()

            if not state_polls_to_date.empty:
                state_polls_to_date['weight'] = np.exp(-0.1 * (date - state_polls_to_date['end_date']).dt.days)  # Assuming LAMBDA=0.1
                state_polls_to_date['weight'] /= state_polls_to_date['weight'].sum()
                biden_state_avg = (state_polls_to_date['Joe Biden'] * state_polls_to_date['weight']).sum()
                trump_state_avg = (state_polls_to_date['Donald Trump'] * state_polls_to_date['weight']).sum()
            else:
                biden_state_avg = biden_estimated_share
                trump_state_avg = trump_estimated_share

            # Save averages and probabilities
            biden_averages.loc[date, state] = biden_state_avg
            trump_averages.loc[date, state] = trump_state_avg

            margin = biden_state_avg - trump_state_avg
            z_score = margin / STANDARD_DEVIATION
            biden_win_prob = norm.cdf(z_score)
            biden_win_probabilities.loc[date, state] = biden_win_prob

    # Reset the index and rename the 'index' column to 'Date'
    biden_averages.reset_index(inplace=True)
    biden_averages.rename(columns={'index': 'Date'}, inplace=True)
    trump_averages.reset_index(inplace=True)
    trump_averages.rename(columns={'index': 'Date'}, inplace=True)
    biden_win_probabilities.reset_index(inplace=True)
    biden_win_probabilities.rename(columns={'index': 'Date'}, inplace=True)

    # Output to CSV
    biden_averages.to_csv('processed_data/biden_state_averages.csv', index=False)
    trump_averages.to_csv('processed_data/trump_state_averages.csv', index=False)
    biden_win_probabilities.to_csv('processed_data/biden_win_probabilities.csv', index=False)


def simulate_electoral_votes():
    # Load the electoral votes data
    electoral_votes = pd.read_csv('raw_data/raw_electoral_votes.csv')
    electoral_votes.set_index('Location', inplace=True)

    # Load Biden's win probabilities
    biden_win_probs = pd.read_csv('processed_data/biden_win_probabilities.csv')

    # Define the correlation matrix for the states
    states = electoral_votes.index.tolist()
    num_states = len(states)
    correlation_matrix = np.full((num_states, num_states), 0.5)  # Example: 0.5 correlation between all pairs
    np.fill_diagonal(correlation_matrix, 1)

    # Number of simulations
    num_simulations = 10000

    results = []

    # Process each date in Biden's win probability file
    for date in biden_win_probs['Date'].unique():
        daily_data = biden_win_probs[biden_win_probs['Date'] == date]
        
        state_indices = [states.index(state) for state in states if state in daily_data.columns]
        win_probs = daily_data[states].values[0, state_indices]
        
        # Generate correlated random outcomes based on the win probabilities
        mean = np.arcsin(2 * win_probs - 1)  # Apply a sin transformation for the mean
        correlated_normals = np.random.multivariate_normal(mean, correlation_matrix, size=num_simulations)
        correlated_outcomes = (np.sin(correlated_normals) + 1) / 2 > 0.5  # Convert back and determine win/loss

        # Calculate electoral votes for each simulation
        electoral_votes_array = electoral_votes.loc[states, 'Electoral Votes'].values[state_indices]
        simulated_electoral_votes = (correlated_outcomes * electoral_votes_array).sum(axis=1)
        
        # Calculate the probability of Biden winning in each simulation
        biden_wins = (simulated_electoral_votes > 269).mean()
        trump_wins = 1 - biden_wins

        # Append results for the current date
        results.append({
            'Date': date,
            'Biden Win Probability': biden_wins,
            'Trump Win Probability': trump_wins
        })

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('processed_data/simulated_national_election_outcomes_correlated.csv', index=False)


if __name__ == '__main__':
    url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
    processed_file = 'processed_data/processed_polls.csv'
    output_file = 'processed_data/president_polls_daily.csv'

    start_time = time.time()
    get_polling_data(url, processed_file)
    print("Finished processing polling data.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    create_national_polling_averages(processed_file, output_file)
    print("Finished creating national polling averages.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    create_state_polling_averages()
    print("Finished creating state polling averages.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    simulate_electoral_votes()
    print("Finished calculating expected electoral votes.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
