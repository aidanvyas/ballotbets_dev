"""
This script is designed to manage the workflow of polling data analysis for political candidates. It includes functionality to download and preprocess polling data, calculate daily and state-level weighted averages, simulate electoral vote outcomes, and visualize the results through various plots and maps.

Functions included:
- get_polling_data: Downloads and preprocesses raw polling data.
- create_national_polling_averages: Calculates national daily weighted averages and win probabilities.
- create_state_polling_averages: Computes state-level polling averages and probabilities, adjusting for historical election results.
- simulate_electoral_votes: Simulates electoral vote outcomes using state win probabilities to model possible election results.
- generate_plots: Generates visual representations of polling averages and electoral probabilities.
- generate_map: Creates a choropleth map visualizing state-specific win probabilities.

This script supports extensive data analysis workflows, making it suitable for use in political campaign strategies, academic research, or news analysis. The results provide insights into the current political landscape and potential election outcomes based on polling data.
"""


import io
import os
import time
from datetime import timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

# Constants
LAMBDA = 0.0619
STANDARD_DEVIATION = 5.356


def get_polling_data(url, output_file):
    """
    Download the CSV file from the specified URL and save it locally.

    Parameters:
        url (str): The URL of the CSV file to download.
        output_file (str): The path to save the downloaded CSV file.
    """
    # Get the polling data from the URL.
    response = requests.get(url)

    # Decode the content of the response.
    content = response.content.decode('utf-8')

    # Read the polling data into a DataFrame.
    polling_data = pd.read_csv(io.StringIO(content))

    # Identify non-candidate columns.
    non_candidate_columns = [
        col for col in polling_data.columns if col not in ('candidate_name', 'pct')
    ]

    # Drop duplicate rows based on the poll_id column.
    unique_poll_details = polling_data[non_candidate_columns].drop_duplicates('poll_id')

    # Pivot the candidate names and percentages to columns.
    candidate_percentages = polling_data.pivot_table(
        index='poll_id', columns='candidate_name', values='pct', aggfunc='first'
    ).reset_index()

    # Merge the unique poll details with the candidate percentages.
    merged_poll_data = pd.merge(unique_poll_details, candidate_percentages, on='poll_id', how='left')

    # Fill missing values with 0.
    merged_poll_data.fillna(0, inplace=True)

    # Rename the columns for consistency.
    merged_poll_data = merged_poll_data[
        ['poll_id', 'display_name', 'state', 'end_date', 'sample_size', 'url'] +
        [col for col in candidate_percentages.columns if col != 'poll_id']
    ]

    # Filter out rows where both candidates have 0 percentage.
    merged_poll_data = merged_poll_data[
        (merged_poll_data['Joe Biden'] != 0) & (merged_poll_data['Donald Trump'] != 0)
    ]

    # Create the output directory if it does not exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the processed polling data to a CSV file.
    merged_poll_data.to_csv(output_file, index=False)


def create_national_polling_averages(input_file, output_file):
    """
    Process polling data from a CSV file to calculate and save daily weighted averages for the candidates along with their win probabilities.

    This function reads polling data, computes daily averages based on weights, and estimates win probabilities for each candidate.

    Parameters:
        input_file (str): Path to the input CSV file containing the polling data.
        output_file (str): Path to save the output CSV file with daily averages.
    """
    # Read polling data from the input file.
    polling_data = pd.read_csv(input_file)

    # Filter to include only national polls.
    polling_data = polling_data[polling_data['state'] == '0']

    # Convert 'end_date' to a datetime object and set date range.
    polling_data['end_date'] = pd.to_datetime(polling_data['end_date'])
    first_end_date = polling_data['end_date'].min() + pd.Timedelta(days=1)
    last_end_date = polling_data['end_date'].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start=first_end_date, end=last_end_date)

    # Initialize output file and write the header.
    header = "Date,Joe Biden,Donald Trump,Joe Biden Win Probability,Donald Trump Win Probability\n"
    with open(output_file, 'w') as file:
        file.write(header)

    # Calculate daily weighted averages and probabilities.
    for day in dates:
        # Filter polling data based on end date.
        polls = polling_data[polling_data['end_date'] < day].copy()
        polls['days_diff'] = (day - polls['end_date']).dt.days

        # Calculate weights and normalize them.
        polls['weight'] = np.exp(-LAMBDA * polls['days_diff']) * np.sqrt(polls['sample_size'])
        polls['weight'] /= polls['weight'].sum()

        # Calculate weighted averages for each candidate.
        biden_avg = (polls['Joe Biden'] * polls['weight']).sum()
        trump_avg = (polls['Donald Trump'] * polls['weight']).sum()

        # Calculate the z-score and win probabilities.
        margin = biden_avg - trump_avg
        z_score = margin / STANDARD_DEVIATION
        biden_win_prob = norm.cdf(z_score)
        trump_win_prob = 1 - biden_win_prob

        # Append results to the output file.
        results = f"{day.strftime('%Y-%m-%d')},{biden_avg},{trump_avg},{biden_win_prob},{trump_win_prob}\n"
        with open(output_file, 'a') as file:
            file.write(results)

    return f"Biden is currently polling at {biden_avg / 100:.2%}, while Trump is at {trump_avg / 100:.2%}."


def create_state_polling_averages():
    """
    Calculate state-level polling averages and win probabilities based on national and state polls.
    This function adjusts shares and boost factors according to past election results and saves the outputs to CSV files.
    """
    # Load data from CSV files
    past_results = pd.read_csv('raw_data/raw_past_results.csv')
    national_polling = pd.read_csv('processed_data/president_polls_daily.csv')
    state_polling = pd.read_csv('processed_data/processed_polls.csv')

    # Convert date columns to datetime objects only once
    national_polling['Date'] = pd.to_datetime(national_polling['Date'])
    state_polling['end_date'] = pd.to_datetime(state_polling['end_date'])

    # Extract states excluding national results
    states = past_results.loc[past_results['Location'] != 'National', 'Location'].unique()

    # Define date range for averaging
    start_date = national_polling['Date'].min() + timedelta(days=14)
    end_date = state_polling['end_date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)

    # Pre-calculate national past results for optimization
    national_past_results = past_results.loc[past_results['Location'] == 'National']
    biden_past_national_share = national_past_results['Biden Share'].values[0]
    trump_past_national_share = national_past_results['Trump Share'].values[0]

    # Prepare DataFrames to hold results
    biden_averages = pd.DataFrame(index=date_range, columns=states)
    trump_averages = pd.DataFrame(index=date_range, columns=states)
    biden_win_probabilities = pd.DataFrame(index=date_range, columns=states)

    # Process state data and compute averages and probabilities
    for state in states:
        state_polls = state_polling.loc[state_polling['state'] == state]
        state_past_results = past_results.loc[past_results['Location'] == state]
        biden_past_share = state_past_results['Biden Share'].values[0]
        trump_past_share = state_past_results['Trump Share'].values[0]

        for date in date_range:
            # Calculate boost factors from national polls
            national_polls_to_date = national_polling.loc[national_polling['Date'] <= date]
            current_total = national_polls_to_date.iloc[-1]['Joe Biden'] + national_polls_to_date.iloc[-1]['Donald Trump']
            biden_boost = (national_polls_to_date.iloc[-1]['Joe Biden'] / current_total) / biden_past_national_share
            trump_boost = (national_polls_to_date.iloc[-1]['Donald Trump'] / current_total) / trump_past_national_share

            biden_estimated_share = biden_boost * biden_past_share * current_total
            trump_estimated_share = trump_boost * trump_past_share * current_total

            # Aggregate state-specific polling data up to current date
            state_polls_to_date = state_polls.loc[state_polls['end_date'] <= date]
            national_poll_date = date - timedelta(days=14)
            
            # Add national polling estimate as an additional "poll"
            national_poll_entry = pd.DataFrame({
                'Joe Biden': [biden_estimated_share],
                'Donald Trump': [trump_estimated_share],
                'end_date': [national_poll_date],
                'sample_size': [1000]
            })
            state_polls_to_date = pd.concat([state_polls_to_date, national_poll_entry], ignore_index=True)

            state_polls_to_date['weight'] = np.exp(-LAMBDA * (date - state_polls_to_date['end_date']).dt.days) * np.sqrt(state_polls_to_date['sample_size'])
            state_polls_to_date['weight'] /= state_polls_to_date['weight'].sum()  # Normalize weights
            biden_state_avg = (state_polls_to_date['Joe Biden'] * state_polls_to_date['weight']).sum()
            trump_state_avg = (state_polls_to_date['Donald Trump'] * state_polls_to_date['weight']).sum()


            # Save daily averages and win probabilities
            biden_averages.loc[date, state] = biden_state_avg
            trump_averages.loc[date, state] = trump_state_avg
            margin = biden_state_avg - trump_state_avg
            z_score = margin / STANDARD_DEVIATION
            biden_win_prob = norm.cdf(z_score)
            biden_win_probabilities.loc[date, state] = biden_win_prob

        print(f"Processed state: {state}, latest Biden share: {biden_state_avg / 100:.2%}, latest Trump share: {trump_state_avg / 100:.2%}")

    # Save results to CSV files
    for df, filename in zip([biden_averages, trump_averages, biden_win_probabilities],
                            ['biden_state_averages.csv', 'trump_state_averages.csv', 'biden_win_probabilities.csv']):
        df.reset_index().rename(columns={'index': 'Date'}).to_csv(f'processed_data/{filename}', index=False)

    # for all the states where biden is between 5% and 95% chance of winning, print them out and each candidates' win probability
    biden_win_probabilities = biden_win_probabilities.iloc[-1]
    trump_win_probabilities = 1 - biden_win_probabilities
    closest_states = biden_win_probabilities[(biden_win_probabilities > 0.05) & (biden_win_probabilities < 0.95)]
    closest_states = closest_states.sort_values()
    # should include the win probability of each candidate
    closest_states_string = ', '.join([f"{state} ({biden_win_probabilities[state]:.2%} Biden, {trump_win_probabilities[state]:.2%} Trump)" for state in closest_states.index])

    return f"The closest states are {closest_states_string}."


def simulate_electoral_votes():
    """
    Simulate the electoral vote outcomes using Biden's state win probabilities, accounting for correlations between states.

    The results are saved to a CSV file.
    """
    # Load the electoral votes data and set the index
    electoral_votes = pd.read_csv('raw_data/raw_electoral_votes.csv')
    electoral_votes.set_index('Location', inplace=True)

    # Load Biden's win probabilities
    biden_win_probs = pd.read_csv('processed_data/biden_win_probabilities.csv')

    # Define the correlation matrix for the states
    states = electoral_votes.index.tolist()
    num_states = len(states)
    correlation_matrix = np.full((num_states, num_states), 0.5)
    np.fill_diagonal(correlation_matrix, 1)

    # Set the number of simulations
    num_simulations = 10000

    results = []

    # Simulate electoral vote outcomes for each date
    for date in biden_win_probs['Date'].unique():
        daily_data = biden_win_probs[biden_win_probs['Date'] == date]

        state_indices = [states.index(state) for state in states if state in daily_data.columns]
        win_probs = daily_data[states].iloc[0, state_indices]

        # Generate correlated random outcomes based on win probabilities
        mean = np.arcsin(2 * win_probs - 1)
        correlated_normals = np.random.multivariate_normal(mean, correlation_matrix, size=num_simulations)
        correlated_outcomes = (np.sin(correlated_normals) + 1) / 2 > 0.5

        # Calculate electoral votes for each simulation
        electoral_votes_array = electoral_votes.loc[states, 'Electoral Votes'].values[state_indices]
        simulated_electoral_votes = (correlated_outcomes * electoral_votes_array).sum(axis=1)

        # Calculate probabilities of outcomes
        biden_wins = (simulated_electoral_votes > 269).mean()
        tie = (simulated_electoral_votes == 269).mean()
        trump_wins = 1 - biden_wins - tie

        # Append results for the current date
        results.append({
            'Date': date,
            'Biden Win Probability': biden_wins,
            'Trump Win Probability': trump_wins,
            'Tie Probability': tie
        })

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('processed_data/simulated_national_election_outcomes_correlated.csv', index=False)

    return [f"Biden has an {results[-1]['Biden Win Probability']:.2%} chance of winning the election, Trump has an {results[-1]['Trump Win Probability']:.2%} chance of winning the election, and there is an {results[-1]['Tie Probability']:.2%} chance of a tie.", f"Biden is expected to win {np.median([simulated_electoral_votes]):.0f} electoral votes, while Trump is expected to win {538 - np.median([simulated_electoral_votes]):.0f} electoral votes."]


def generate_plots(polling_data_file, probabilities_file):
    """
    Plot the national polling averages and electoral college probabilities for Biden and Trump.

    Parameters:
        polling_data_file (str): Path to the CSV file containing the polling data.
        probabilities_file (str): Path to the CSV file containing the election probabilities.
    """
    # Read the data from CSV files
    polling_data = pd.read_csv(polling_data_file)
    probabilities = pd.read_csv(probabilities_file)

    # Convert 'Date' columns to datetime format
    polling_data['Date'] = pd.to_datetime(polling_data['Date'])
    probabilities['Date'] = pd.to_datetime(probabilities['Date'])

    # Filter data for events starting from 2023
    polling_data = polling_data[polling_data['Date'] >= pd.Timestamp('2023-01-01')]
    probabilities = probabilities[probabilities['Date'] >= pd.Timestamp('2023-01-01')]

    # Ensure the output directory exists
    os.makedirs('plots', exist_ok=True)

    # Plot national polling averages
    plt.figure(figsize=(12, 8))
    plt.plot(polling_data['Date'], polling_data['Joe Biden'], label='Joe Biden', color='blue')
    plt.plot(polling_data['Date'], polling_data['Donald Trump'], label='Donald Trump', color='red')
    plt.title('National Polling Averages')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim([min(polling_data['Joe Biden'].min(), polling_data['Donald Trump'].min()) - 5,
              max(polling_data['Joe Biden'].max(), polling_data['Donald Trump'].max()) + 5])
    plt.savefig('plots/national_polling_averages.png')

    # Plot electoral college probabilities
    plt.figure(figsize=(12, 8))
    plt.plot(probabilities['Date'], probabilities['Biden Win Probability'] * 100, label='Joe Biden', color='blue')
    plt.plot(probabilities['Date'], probabilities['Trump Win Probability'] * 100, label='Donald Trump', color='red')
    plt.plot(probabilities['Date'], probabilities['Tie Probability'] * 100, label='Tie', color='yellow')
    plt.title('Electoral College Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Probability (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig('plots/election_win_probabilities.png')


def generate_map(state_probabilities, states_shapefile):
    """
    Generate a choropleth map visualizing the probabilities of a specified outcome by state.

    Parameters:
        state_probabilities (str): Path to the CSV file containing state probabilities.
        states_shapefile (str): Path to the shapefile of US states.
    """
    # Load probability data and the US states shapefile
    data = pd.read_csv(state_probabilities)
    states = gpd.read_file(states_shapefile)

    # Prepare the data
    last_row = data.iloc[-1].rename('Probability').to_frame().reset_index().rename(columns={'index': 'State'})
    states = states.rename(columns={'NAME': 'State'})
    states = states[~states['State'].isin([
        'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'Guam', 'United States Virgin Islands',
        'American Samoa', 'Alaska', 'Hawaii'])]

    # Merge and prepare states data
    states_data = states.merge(last_row, on='State', how='left')
    states_data['Probability'] = states_data['Probability'].fillna(0)

    # Define colors
    white = '#FFFFFF'

    # Create a custom colormap based on Biden's win probability
    def custom_color(prob):
        if prob == 0.5:
            return white
        elif prob > 0.5:
            white_share = 1 - (prob - 0.5) * 2
            return plt.cm.colors.to_hex([white_share, white_share, 1])
        else:
            white_share = 1 - (0.5 - prob) * 2
            return plt.cm.colors.to_hex([1, white_share, white_share])

    states_data['Color'] = states_data['Probability'].apply(custom_color)

    # Create and configure the plot
    _, ax = plt.subplots(1, figsize=(15, 8))
    states_data.plot(color=states_data['Color'], linewidth=0.2, ax=ax, edgecolor='black')
    ax.axis('off')
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs('plots', exist_ok=True)

    # Save the map to a file
    plt.savefig('plots/win_probability_map.png', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
    processed_file = 'processed_data/processed_polls.csv'
    output_file = 'processed_data/president_polls_daily.csv'

    storage = []

    get_polling_data(url, processed_file)

    polling_averages_string = create_national_polling_averages(processed_file, output_file)
    storage.append(polling_averages_string)

    close_states_string = create_state_polling_averages()
    storage.append(close_states_string)

    electoral_college_votes_list = simulate_electoral_votes()
    storage.extend(electoral_college_votes_list)

    generate_plots('processed_data/president_polls_daily.csv', 'processed_data/simulated_national_election_outcomes_correlated.csv')

    generate_map('processed_data/biden_win_probabilities.csv', 'raw_data/cb_2023_us_state_500k.shp')

    print(storage)


# the closest states are x, x, and x, with win probabilities of x%, x%, and x%, respectively.
