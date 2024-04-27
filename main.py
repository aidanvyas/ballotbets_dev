"""
Download polling data, process it, and calculate daily weighted averages.

This script downloads polling data, processes it, and calculates daily weighted averages
for the candidates based on an exponential function and sample size.
"""


import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
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
        file.write("Date,Joe Biden,Donald Trump\n")

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

        # Open the output file in append mode and write the daily averages.
        with open(output_file, 'a') as file:
            file.write(f"{day.strftime('%Y-%m-%d')},{biden_avg},{trump_avg}\n")


def create_state_polling_averages():

    # Read in the csv files.
    past_results = pd.read_csv('raw_data/raw_past_results.csv')
    national_polling = pd.read_csv('president_polls_daily.csv')
    state_polling = pd.read_csv('processed_polls.csv')

    # Remove national polls from the state polling data.
    states = past_results[past_results['Location'] != 'National']['Location'].unique()

    # Convert dates to datetime objects.
    national_polling['Date'] = pd.to_datetime(national_polling['Date'])
    state_polling['end_date'] = pd.to_datetime(state_polling['end_date'])

    # Set the start and end dates for the state polling averages.
    start_date = national_polling['Date'].min() + timedelta(days=14)
    end_date = national_polling['Date'].max()

    # Create a date range from the start to end date.
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame to store the state polling averages.
    state_averages = pd.DataFrame(index=date_range, columns=['Joe Biden', 'Donald Trump'])

    # Create a directory to store the state polling averages.
    os.makedirs('state_polling_averages', exist_ok=True)

    # Iterate through the states.
    for state in states:

        # Get the polling and past results for the state.
        state_polls = state_polling[state_polling['state'] == state]
        state_past_results = past_results[past_results['Location'] == state]

        # Get the past shares for Biden and Trump.
        biden_past_share = state_past_results['Biden Share'].values[0]
        trump_past_share = state_past_results['Trump Share'].values[0]

        # Iterate through the date range.
        for date in date_range:

            # Calculate the estimated poll date.
            estimated_poll_date = date - timedelta(days=14)

            national_polls_to_date = national_polling[national_polling['Date'] <= date]
            current_total = national_polls_to_date.iloc[-1]['Joe Biden'] + national_polls_to_date.iloc[-1]['Donald Trump']

            national_past_results = past_results[past_results['Location'] == 'National']
            biden_past_national_share = national_past_results['Biden Share'].values[0]
            trump_past_national_share = national_past_results['Trump Share'].values[0]

            biden_boost = (national_polls_to_date.iloc[-1]['Joe Biden'] / current_total) / biden_past_national_share
            trump_boost = (national_polls_to_date.iloc[-1]['Donald Trump'] / current_total) / trump_past_national_share

            biden_estimated_share = biden_boost * biden_past_share * current_total
            trump_estimated_share = trump_boost * trump_past_share * current_total

            state_polls_to_date = state_polls[state_polls['end_date'] <= date].copy()

            if not state_polls_to_date.empty:
                state_polls_to_date.loc[:, 'weight'] = np.exp(-LAMBDA * (date - state_polls_to_date['end_date']).dt.days)
                state_polls_to_date.loc[:, 'weight'] /= state_polls_to_date['weight'].sum()
                biden_state_avg = (state_polls_to_date['Joe Biden'] * state_polls_to_date['weight']).sum()
                trump_state_avg = (state_polls_to_date['Donald Trump'] * state_polls_to_date['weight']).sum()
            else:
                biden_state_avg = biden_estimated_share
                trump_state_avg = trump_estimated_share

            estimated_poll = pd.DataFrame({
                'state': [state],
                'end_date': [estimated_poll_date],
                'Joe Biden': [biden_estimated_share],
                'Donald Trump': [trump_estimated_share]
            })
            state_polls = pd.concat([state_polls, estimated_poll], ignore_index=True)

            state_averages.loc[date, 'Joe Biden'] = biden_state_avg
            state_averages.loc[date, 'Donald Trump'] = trump_state_avg

        # Reset the index and rename the index column to 'Date'
        state_averages.reset_index(inplace=True)
        state_averages.rename(columns={'index': 'Date'}, inplace=True)

        # Reorder the columns to have 'Date' as the first column
        columns = ['Date'] + [col for col in state_averages.columns if col != 'Date']
        state_averages = state_averages[columns]

        # Save the state polling averages to the CSV file
        csv_file = f'state_polling_averages/{state}_polling_averages.csv'
        state_averages.to_csv(csv_file, columns=['Date', 'Joe Biden', 'Donald Trump'], index=False)

        print(f"Finished processing {state} polling data.")


def calculate_victory_probabilities(csv_directory, standard_deviation):
    """
    Calculate the probabilistic chance of victory for each candidate in each state's CSV file.

    Args:
        csv_directory (str): The directory containing the state CSV files.
        standard_deviation (float): The standard deviation to use for the normal distribution.
    """
    for csv_file in os.listdir(csv_directory):
        if csv_file.endswith('_polling_averages.csv'):
            csv_path = os.path.join(csv_directory, csv_file)
            state_averages = pd.read_csv(csv_path)

            # Calculate the margin and the probabilistic chance of victory
            state_averages['Margin'] = state_averages['Joe Biden'] - state_averages['Donald Trump']
            state_averages['Biden Victory Chance'] = norm.cdf(state_averages['Margin'] / standard_deviation)
            state_averages['Trump Victory Chance'] = 1 - state_averages['Biden Victory Chance']

            # Save the updated DataFrame back to the CSV file
            state_averages.to_csv(csv_path, index=False)

            print(f"Updated victory probabilities in {csv_file}")

def simulate_election(csv_directory, electoral_votes_csv, num_simulations=10000):
    """
    Simulate the election outcome based on the probabilistic chances of victory in each state.

    Args:
        csv_directory (str): The directory containing the state CSV files with victory probabilities.
        electoral_votes_csv (str): The CSV file containing the electoral votes for each state.
        num_simulations (int): The number of simulations to run. Default is 10,000.
    """
    # Read the electoral votes data
    electoral_votes_df = pd.read_csv(electoral_votes_csv)
    electoral_votes = electoral_votes_df.set_index('State')['ElectoralVotes'].to_dict()

    # Preload the latest victory chances for each state
    state_victory_chances = {}
    for csv_file in os.listdir(csv_directory):
        if csv_file.endswith('_polling_averages.csv'):
            csv_path = os.path.join(csv_directory, csv_file)
            state_averages = pd.read_csv(csv_path)
            state = csv_file.replace('_polling_averages.csv', '').replace('DC', 'District of Columbia')
            state_victory_chances[state] = {
                'Biden': state_averages['Biden Victory Chance'].iloc[-1],
                'Trump': state_averages['Trump Victory Chance'].iloc[-1]
            }

    # Initialize counters for the number of wins for each candidate
    biden_wins = 0
    trump_wins = 0

    # Run the simulations
    for _ in range(num_simulations):
        biden_electoral_votes = 0
        trump_electoral_votes = 0

        # Simulate the election outcome for each state
        for state, victory_chances in state_victory_chances.items():
            if np.random.rand() < victory_chances['Biden']:
                biden_electoral_votes += electoral_votes[state]
            else:
                trump_electoral_votes += electoral_votes[state]

        # Determine the winner of the simulation
        if biden_electoral_votes > trump_electoral_votes:
            biden_wins += 1
        else:
            trump_wins += 1

    # Calculate the percent chance of victory for each candidate
    biden_victory_percent = (biden_wins / num_simulations) * 100
    trump_victory_percent = (trump_wins / num_simulations) * 100

    # Output the results to a CSV file
    results_df = pd.DataFrame({
        'Candidate': ['Joe Biden', 'Donald Trump'],
        'Chance of Victory (%)': [biden_victory_percent, trump_victory_percent]
    })
    results_df.to_csv('/home/ubuntu/ballotbets_dev/state_polling_averages/election_simulation_results.csv', index=False)

    print(f"Simulation complete. Biden's chance of victory: {biden_victory_percent}%, Trump's chance of victory: {trump_victory_percent}%")

if __name__ == '__main__':
    url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
    processed_file = 'processed_polls.csv'
    output_file = 'president_polls_daily.csv'

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

    calculate_victory_probabilities('state_polling_averages', STANDARD_DEVIATION)
    simulate_election('state_polling_averages', '/home/ubuntu/ballotbets_dev/raw_data/electoral_college_votes_2024.csv')
