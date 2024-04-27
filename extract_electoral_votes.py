import pandas as pd
from bs4 import BeautifulSoup

# Assuming the HTML content is saved in a file named 'electoral_college_votes_2024.html' in the 'raw_data' directory
html_file_path = 'raw_data/electoral_college_votes_2024.html'
csv_file_path = 'raw_data/electoral_college_votes_2024.csv'

# Load the HTML content
with open(html_file_path, 'r') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find the specific table with the electoral votes by its class
table = soup.find('table', {'class': 'marqueetable endorsements collapsible'})

# Extract the rows from the table
rows = table.find_all('tr')

# Initialize a list to hold the data
data = []

# Iterate over the rows and extract the state and electoral votes
for row in rows[1:]:  # Skip the header row
    cols = row.find_all('td')
    if len(cols) >= 2:  # Ensure there are at least two columns in the row
        state = cols[0].text.strip()
        votes = int(cols[1].text.strip())
        data.append({'State': state, 'ElectoralVotes': votes})

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
