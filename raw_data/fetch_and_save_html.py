import requests

# The URL of the Ballotpedia page containing the electoral votes by state for the 2024 presidential election
url = 'https://ballotpedia.org/Electoral_College_in_the_2024_presidential_election'

# The path to the file where the HTML content will be saved
file_path = 'electoral_college_votes_2024.html'

# Fetch the HTML content of the page
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Write the HTML content to the file
    with open(file_path, 'w') as file:
        file.write(response.text)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
