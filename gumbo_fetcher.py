import requests
import json

def fetch_gumbo_data(game_pk):
    """Fetches GUMBO data for a given game ID."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


if __name__ == "__main__": 
    # Replace with an actual game ID (e.g., from today's games if there are any)
    game_id = 748565 # Example game id

    gumbo_data = fetch_gumbo_data(game_id)

    if gumbo_data:
        # Print the JSON structure to understand it
        print(json.dumps(gumbo_data, indent=4))  # Pretty print with indent