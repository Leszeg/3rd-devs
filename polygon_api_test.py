import requests
from os import environ


def fetch_data():
    # Get data from the first URL
    response = requests.get(environ.get("POLIGON_DATA_URL"))
    if response.status_code == 200:
        # Split the content into two strings
        data = response.text.strip().split()
        return data
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

def verify_data(data):
    # Prepare the JSON payload
    payload = {
        "task": environ.get("POLIGON_TEST_TASK"),
        "apikey": environ.get("POLIGON_API_KEY"),
        "answer": data
    }
    
    # Make POST request
    response = requests.post(environ.get("POLIGON_VERIFY_URL"), json=payload)
    return response.json()

def main():
    try:        
        # Fetch the data
        data = fetch_data()
        if len(data) != 2:
            raise Exception(f"Expected 2 strings, but got {len(data)}")
        print(data)

        # Verify the data
        result = verify_data(data)
        print("Response:", result)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
