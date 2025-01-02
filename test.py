import requests


def test_gpt4all_chat_api():
    # Define the API endpoint
    url = "http://localhost:4891/v1/chat/completions"

    # Define the payload
    payload = {
        "model": "Reasoner v1",  # Replace with your model's name
        "messages": [{"role": "user", "content": "Who is Lionel Messi?"}],
        "max_tokens": 50,
        "temperature": 0.28
    }

    try:
        # Send the POST request to the API server
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            print("Server Response:", response.json())
        else:
            print(f"Error: Received status code {response.status_code}")
            print("Response content:", response.text)
    except Exception as e:
        print("Error connecting to the GPT4All API server:", e)


# Run the test
test_gpt4all_chat_api()
