import requests
import json
from configurations.config import CHANNELS, headers

limit = 100

def fetch_discord_messages(channel_type: str, output_file: str):
    channel_id = CHANNELS.get(channel_type)
    if not channel_id:
        print(f"Channel type '{channel_type}' not found in configuration.")
        return

    url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"


    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        messages = response.json()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"{len(messages)} messages saved to '{output_file}'")
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)
