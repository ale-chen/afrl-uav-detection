import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def download_wav_files(credentials_file, download_string, base_directory):
    # Set up the Drive API credentials
    creds = Credentials.from_authorized_user_file(credentials_file, ["https://www.googleapis.com/auth/drive"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("Invalid credentials. Please provide a valid credentials file.")
            return

    # Create a Drive API client
    service = build("drive", "v3", credentials=creds)

    # Extract the necessary information from the download string
    parts = download_string.split("s")
    series = "s" + parts[1][:5]
    p_level = "p" + parts[1][5:7]

    # Construct the file structure path
    file_structure = os.path.join(base_directory, series, p_level)

    try:
        # Search for the parent folder
        query = f"name='{p_level}' and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query, fields="files(id)").execute()
        folders = results.get("files", [])

        if not folders:
            print(f"Folder '{p_level}' not found.")
        else:
            parent_folder_id = folders[0]["id"]

            # Search for the .wav files within the parent folder
            query = f"'{parent_folder_id}' in parents and mimeType='audio/wav'"
            results = service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get("files", [])

            if not files:
                print("No .wav files found.")
            else:
                # Download each file
                for file in files:
                    file_id = file["id"]
                    file_name = file["name"]
                    if file_name.startswith("d") and file_name.endswith(".wav"):
                        request = service.files().get_media(fileId=file_id)
                        fh = open(file_name, "wb")
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while done is False:
                            status, done = downloader.next_chunk()
                            print(f"Downloading {file_name}: {int(status.progress() * 100)}%")
                        fh.close()
                        print(f"Downloaded: {file_name}")

    except HttpError as error:
        print(f"An error occurred: {error}")


# Example usage
credentials_file = "path/to/your/credentials.json"
download_string = "dXXXsA1r01p0120210823.wav"
base_directory = "My Drive/afrl-uav-detection-data/DataStores/Escape_Acoustic_Data/ESII_from_Z/ESCAPE_FORMAT_ONECHANNEL"

download_wav_files(credentials_file, download_string, base_directory)
