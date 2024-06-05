import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

def download_wav_files(credentials_file, download_string, base_directory, output_directory):
    ''' 
    Example Usage:
    download_data.download_wav_files('./credentials.json', 'dXXXsA1r01p0120210823.wav', 'afrl-uav-detection-data/DataStores/Escape_Acoustic_Data/ESII_from_Z/ESCAPE_FORMAT_ONECHANNEL', 'working_data')


    Inputs:
        - credentials_file: filepath to credentials json
        - download_string: example: 'dXXXsA1r01p0120210823.wav'
        - base_directory: Google Drive base directory to pull from
        - output_directory: filepath to output directory
    Outputs:
        - N/A: Stores downloaded files in output_directory.
    ''' 
    # Set up the Drive API credentials

    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive'])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file,
                scopes=['https://www.googleapis.com/auth/drive'],
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')
            print('Please go to this URL and authorize the application:', auth_url)
            auth_code = input('Enter the authorization code: ')
            flow.fetch_token(code=auth_code)
            creds = flow.credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)


    parts = download_string.split("s")
    series = "s" + parts[1][:5]
    p_level = parts[1][5:8]  # Corrected extraction of 'p01'

    print(f"Series: {series}")
    print(f"P-level: {p_level}")

    file_structure = os.path.join(base_directory, series, p_level)

    print(f"File structure: {file_structure}")

    try:
        query = f"name='{p_level}' and mimeType='application/vnd.google-apps.folder'"
        print(f"Searching for folder: {query}")
        results = service.files().list(q=query, fields="files(id)").execute()
        folders = results.get("files", [])

        if not folders:
            print(f"Folder '{p_level}' not found in the file structure: {file_structure}")
        else:
            parent_folder_id = folders[0]["id"]
            print(f"Found parent folder: {parent_folder_id}")

            query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name contains 'd30'"
            print(f"Searching for 'd30X' folders: {query}")
            results = service.files().list(q=query, fields="files(id)").execute()
            d30x_folders = results.get("files", [])

            if not d30x_folders:
                print("No 'd30X' folders found.")
            else:
                print(f"Found {len(d30x_folders)} 'd30X' folders.")
                os.makedirs(output_directory, exist_ok=True)

                for folder in d30x_folders:
                    folder_id = folder["id"]
                    print(f"Searching in folder: {folder_id}")
                    query = f"'{folder_id}' in parents and trashed=false and name contains '.wav'"
                    print(f"Searching for .wav files: {query}")
                    results = service.files().list(q=query, fields="files(id, name)").execute()
                    files = results.get("files", [])

                    if not files:
                        print("No .wav files found.")
                    else:
                        print(f"Found {len(files)} .wav files.")
                        for file in files:
                            file_id = file["id"]
                            file_name = file["name"]
                            output_path = os.path.join(output_directory, file_name)
                            print(f"Downloading: {file_name}")
                            request = service.files().get_media(fileId=file_id)
                            fh = open(output_path, "wb")
                            downloader = MediaIoBaseDownload(fh, request)
                            done = False
                            while done is False:
                                status, done = downloader.next_chunk()
                                print(f"Download progress: {int(status.progress() * 100)}%")
                            fh.close()
                            print(f"Downloaded: {file_name}")

    except HttpError as error:
        print(f"An error occurred: {error}")
'''
# Example usage
credentials_file = "path/to/your/credentials.json"
download_string = "dXXXsA1r01p0120210823.wav"
base_directory = "*/afrl-uav-detection-data/DataStores/Escape_Acoustic_Data/ESII_from_Z/ESCAPE_FORMAT_ONECHANNEL"
output_directory = "path/to/output/directory"

download_wav_files(credentials_file, download_string, base_directory, output_directory)
'''
