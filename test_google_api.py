import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

def test_working_directory(credentials_file):
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

    service = build('drive', 'v3', credentials=creds)

    # Get the root folder information
    root_folder = service.files().get(fileId='root', fields='id, name').execute()
    root_folder_id = root_folder['id']
    root_folder_name = root_folder['name']

    print(f"Current working directory: {root_folder_name} (ID: {root_folder_id})")

# Example usage
credentials_file = './credentials.json'
test_working_directory(credentials_file)
