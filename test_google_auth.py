import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_core.documents import Document

def _get_drive_service(creds_path):
    """Singleton to authenticate only once"""
    creds = service_account.Credentials.from_service_account_file(
        creds_path, 
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def load_drive_folder(folder_id, creds_path, service=None):
    """
    Recursively loads GDocs from a folder and its subfolders.
    Returns a list of LangChain Document objects.
    """
    # 1. Authenticate (only on the first call)
    if service is None:
        service = _get_drive_service(creds_path)

    print(f"📂 Scanning folder ID: {folder_id}...")
    
    docs = []
    
    # 2. List all items in this specific folder
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)",
            pageSize=100
        ).execute()
        files = results.get('files', [])
    except Exception as e:
        print(f"   ⚠️ Error accessing folder {folder_id}: {e}")
        return []

    # 3. Process each item
    for file in files:
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']

        # CASE A: It is a Google Doc -> EXPORT IT
        if mime == 'application/vnd.google-apps.document':
            try:
                # print(f"   📄 Found GDoc: {name}")
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
                content = request.execute().decode('utf-8')
                
                # Create LangChain Document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"https://docs.google.com/document/d/{file_id}",
                        "title": name,
                        "id": file_id
                    }
                )
                docs.append(doc)
            except Exception as e:
                print(f"   ❌ Failed to read {name}: {e}")

        # CASE B: It is a Folder -> DIVE DEEPER (Recursion)
        elif mime == 'application/vnd.google-apps.folder':
            # print(f"   Lr found subfolder: {name} -> Diving in...")
            # Recursive call!
            sub_docs = load_drive_folder(file_id, creds_path, service)
            docs.extend(sub_docs)

        # CASE C: Other files (PDFs, etc) - Add here if needed
        else:
            pass 
            # print(f"   ⚠️ Skipping {name} ({mime})")

    return docs

# --- TEST BLOCK (Only runs if you run this file directly) ---
if __name__ == "__main__":
    FOLDER_ID = "1f04r-mTCqEq0hYqTR5YK8ue0f7NRBT1e"
    CREDS = "credentials.json"
    
    all_docs = load_drive_folder(FOLDER_ID, CREDS)
    print(f"\n✅ SUCCESS: Loaded {len(all_docs)} documents total (including subfolders).")