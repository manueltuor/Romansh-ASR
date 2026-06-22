import requests
import json
import os
import time
from datetime import datetime

class RTRLinguisticDownloader:
    def __init__(self, consumer_key, consumer_secret):
        self.consumer_key = consumer_key.strip()
        self.consumer_secret = consumer_secret.strip()
        self.base_url = "https://api.srgssr.ch"
        self.access_token = None
        self._authenticate()

    def _authenticate(self):
        print("Getting access token...")
        token_url = f"{self.base_url}/oauth/v1/accesstoken"
        params = {'grant_type': 'client_credentials'}
        response = requests.post(token_url, params=params,
                                 auth=(self.consumer_key, self.consumer_secret), timeout=10)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            print(f"Token obtained: {self.access_token[:20]}...")
        else:
            raise Exception(f"Auth failed: {response.status_code} - {response.text}")

    def download_file(self, pre_signed_url, save_path):
        print(f"\nDownloading to: {save_path}")
        response = requests.get(pre_signed_url, stream=True, timeout=60)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            if total_size:
                size_mb = total_size / (1024 * 1024)
                print(f"File size: {size_mb:.2f} MB")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            downloaded = 0
            start_time = time.time()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size and time.time() - start_time > 1:
                            percent = (downloaded / total_size) * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / elapsed / (1024 * 1024) if elapsed > 0 else 0
                            print(f"   {percent:.1f}% | {speed:.1f} MB/s", end='\r')
            elapsed = time.time() - start_time
            final_mb = downloaded / (1024 * 1024)
            print(f"   Downloaded {final_mb:.2f} MB in {elapsed:.1f}s")
            return save_path
        else:
            raise Exception(f"Download failed: {response.status_code}")

    def get_metadata(self):
        """Get metadata for all languages."""
        print("\nGetting metadata...")
        
        metadata_url = f"{self.base_url}/rtr-linguistic/v1/meta"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        
        params = {'language': 'All'}
        
        response = requests.get(metadata_url, headers=headers, params=params, timeout=10)
        
        print(f"Metadata status: {response.status_code}")
        
        if response.status_code == 200:
            metadata = response.json()
            
            print(f"Metadata keys: {list(metadata.keys())}")
            
            if 'metadata' in metadata:
                print(f"'metadata' key found, checking structure...")
                
                if isinstance(metadata['metadata'], dict):
                    if 'metadata' in metadata['metadata']:
                        lang_data = metadata['metadata']['metadata']
                        print(f"Found {len(lang_data)} languages in nested structure")
                        return {'languages': lang_data, 'raw': metadata}
                    else:
                        lang_data = metadata['metadata']
                        print(f"Found {len(lang_data)} languages in direct structure")
                        return {'languages': lang_data, 'raw': metadata}
            
            for key in metadata.keys():
                if isinstance(metadata[key], dict):
                    print(f"Checking key '{key}'...")
                    lang_names = ["Rumantsch Grischun", "Sursilvan", "Vallader", 
                                 "Puter", "Sutsilvan", "Surmiran"]
                    
                    for lang in lang_names:
                        if lang in metadata[key]:
                            print(f"Found language data in '{key}'")
                            return {'languages': metadata[key], 'raw': metadata}
            
            print(f"Unexpected structure, returning raw data")
            print(f"Full response preview: {json.dumps(metadata, indent=2)[:500]}...")
            return {'languages': {}, 'raw': metadata}
            
        else:
            print(f"Response: {response.text[:200]}")
            raise Exception(f"Metadata failed: {response.status_code}")

    def get_all_data(self, download_folder="./raw-data"):
        print("\n" + "=" * 60)
        print("STARTING DOWNLOAD OF ALL LANGUAGES")
        print("=" * 60)

        result = self.get_metadata()
        lang_data = result.get('languages', {})
        if not lang_data:
            print("No language data found in metadata.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_folder = os.path.join(download_folder, f"download_{timestamp}")
        os.makedirs(download_folder, exist_ok=True)

        print(f"\nDownload folder: {download_folder}")
        print(f"Found {len(lang_data)} languages to download")

        print("\nAvailable languages:")
        for i, (lang, info) in enumerate(lang_data.items(), 1):
            if isinstance(info, dict):
                print(f"{i:2d}. {lang}")
                if 'name' in info:
                    print(f"    File: {info['name']}")

        successful = []
        failed = []

        for lang, info in lang_data.items():
            if isinstance(info, dict) and 'url' in info:
                print(f"\n{'='*40}")
                print(f"DOWNLOADING: {lang}")
                print(f"{'='*40}")
                pre_signed_url = info['url']
                filename = info.get('name', f"{lang.replace(' ', '_')}.tgz")
                save_path = os.path.join(download_folder, filename)
                try:
                    self.download_file(pre_signed_url, save_path)
                    successful.append((lang, save_path))
                except Exception as e:
                    print(f"Failed to download {lang}: {e}")
                    failed.append((lang, str(e)))
            else:
                print(f"\nSkipping {lang}: No URL or invalid data")
                failed.append((lang, "No URL in metadata"))

        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)

        print(f"\nSuccessful: {len(successful)}")
        for lang, path in successful:
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"   {lang}: {size:.2f} MB")

        if failed:
            print(f"\nFailed: {len(failed)}")
            for lang, error in failed:
                print(f"   {lang}: {error}")

        print(f"\nAll files saved in: {os.path.abspath(download_folder)}")