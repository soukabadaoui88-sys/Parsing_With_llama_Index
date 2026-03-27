import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
BASE_URL = "https://api.cloud.llamaindex.ai"

def parse_to_json(file_path, output_json=None):
    """Parse un fichier PDF et sauvegarde le résultat en JSON avec metadata"""
    
    print(f" Parsing: {Path(file_path).name}")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    
    # Étape 1: Upload et création du job
    print(" Upload...")
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/pdf')}
        
        data = {
            'tier': 'agentic_plus',
            'version': 'latest',
            'language': 'fr'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/parsing/upload",
            headers=headers,
            files=files,
            data=data
        )
    
    if response.status_code != 200:
        print(f" Upload failed: {response.text}")
        return None
    
    result = response.json()
    job_id = result.get("job_id") or result.get("id")
    print(f"   Job ID: {job_id}")
    
    # Étape 2: Attendre la fin du traitement
    print(" Processing...")
    while True:
        status_resp = requests.get(
            f"{BASE_URL}/api/v1/parsing/job/{job_id}",
            headers=headers
        )
        
        if status_resp.status_code != 200:
            print(f" Status check failed: {status_resp.text}")
            return None
        
        status_data = status_resp.json()
        status = status_data.get("status")
        
        print(f"   Status: {status}")
        
        if status == "SUCCESS":
            print(" Parsing complete!")
            break
        elif status == "ERROR":
            print(f" Error: {status_data.get('error_message', 'Unknown error')}")
            return None
        else:
            time.sleep(2)
    
    # Étape 3: Récupérer le JSON
    print(" Fetching JSON result...")
    json_resp = requests.get(
        f"{BASE_URL}/api/v1/parsing/job/{job_id}/result/json",
        headers=headers
    )
    
    if json_resp.status_code != 200:
        print(f" Failed to get JSON: {json_resp.status_code}")
        return None
    
    parsed_content = json_resp.json()
    
    # Déterminer le nom du fichier de sortie
    if output_json is None:
        output_json = Path(file_path).stem + ".json"
    
    # Sauvegarder le JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(parsed_content, f, ensure_ascii=False, indent=2)
    
    print(f"\n JSON saved to: {output_json}")
    
    # Aperçu
    print("\n" + "="*60)
    print(" PREVIEW (first 500 chars):")
    print("="*60)
    print(json.dumps(parsed_content, ensure_ascii=False)[:500])
    print("="*60)
    
    return output_json

def main():
    file_path = "test2_simple.pdf"
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    output = parse_to_json(file_path, "cours_machine_learning.json")
    
    if output:
        print(f"\n Done! Your JSON file is ready: {output}")
    else:
        print(" Extraction failed")

if __name__ == "__main__":
    main()