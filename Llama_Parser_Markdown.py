import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
BASE_URL = "https://api.cloud.llamaindex.ai"

def parse_to_markdown(file_path, output_md=None):
    """Parse un fichier et sauvegarde directement le résultat en Markdown"""
    
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
    
    # Étape 3: Récupérer le markdown
    print(" Fetching markdown...")
    markdown_resp = requests.get(
        f"{BASE_URL}/api/v1/parsing/job/{job_id}/result/markdown",
        headers=headers
    )
    
    if markdown_resp.status_code != 200:
        print(f" Failed to get markdown: {markdown_resp.status_code}")
        return None
    
    markdown_content = markdown_resp.text
    
    # Nettoyer le markdown (enlever les guillemets JSON si présents)
    try:
        # Parfois le markdown est retourné comme une string JSON
        parsed = json.loads(markdown_content)
        if isinstance(parsed, dict) and "markdown" in parsed:
            markdown_content = parsed["markdown"]
    except:
        pass
    
    # Déterminer le nom du fichier de sortie
    if output_md is None:
        output_md = Path(file_path).stem + ".md"
    
    # Sauvegarder le markdown
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n Markdown saved to: {output_md}")
    
    # Afficher un aperçu
    print("\n" + "="*60)
    print(" PREVIEW (first 500 characters):")
    print("="*60)
    print(markdown_content[:500])
    print("="*60)
    print(f"\n Total size: {len(markdown_content):,} characters")
    
    return output_md

def main():
    # Fichier à parser
    file_path = "test2_simple.pdf"
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    # Parser et sauvegarder en markdown
    output = parse_to_markdown(file_path, "cours_machine_learning.md")
    
    if output:
        print(f"\n Done! Your markdown file is ready: {output}")
        print("\n You can now:")
        print("   - Open it with any text editor")
        print("   - View it formatted with a markdown viewer")
        print("   - Use it in your documentation")
    else:
        print(" Extraction failed")

if __name__ == "__main__":
    main()