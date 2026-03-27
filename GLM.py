import subprocess

MODEL_NAME = "haervwe/GLM-4.6V-Flash-9B:latest"
PROMPT_FILE = "prompt_GLM.txt"
FILE_PATH = "cours_machine_learning.json"

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def run_ollama(prompt_file, extra_text=""):
    # Lire le prompt depuis le fichier
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Ajouter le contenu du fichier source
    full_prompt = prompt_text + "\n\n" + extra_text

    # Préparer la commande : le prompt est passé directement comme argument
    cmd = ["ollama", "run", MODEL_NAME, full_prompt]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'appel au modèle Ollama :")
        print(e.stderr)
        return None

def main():
    file_content = read_file(FILE_PATH)
    print(" Génération en cours...")

    output = run_ollama(PROMPT_FILE, extra_text=f"Fichier source :\n{file_content}")

    if output:
        print("\n=== Résultat du modèle ===\n")
        print(output)
    else:
        print(" Échec de génération.")

if __name__ == "__main__":
    main()