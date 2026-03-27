import re
import json

# ---------------------------
# Config
# ---------------------------
MAX_WORDS_PER_CHUNK = 300  # tu peux ajuster selon taille de contexte LLM
INPUT_MD_FILE = "cours_machine_learning.md"
SOURCE_FILE = "cours_ml.pdf"
OUTPUT_JSON_FILE = "chunks_with_metadata.json"

# ---------------------------
# Étape 1 : split markdown en sections
# ---------------------------
def split_markdown_sections(md_text):
    """
    Split le markdown en sections basées sur #, ##, ###
    """
    pattern = r"(#{1,3} .+)"
    parts = re.split(pattern, md_text)

    sections = []
    current = {"title": None, "content": ""}

    for part in parts:
        if re.match(pattern, part):
            if current["title"]:
                sections.append(current)
            current = {"title": part.strip("# ").strip(), "content": ""}
        else:
            current["content"] += part.strip() + "\n"

    if current["title"]:
        sections.append(current)

    return sections

# ---------------------------
# Étape 2 : chunker le texte
# ---------------------------
def chunk_text(text, max_words=MAX_WORDS_PER_CHUNK):
    """
    Découpe un texte en chunks de max_words
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)

    return chunks

# ---------------------------
# Étape 3 : créer les chunks avec metadata
# ---------------------------
def build_chunks(md_text, file_name):
    sections = split_markdown_sections(md_text)
    all_chunks = []
    chunk_id = 0
    current_chapter = None

    for sec in sections:
        title = sec["title"]

        # Détecter chapitre vs section
        if title.lower().startswith("chapitre") or title.lower().startswith("chapter"):
            current_chapter = title
            continue

        text_chunks = chunk_text(sec["content"])

        for chunk in text_chunks:
            chunk_id += 1
            all_chunks.append({
                "chunk_id": f"{file_name}_{chunk_id}",
                "text": f"{title}\n\n{chunk}",  # inclure le titre dans le chunk
                "section": title,
                "chapter": current_chapter,
                "source_file": file_name,
                "page": None,  # à enrichir si tu as info de page
                "tags": []     # sera rempli après
            })

    return all_chunks

# ---------------------------
# Étape 4 : ajouter des tags simples (exemple basé sur mots-clés)
# ---------------------------
def add_tags(chunk):
    tags = []

    keywords_tags = {
        "régression": "regression",
        "classification": "classification",
        "machine learning": "ml",
        "supervisé": "supervised",
        "non supervisé": "unsupervised",
    }

    text_lower = chunk["text"].lower()
    for kw, tag in keywords_tags.items():
        if kw in text_lower:
            tags.append(tag)

    chunk["tags"] = tags
    return chunk

# ---------------------------
# Étape 5 : Pipeline complet
# ---------------------------
def main():
    with open(INPUT_MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    chunks = build_chunks(md_text, SOURCE_FILE)
    chunks = [add_tags(c) for c in chunks]

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(chunks)} chunks created and saved to {OUTPUT_JSON_FILE}")

# ---------------------------
# Lancer
# ---------------------------
if __name__ == "__main__":
    main()