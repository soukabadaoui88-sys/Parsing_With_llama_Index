import os
import sys
import fitz  # PyMuPDF
from pathlib import Path
import argparse
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
from collections import Counter

class PDFParsingQualityChecker:
    """
    Vérifie si un PDF peut être parsé correctement par LlamaParse
    Se concentre sur les obstacles techniques et la fiabilité de l'extraction
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_name = Path(pdf_path).name
        
        # Métriques critiques
        self.is_encrypted = False
        self.is_corrupted = False
        self.corruption_error = None
        
        # Métriques OCR et extraction
        self.total_pages = 0
        self.pages_extractable = 0      # Pages avec contenu extractible
        self.pages_failed = 0            # Pages qui risquent d'échouer
        self.confidence_scores = []      # Scores de confiance par page
        self.needs_ocr = False           # PDF scanné nécessitant OCR
        self.has_multiple_columns = False # Mise en page complexe
        
        # Métriques de performance
        self.estimated_processing_time = 0  # secondes estimées
        
        # Résultat final
        self.score = 0
        self.can_proceed = False
        self.quality_level = ""
        self.recommendation = ""
        
    def analyze(self) -> Dict[str, Any]:
        """Exécute l'analyse complète"""
        
        print(f"\n{'='*60}")
        print(f" VÉRIFICATION PARSING: {self.pdf_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Étape 1: Ouvrir le PDF
            doc = fitz.open(self.pdf_path)
            self.total_pages = len(doc)
            print(f" Pages: {self.total_pages}")
            
            # Vérification 1: PDF protégé ?
            if doc.is_encrypted:
                self.is_encrypted = True
                print(" PDF PROTÉGÉ PAR MOT DE PASSE")
                return self._get_results()
            
            # Étape 2: Analyser chaque page
            print("\n Analyse des pages...")
            
            for page_num in range(self.total_pages):
                page = doc[page_num]
                
                # Extraire le texte et les images
                text = page.get_text()
                text_len = len(text.strip())
                images = page.get_images()
                
                # Évaluer l'extractibilité de la page
                self._evaluate_page(page_num + 1, text_len, len(images))
                
                # Détecter les colonnes
                if not self.has_multiple_columns:
                    self._detect_columns(page)
                
                # Simuler un score de confiance
                self._calculate_page_confidence(text_len, len(images))
            
            doc.close()
            
            # Étape 3: Calculer les métriques globales
            self._calculate_global_metrics()
            
            # Étape 4: Estimer le temps de traitement
            self._estimate_processing_time()
            
            # Étape 5: Calculer le score de décision
            self._calculate_decision_score()
            
        except Exception as e:
            self.is_corrupted = True
            self.corruption_error = str(e)
            print(f" PDF CORROMPU: {e}")
            return self._get_results()
        
        elapsed = time.time() - start_time
        print(f"\n  Analyse terminée en {elapsed:.1f} secondes")
        
        return self._get_results()
    
    def _evaluate_page(self, page_num: int, text_len: int, image_count: int):
        """Évalue l'extractibilité d'une page"""
        
        # Page avec du texte
        if text_len > 50:
            self.pages_extractable += 1
            status = ""
            confidence = 0.9
        # Page avec images seulement (nécessite OCR)
        elif image_count > 0:
            self.pages_extractable += 1
            self.needs_ocr = True
            status = ""
            confidence = 0.6  # OCR moins fiable
        # Page vide
        else:
            self.pages_failed += 1
            status = ""
            confidence = 0.0
        
        # Stocker le score de confiance
        self.confidence_scores.append(confidence)
        
        # Afficher le résultat
        print(f"   Page {page_num}: {status} texte:{text_len} chars, images:{image_count}")
    
    def _detect_columns(self, page):
        """Détecte la présence de colonnes (mise en page complexe)"""
        try:
            blocks = page.get_text("dict")
            x_positions = []
            
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            x_positions.append(span["bbox"][0])
            
            if len(x_positions) > 20:
                x_positions.sort()
                # Calculer les écarts entre positions X
                gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                # S'il y a un grand écart, probablement des colonnes
                if max(gaps) > 150 and len(set([int(g/50) for g in gaps if g > 50])) > 1:
                    self.has_multiple_columns = True
                    print(f"    Colonnes détectées")
        except:
            pass
    
    def _calculate_page_confidence(self, text_len: int, image_count: int):
        """Calcule un score de confiance simulé pour la page"""
        # Cette simulation sera remplacée par les vrais scores de LlamaParse
        pass
    
    def _calculate_global_metrics(self):
        """Calcule les métriques globales"""
        
        # Taux d'échec = pages qui ne pourront pas être extraites
        if self.total_pages > 0:
            self.failure_rate = (self.pages_failed / self.total_pages) * 100
        else:
            self.failure_rate = 0
        
        # Confiance moyenne OCR (simulation basée sur le contenu)
        if self.needs_ocr:
            # PDF scanné: confiance basée sur la présence de texte
            if self.pages_extractable > 0:
                text_ratio = self.pages_extractable / self.total_pages
                self.avg_confidence = 50 + (text_ratio * 30)  # Entre 50% et 80%
            else:
                self.avg_confidence = 30  # Très faible
        else:
            # PDF texte: confiance élevée
            self.avg_confidence = 85 + (self.pages_extractable / self.total_pages * 15)
        
        self.avg_confidence = min(100, self.avg_confidence)
        
        print(f"\n RÉSULTATS DE L'ANALYSE:")
        print(f"   Pages extractibles: {self.pages_extractable}/{self.total_pages}")
        print(f"   Pages en échec: {self.pages_failed}")
        print(f"   Taux d'échec: {self.failure_rate:.1f}%")
        print(f"   Confiance estimée: {self.avg_confidence:.1f}%")
        print(f"   OCR nécessaire: {' Oui' if self.needs_ocr else ' Non'}")
        print(f"   Colonnes: {' Oui' if self.has_multiple_columns else ' Non'}")
    
    def _estimate_processing_time(self):
        """Estime le temps de traitement par LlamaParse"""
        
        # Base: 2 secondes par page pour un PDF texte
        base_time = self.total_pages * 2
        
        # Pénalités
        if self.needs_ocr:
            base_time *= 3  # L'OCR prend plus de temps
        
        if self.has_multiple_columns:
            base_time *= 1.5  # Les colonnes sont plus complexes
        
        # Pénalité pour les pages en échec (temps perdu)
        if self.pages_failed > 0:
            base_time += self.pages_failed * 1
        
        self.estimated_processing_time = base_time
        
        print(f"  Temps estimé: {base_time:.0f} secondes ({base_time/60:.1f} minutes)")
    
    def _calculate_decision_score(self):
        """Calcule le score de décision basé sur la formule validée"""
        
        # Vérifications critiques (bloquantes)
        if self.is_encrypted or self.is_corrupted:
            self.score = 0
            self.can_proceed = False
            self.quality_level = "CRITIQUE"
            self.recommendation = " Fichier illisible. Vérifiez qu'il n'est pas protégé ou corrompu."
            return
        
        # 1. Taux d'échec (30% du score)
        failure_score = (1 - (self.pages_failed / max(self.total_pages, 1))) * 30
        
        # 2. Confiance OCR (30% du score)
        confidence_score = (self.avg_confidence / 100) * 30
        
        # 3. Colonnes (15% du score) - pénalité si colonnes détectées
        columns_score = 0 if self.has_multiple_columns else 15
        
        # 4. OCR nécessaire (15% du score) - pénalité si OCR nécessaire
        ocr_score = 0 if self.needs_ocr else 15
        
        # 5. Temps de traitement (10% du score) - pénalité si trop long
        # Temps acceptable: 30 secondes, au-delà pénalité progressive
        max_acceptable_time = 30
        if self.estimated_processing_time <= max_acceptable_time:
            time_score = 10
        else:
            penalty = min(10, (self.estimated_processing_time - max_acceptable_time) / 10)
            time_score = max(0, 10 - penalty)
        
        # Score total
        self.score = failure_score + confidence_score + columns_score + ocr_score + time_score
        
        # Décision basée sur le score
        if self.score >= 80:
            self.can_proceed = True
            self.quality_level = "EXCELLENTE"
            self.recommendation = " Fichier de bonne qualité, parsing recommandé"
        elif self.score >= 60:
            self.can_proceed = True
            self.quality_level = "MOYENNE"
            self.recommendation = " Qualité acceptable mais risque d'extraction partielle. Voulez-vous continuer ?"
        elif self.score >= 40:
            self.can_proceed = False
            self.quality_level = "FRAGILE"
            self.recommendation = " Fichier fragile. Extraction risquée. Un autre fichier serait préférable."
        else:
            self.can_proceed = False
            self.quality_level = "MAUVAISE"
            self.recommendation = " Fichier non parsable. Veuillez uploader un PDF avec du texte sélectionnable."
        
        print(f"\n{'─'*60}")
        print(f" SCORE DE DÉCISION: {self.score:.1f}%")
        print(f"   Niveau: {self.quality_level}")
        print(f"   Décision: {' ACCEPTER' if self.can_proceed else '❌ REJETER'}")
    
    def _get_results(self) -> Dict[str, Any]:
        """Retourne les résultats complets"""
        
        return {
            "file": {
                "name": self.pdf_name,
                "path": self.pdf_path,
                "size_kb": round(os.path.getsize(self.pdf_path) / 1024, 2)
            },
            "metrics": {
                "pages": {
                    "total": self.total_pages,
                    "extractable": self.pages_extractable,
                    "failed": self.pages_failed
                },
                "quality": {
                    "failure_rate": round(self.failure_rate, 1) if hasattr(self, 'failure_rate') else 0,
                    "ocr_confidence": round(self.avg_confidence, 1) if hasattr(self, 'avg_confidence') else 0,
                    "needs_ocr": self.needs_ocr,
                    "has_multiple_columns": self.has_multiple_columns,
                    "estimated_seconds": round(self.estimated_processing_time, 0)
                },
                "critical_issues": {
                    "is_encrypted": self.is_encrypted,
                    "is_corrupted": self.is_corrupted,
                    "corruption_error": self.corruption_error
                }
            },
            "decision": {
                "score": round(self.score, 1),
                "level": self.quality_level,
                "can_proceed": self.can_proceed,
                "recommendation": self.recommendation
            }
        }
    
    def print_report(self, results: Dict[str, Any]):
        """Affiche un rapport clair et concis"""
        
        print(f"\n{'='*60}")
        print(" RAPPORT DE VÉRIFICATION PARSING")
        print(f"{'='*60}")
        
        # Fichier
        print(f"\n FICHIER:")
        print(f"   {results['file']['name']}")
        print(f"   Taille: {results['file']['size_kb']} KB")
        
        # Pages
        print(f"\n PAGES:")
        print(f"   Total: {results['metrics']['pages']['total']}")
        print(f"   Extractibles: {results['metrics']['pages']['extractable']}")
        print(f"   En échec: {results['metrics']['pages']['failed']}")
        
        # Métriques de qualité
        print(f"\n QUALITÉ:")
        print(f"   Taux d'échec: {results['metrics']['quality']['failure_rate']}%")
        print(f"   Confiance OCR: {results['metrics']['quality']['ocr_confidence']}%")
        print(f"   OCR nécessaire: {' Oui' if results['metrics']['quality']['needs_ocr'] else ' Non'}")
        print(f"   Colonnes: {' Oui' if results['metrics']['quality']['has_multiple_columns'] else ' Non'}")
        print(f"   Temps estimé: {results['metrics']['quality']['estimated_seconds']:.0f} sec ({results['metrics']['quality']['estimated_seconds']/60:.1f} min)")
        
        # Décision
        print(f"\n{'─'*60}")
        print(f" SCORE: {results['decision']['score']}%")
        print(f" NIVEAU: {results['decision']['level']}")
        print(f"{'─'*60}")
        print(f" {results['decision']['recommendation']}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Vérifie si un PDF peut être parsé correctement'
    )
    parser.add_argument('pdf_file', help='Chemin vers le fichier PDF')
    parser.add_argument('--json', action='store_true',
                       help='Afficher le résultat au format JSON')
    parser.add_argument('--quiet', action='store_true',
                       help='Mode silencieux (juste le code de sortie)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_file):
        print(f" Fichier non trouvé: {args.pdf_file}")
        sys.exit(1)
    
    # Analyser
    checker = PDFParsingQualityChecker(args.pdf_file)
    results = checker.analyze()
    
    # Afficher
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif not args.quiet:
        checker.print_report(results)
    
    # Code de sortie
    if results['decision']['can_proceed']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()