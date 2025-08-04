#!/usr/bin/env python3
"""
Analyseur de thématiques automatique avec génération HTML colorée
Utilise semantic-text-splitter, fastembed et BERTopic pour une analyse de pointe.
"""

import colorsys
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

# --- Dépendances requises ---
try:
    from semantic_text_splitter import TextSplitter
except ImportError:
    print("Installation requise: pip install semantic-text-splitter")
    exit(1)

try:
    from bertopic import BERTopic
    from bertopic.backend import FastEmbedBackend
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    print("Installation requise: pip install bertopic scikit-learn")
    exit(1)

try:
    import pandas as pd
except ImportError:
    print("Installation requise: pip install pandas")
    exit(1)

try:
    from stop_words import get_stop_words
except ImportError:
    print("Installation requise: pip install stop-words")
    exit(1)


@dataclass
class ThematicChunk:
    text: str
    theme: str
    confidence: float
    start_pos: int
    end_pos: int


class ThematicAnalyzer:
    def __init__(self):
        """Initialise l'analyseur avec les modèles et configurations optimisés."""

        # 1. Configuration du CountVectorizer avec les stop words français
        print("Configuration du vectorizer pour le français...")
        try:
            french_stop_words = get_stop_words("fr")
            vectorizer_model = CountVectorizer(stop_words=french_stop_words)
        except Exception as e:
            print(f"Impossible de charger les stop words français: {e}")
            print("Utilisation d'un vectorizer sans stop words.")
            vectorizer_model = CountVectorizer()

        # 2. Création du backend d'embedding avec FastEmbed
        embedding_model_name = "jinaai/jina-embeddings-v2-base-code"
        embedding_model_name = "jinaai/jina-embeddings-v2-base-de"
        embedding_model_name = (
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        print(
            f"Configuration du backend d'embedding avec le modèle '{embedding_model_name}'..."
        )
        embedding_model = FastEmbedBackend(embedding_model_name)

        # 3. Configuration de BERTopic en lui passant le backend
        print("Initialisation de BERTopic...")
        self.topic_model = BERTopic(
            language="multilingual",
            embedding_model=embedding_model,  # On passe le backend FastEmbed
            vectorizer_model=vectorizer_model,
            min_topic_size=2,
            verbose=True,
        )

        # Configuration pour un chunking sémantique intelligent
        self.splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity=500)

    def process_text(self, text: str) -> List[ThematicChunk]:
        """Traite le texte complet, découvre les thèmes avec BERTopic et retourne les chunks thématiques."""
        print("Découpage sémantique du texte en segments...")
        chunks = self.splitter.chunks(text)
        print(f"Texte divisé en {len(chunks)} segments (documents).")

        if len(chunks) < self.topic_model.min_topic_size:
            print(
                f"Erreur: Le texte est trop court pour une analyse (moins de {self.topic_model.min_topic_size} segments)."
            )
            return [ThematicChunk(text, "Général", 1.0, 0, len(text))]

        print(
            "\nLancement de l'analyse BERTopic avec fastembed... (cela peut prendre quelques instants)"
        )
        topics, probs = self.topic_model.fit_transform(chunks)
        print("Analyse BERTopic terminée.")

        print("\nRécupération et nettoyage des noms de thèmes...")
        topic_info_df = self.topic_model.get_topic_info()
        print("Informations sur les thèmes trouvés :")
        print(topic_info_df)

        theme_map = {}
        for _, row in topic_info_df.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                theme_map[topic_id] = "Thème Divers"
            else:
                clean_name = (
                    re.sub(r"^\d+_", "", row["Name"]).replace("_", " ").capitalize()
                )
                theme_map[topic_id] = clean_name

        print(f"\nThématiques identifiées: {', '.join(list(theme_map.values()))}")

        print("\nAssemblage des résultats finaux...")
        thematic_chunks = []
        current_pos = 0
        for i, chunk in enumerate(chunks):
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(chunk)
            current_pos = end_pos

            topic_id = topics[i]
            theme = theme_map.get(topic_id, "Inconnu")
            confidence = probs[i] if probs is not None else 0.5

            thematic_chunks.append(
                ThematicChunk(
                    text=chunk,
                    theme=theme,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

        return thematic_chunks


class HTMLGenerator:
    def __init__(self):
        self.color_map = {}

    def generate_color_palette(self, themes: List[str]) -> Dict[str, str]:
        """Génère une palette de couleurs harmonieuse pour les thèmes"""
        colors = {}
        sorted_themes = sorted(themes)

        if "Thème Divers" in sorted_themes:
            colors["Thème Divers"] = "#888888"
            sorted_themes.remove("Thème Divers")

        for i, theme in enumerate(sorted_themes):
            hue = (i * (360 / (len(sorted_themes) + 1))) % 360
            saturation = 0.7
            value = 0.85
            rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors[theme] = hex_color

        return colors

    def generate_insights(self, chunks: List[ThematicChunk]) -> Dict[str, any]:
        """Génère des insights sur l'analyse thématique"""
        theme_stats = defaultdict(
            lambda: {"count": 0, "total_confidence": 0, "text_length": 0}
        )
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {}

        for chunk in chunks:
            stats = theme_stats[chunk.theme]
            stats["count"] += 1
            stats["total_confidence"] += chunk.confidence
            stats["text_length"] += len(chunk.text)

        insights = {}
        for theme, stats in theme_stats.items():
            insights[theme] = {
                "percentage": (stats["count"] / total_chunks) * 100,
                "avg_confidence": stats["total_confidence"] / stats["count"]
                if stats["count"] > 0
                else 0,
                "avg_length": stats["text_length"] / stats["count"]
                if stats["count"] > 0
                else 0,
                "chunk_count": stats["count"],
            }

        return insights

    def generate_html(
        self, chunks: List[ThematicChunk], title: str = "Analyse Thématique"
    ) -> str:
        """Génère le HTML final avec le texte coloré"""
        themes = list(set(chunk.theme for chunk in chunks))
        colors = self.generate_color_palette(themes)
        insights = self.generate_insights(chunks)

        css = self._generate_css(colors)
        content_html = ""
        for chunk in chunks:
            confidence_class = self._get_confidence_class(chunk.confidence)
            theme_color = colors.get(chunk.theme, "#DDDDDD")
            content_html += f"""<span class="chunk {confidence_class}" 
                  style="background-color: {theme_color}20; border-left-color: {theme_color};"
                  title="Thème: {chunk.theme} (Confiance: {chunk.confidence:.2f})">{self._escape_html(chunk.text)}</span>"""

        legend_html = self._generate_legend(colors, insights)

        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Analyse thématique générée par BERTopic et fastembed</p>
        </header>
        {legend_html}
        <main class="content">{content_html}</main>
        <footer>
            <p>Généré automatiquement • {len(chunks)} segments analysés • {len(themes)} thématiques identifiées</p>
        </footer>
    </div>
</body>
</html>"""
        return html

    def _generate_css(self, colors: Dict[str, str]) -> str:
        """Génère le CSS de base."""
        return """
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f4f7f6; }
            .container { max-width: 1200px; margin: 20px auto; padding: 20px; }
            header { text-align: center; margin-bottom: 30px; }
            h1 { color: #2c3e50; }
            .subtitle { color: #7f8c8d; font-size: 1.1em; }
            .legend { position: fixed; top: 20px; right: 20px; width: 320px; max-height: 80vh; overflow-y: auto; background: rgba(255, 255, 255, 0.98); backdrop-filter: blur(5px); padding: 20px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); z-index: 1000; }
            .legend h2 { color: #2c3e50; margin-bottom: 15px; font-size: 1.4em; }
            .legend-grid { display: grid; gap: 10px; }
            .legend-item { display: flex; align-items: center; padding: 10px; border-radius: 6px; }
            .legend-color { width: 18px; height: 18px; border-radius: 4px; margin-right: 12px; flex-shrink: 0; }
            .legend-text { flex-grow: 1; }
            .legend-name { font-weight: 600; color: #2c3e50; }
            .legend-stats { font-size: 0.85em; color: #7f8c8d; }
            .content { background: white; padding: 30px; padding-right: 360px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); font-size: 1.1em; line-height: 1.8; }
            .chunk { padding: 2px 6px; margin: 1px; border-radius: 4px; transition: all 0.2s ease; cursor: help; display: inline; border-left: 3px solid; }
            .chunk:hover { filter: brightness(95%); box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
            .confidence-high { opacity: 1; }
            .confidence-medium { opacity: 0.85; }
            .confidence-low { opacity: 0.7; font-style: italic; }
            footer { text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }
            @media (max-width: 992px) {
                .legend { position: relative; width: 100%; max-height: none; margin-bottom: 20px; right: 0; top: 0; }
                .content { padding-right: 30px; }
            }
        """

    def _generate_legend(self, colors: Dict[str, str], insights: Dict[str, any]) -> str:
        """Génère la légende avec les statistiques"""
        legend_items = ""
        sorted_themes = sorted(
            insights.items(), key=lambda x: x[1]["percentage"], reverse=True
        )

        for theme, stats in sorted_themes:
            color = colors.get(theme, "#DDDDDD")
            legend_items += f"""
            <div class="legend-item" style="background-color: {color}20;">
                <div class="legend-color" style="background-color: {color};"></div>
                <div class="legend-text">
                    <div class="legend-name">{self._escape_html(theme)}</div>
                    <div class="legend-stats">
                        {stats["percentage"]:.1f}% du texte ({stats["chunk_count"]} segments)
                    </div>
                </div>
            </div>"""

        return f"""
        <div class="legend">
            <h2>📊 Répartition Thématique</h2>
            <div class="legend-grid">{legend_items}</div>
        </div>"""

    def _get_confidence_class(self, confidence: float) -> str:
        """Retourne la classe CSS selon le niveau de confiance"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"

    def _escape_html(self, text: str) -> str:
        """Échappe les caractères HTML pour éviter les injections XSS."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


def main():
    """Fonction principale"""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Analyse thématique de texte avec BERTopic/fastembed et génération HTML."
    )
    parser.add_argument("input_file", help="Fichier texte à analyser")
    parser.add_argument("-o", "--output", help="Fichier HTML de sortie (optionnel)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier '{args.input_file}' n'existe pas.")
        return

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return

    if len(text.strip()) < 100:
        print("Erreur: Le texte est trop court pour une analyse pertinente.")
        return

    print(f"Analyse du fichier: {args.input_file} ({len(text)} caractères)")

    try:
        analyzer = ThematicAnalyzer()
        chunks = analyzer.process_text(text)

        print(f"\n✅ Analyse terminée: {len(chunks)} segments analysés.")

        generator = HTMLGenerator()
        title = f"Analyse de {os.path.basename(args.input_file)}"
        html = generator.generate_html(chunks, title)

        output_file = (
            args.output
            or f"{os.path.splitext(args.input_file)[0]}_analyse_bertopic.html"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ Fichier HTML généré: {output_file}")

    except Exception as e:
        print("\n--- ERREUR LORS DU TRAITEMENT ---")
        print(f"Une erreur inattendue est survenue: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
