#!/usr/bin/env python3
"""
Analyseur de thématiques automatique avec génération HTML colorée
Utilise semantic-text-splitter pour le chunking et un modèle local pour l'analyse
"""

import colorsys
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from semantic_text_splitter import TextSplitter
except ImportError:
    print("Installation requise: pip install semantic-text-splitter")
    exit(1)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except ImportError:
    print("Installation requise: pip install transformers torch")
    exit(1)

try:
    import nltk
    import numpy as np
    from nltk.cluster import KMeansClusterer
    from nltk.cluster.util import cosine_distance
    from nltk.corpus import stopwords
except ImportError:
    print("Installation requise: pip install nltk numpy scikit-learn")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installation requise: pip install sentence-transformers")
    exit(1)


@dataclass
class ThematicChunk:
    text: str
    theme: str
    confidence: float
    start_pos: int
    end_pos: int


class ThematicAnalyzer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """Initialise l'analyseur avec un modèle léger"""
        print("Chargement du modèle d'analyse...")

        # Télécharger les ressources NLTK nécessaires
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            print("Téléchargement des stop words...")
            nltk.download("stopwords", quiet=True)

        # Utilise un modèle de classification plus adapté
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
        except Exception:
            # Fallback vers un modèle plus léger
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=-1,
            )

        # Modèle pour les embeddings sémantiques
        try:
            print("Chargement du modèle d'embeddings...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            print(
                "Modèle d'embeddings non disponible, utilisation de la méthode alternative"
            )
            self.embedding_model = None

        # Configuration pour un chunking intelligent mais avec capacité flexible
        # On utilise une capacité plus grande pour permettre des chunks sémantiquement cohérents
        self.splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity=500)
        # Le splitter va naturellement diviser selon les limites sémantiques dans cette capacité

        # Stop words multilingues
        self.stop_words = set()
        for lang in ["french", "english"]:
            try:
                self.stop_words.update(stopwords.words(lang))
            except:
                print(f"Stop words pour {lang} non disponibles")

    def extract_themes_from_text(self, text: str, max_themes: int = 5) -> List[str]:
        """Extruit automatiquement les thématiques principales du texte"""

        # Analyse du vocabulaire pour identifier les concepts clés
        words = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", text.lower())
        word_freq = Counter(words)

        # Filtrer les mots fréquents significatifs avec les stop words NLTK
        meaningful_words = [
            word
            for word, freq in word_freq.most_common(50)
            if word not in self.stop_words and freq > 1
        ]

        # Grouper les mots par domaines sémantiques
        themes = []

        # Analyse par clustering sémantique automatique
        word_groups = self._cluster_words_automatically(meaningful_words[:20])

        for group in word_groups[:max_themes]:
            if len(group) >= 1:
                # Utiliser le mot le plus fréquent comme nom de thème
                theme_name = max(group, key=lambda w: word_freq[w])
                themes.append(theme_name.capitalize())

        # Si pas assez de thèmes, ajouter des thèmes basés sur l'analyse du contenu
        if len(themes) < 3:
            content_themes = self._analyze_content_themes(text)
            themes.extend(content_themes[: max_themes - len(themes)])

        return themes[:max_themes] if themes else ["Général"]

    def _cluster_words_automatically(self, words: List[str]) -> List[List[str]]:
        """Groupe les mots par similarité sémantique automatiquement"""
        if not words:
            return []

        if self.embedding_model:
            # Méthode avancée avec embeddings sémantiques
            return self._cluster_with_embeddings(words)
        else:
            # Méthode alternative basée sur la co-occurrence et la similarité textuelle
            return self._cluster_with_cooccurrence(words)

    def _cluster_with_embeddings(self, words: List[str]) -> List[List[str]]:
        """Clustering avec des embeddings sémantiques"""
        try:
            # Générer les embeddings pour chaque mot
            embeddings = self.embedding_model.encode(words)

            # Déterminer le nombre optimal de clusters (entre 2 et min(len(words)//2, 5))
            n_clusters = min(max(2, len(words) // 3), 5)

            # Clustering K-means avec NLTK
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Organiser les mots par cluster
            clusters = defaultdict(list)
            for word, label in zip(words, cluster_labels):
                clusters[label].append(word)

            return list(clusters.values())

        except Exception as e:
            print(f"Erreur clustering embeddings: {e}")
            return self._cluster_with_cooccurrence(words)

    def _cluster_with_cooccurrence(self, words: List[str]) -> List[List[str]]:
        """Clustering basé sur la similarité textuelle et la longueur"""
        clusters = []
        used_words = set()

        for word in words:
            if word in used_words:
                continue

            cluster = [word]
            used_words.add(word)

            # Chercher des mots similaires (racines communes, préfixes/suffixes)
            for other_word in words:
                if other_word in used_words:
                    continue

                # Similarité basée sur les n-grammes et la distance d'édition
                if self._words_are_similar(word, other_word):
                    cluster.append(other_word)
                    used_words.add(other_word)

            if cluster:
                clusters.append(cluster)

        return clusters

    def _words_are_similar(
        self, word1: str, word2: str, threshold: float = 0.6
    ) -> bool:
        """Détermine si deux mots sont sémantiquement similaires"""

        # Similarité basée sur les n-grammes de caractères
        def get_ngrams(word, n=2):
            return set(word[i : i + n] for i in range(len(word) - n + 1))

        ngrams1 = get_ngrams(word1)
        ngrams2 = get_ngrams(word2)

        if not ngrams1 or not ngrams2:
            return False

        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))

        similarity = intersection / union if union > 0 else 0

        # Bonus pour les racines communes
        min_len = min(len(word1), len(word2))
        if min_len >= 4:
            common_prefix = 0
            for i in range(min_len):
                if word1[i] == word2[i]:
                    common_prefix += 1
                else:
                    break

            if common_prefix >= 3:
                similarity += 0.2

        return similarity >= threshold

    def _analyze_content_themes(self, text: str) -> List[str]:
        """Analyse le contenu pour identifier des thèmes supplémentaires automatiquement"""
        themes = []

        # Extraire les entités nommées et concepts importants
        # Méthode 1: Analyse des bigrammes et trigrammes fréquents
        themes.extend(self._extract_ngram_themes(text))

        # Méthode 2: Analyse des mots composés et expressions techniques
        themes.extend(self._extract_compound_themes(text))

        # Méthode 3: Analyse contextuelle basée sur les verbes d'action
        themes.extend(self._extract_action_themes(text))

        return list(set(themes))  # Éliminer les doublons

    def _extract_ngram_themes(self, text: str, min_freq: int = 3) -> List[str]:
        """Extrait des thèmes basés sur les n-grammes fréquents"""
        themes = []

        # Nettoyer le texte
        words = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", text.lower())
        filtered_words = [w for w in words if w not in self.stop_words]

        # Extraire les bigrammes
        bigrams = []
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i + 1]}"
            bigrams.append(bigram)

        # Compter les bigrammes fréquents
        bigram_freq = Counter(bigrams)
        for bigram, freq in bigram_freq.most_common(10):
            if freq >= min_freq and len(bigram) > 8:  # Éviter les bigrammes trop courts
                # Utiliser le mot le plus significatif du bigramme
                words_in_bigram = bigram.split()
                theme = max(words_in_bigram, key=len).capitalize()
                themes.append(theme)

        return themes

    def _extract_compound_themes(self, text: str) -> List[str]:
        """Extrait des thèmes basés sur les mots composés et expressions techniques"""
        themes = []

        # Rechercher des mots avec des caractères spéciaux (indicateurs de concepts techniques)
        technical_patterns = [
            r"\b[a-zA-ZÀ-ÿ]+[-_][a-zA-ZÀ-ÿ]+\b",  # mots avec tirets ou underscores
            r"\b[A-ZÀ-Ÿ][a-zà-ÿ]+[A-ZÀ-Ÿ][a-zA-Zà-ÿ]*\b",  # CamelCase
            r"\b[a-zA-ZÀ-ÿ]*[0-9]+[a-zA-ZÀ-ÿ]*\b",  # mots avec chiffres
        ]

        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 4:  # Éviter les matches trop courts
                    # Nettoyer et capitaliser
                    clean_match = re.sub(r"[^a-zA-ZÀ-ÿ]", "", match).capitalize()
                    if clean_match and clean_match.lower() not in self.stop_words:
                        themes.append(clean_match)

        return themes

    def _extract_action_themes(self, text: str) -> List[str]:
        """Extrait des thèmes basés sur les contextes d'action et les domaines d'activité"""
        themes = []

        # Rechercher des patterns contextuels qui indiquent des domaines d'activité
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) < 20:  # Ignorer les phrases trop courtes
                continue

            # Extraire le sujet principal de la phrase (souvent le premier nom significatif)
            words = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", sentence)
            significant_words = [w for w in words if w not in self.stop_words]

            if significant_words:
                # Prendre les 2 premiers mots significatifs comme indicateurs thématiques
                for word in significant_words[:2]:
                    if len(word) >= 5:  # Mots suffisamment longs
                        themes.append(word.capitalize())

        # Retourner les thèmes les plus fréquents
        theme_freq = Counter(themes)
        return [theme for theme, freq in theme_freq.most_common(5) if freq >= 2]

    def analyze_chunk_theme(
        self, chunk: str, available_themes: List[str]
    ) -> Tuple[str, float]:
        """Analyse la thématique d'un chunk donné"""
        if not available_themes:
            return "Général", 0.5

        try:
            # Utilise la classification zero-shot si disponible
            if (
                hasattr(self.classifier, "model")
                and "bart" in str(type(self.classifier.model)).lower()
            ):
                result = self.classifier(chunk, available_themes)
                return result["labels"][0], result["scores"][0]
            else:
                # Fallback: analyse basée sur la présence de mots-clés
                chunk_lower = chunk.lower()
                scores = {}

                for theme in available_themes:
                    theme_lower = theme.lower()
                    # Score basé sur la présence du mot thème et mots associés
                    score = 0
                    if theme_lower in chunk_lower:
                        score += 0.5

                    # Compte les mots liés au thème
                    words = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", chunk_lower)
                    theme_words = [
                        word
                        for word in words
                        if theme_lower in word or word in theme_lower
                    ]
                    score += len(theme_words) * 0.1

                    scores[theme] = min(score, 1.0)

                best_theme = max(scores, key=scores.get)
                return best_theme, scores[best_theme]

        except Exception as e:
            print(f"Erreur d'analyse: {e}")
            return available_themes[0], 0.3

    def process_text(self, text: str) -> List[ThematicChunk]:
        """Traite le texte complet et retourne les chunks thématiques"""
        print("Découpage intelligent du texte...")
        # Le splitter utilise maintenant un chunking sémantique intelligent
        # Il divise selon le sens plutôt que selon une taille fixe
        chunks = self.splitter.chunks(text)
        print(f"Texte divisé en {len(chunks)} chunks sémantiques")

        print("Extraction des thématiques depuis les chunks...")
        # Approche en 2 phases : d'abord analyser les chunks, puis extraire les thèmes
        themes = self.extract_themes_from_chunks(chunks)
        print(f"Thématiques identifiées: {', '.join(themes)}")

        print("Analyse thématique des chunks...")
        thematic_chunks = []
        current_pos = 0

        for i, chunk in enumerate(chunks):
            start_pos = text.find(chunk, current_pos)
            end_pos = start_pos + len(chunk)
            current_pos = end_pos

            theme, confidence = self.analyze_chunk_theme(chunk, themes)

            thematic_chunks.append(
                ThematicChunk(
                    text=chunk,
                    theme=theme,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

            if (i + 1) % 10 == 0:
                print(f"Analysé {i + 1}/{len(chunks)} chunks...")

        return thematic_chunks

    def extract_themes_from_chunks(
        self, chunks: List[str], max_themes: int = 5
    ) -> List[str]:
        """Extrait les thématiques principales en analysant le contexte sémantique"""

        # Phase 1: Identifier le contexte global et les entités principales
        context_themes = self._extract_contextual_themes(chunks)

        # Phase 2: Analyser les concepts récurrents inter-chunks
        recurring_themes = self._extract_recurring_themes(chunks)

        # Phase 3: Extraire les thèmes à partir des relations conceptuelles
        conceptual_themes = self._extract_conceptual_themes(chunks)

        # Combiner et prioriser les thèmes
        all_themes = {}

        # Poids plus élevé pour les thèmes contextuels (le "quoi" principal)
        for theme in context_themes:
            all_themes[theme] = all_themes.get(theme, 0) + 3

        # Poids moyen pour les thèmes récurrents
        for theme in recurring_themes:
            all_themes[theme] = all_themes.get(theme, 0) + 2

        # Poids plus faible pour les thèmes conceptuels
        for theme in conceptual_themes:
            all_themes[theme] = all_themes.get(theme, 0) + 1

        # Trier par score et retourner les meilleurs
        sorted_themes = sorted(all_themes.items(), key=lambda x: x[1], reverse=True)
        final_themes = [
            theme for theme, score in sorted_themes[:max_themes] if score >= 2
        ]

        return final_themes if final_themes else ["Général"]

    def _extract_contextual_themes(self, chunks: List[str]) -> List[str]:
        """Identifie les thèmes principaux basés sur le contexte global"""
        themes = []
        combined_text = " ".join(chunks).lower()

        # Dictionnaire de contextes sémantiques avec leurs indicateurs
        contextual_patterns = {
            "Piscine": [
                r"\b(piscine|bassin|nage|natation|aquatique|baignade)\b",
                r"\b(nageur|maitre.nageur|surveillant)\b",
                r"\b(couloir|longueur|brassard)\b",
            ],
            "Horaires": [
                r"\b(horaire|heure|ouverture|fermeture|planning)\b",
                r"\b(\d{1,2}h\d{2}|\d{1,2}:\d{2})\b",
                r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b.*\b(\d|\h)\b",
            ],
            "Restaurant": [
                r"\b(restaurant|menu|plat|cuisine|repas|déjeuner|dîner)\b",
                r"\b(chef|cuisinier|serveur|service)\b",
                r"\b(carte|prix|réservation)\b",
            ],
            "Médical": [
                r"\b(médecin|docteur|patient|consultation|traitement)\b",
                r"\b(hôpital|clinique|cabinet|soin)\b",
                r"\b(rendez.vous|symptôme|diagnostic)\b",
            ],
            "Événement": [
                r"\b(événement|spectacle|concert|festival|conférence)\b",
                r"\b(billet|réservation|tarif|entrée)\b",
                r"\b(artiste|programme|scène)\b",
            ],
            "Éducation": [
                r"\b(école|université|cours|formation|étudiant)\b",
                r"\b(professeur|enseignant|élève|classe)\b",
                r"\b(diplôme|examen|programme)\b",
            ],
            "Commerce": [
                r"\b(magasin|boutique|vente|achat|produit)\b",
                r"\b(prix|promotion|solde|offre)\b",
                r"\b(client|vendeur|caisse)\b",
            ],
            "Transport": [
                r"\b(transport|bus|métro|train|avion)\b",
                r"\b(gare|station|aéroport|arrêt)\b",
                r"\b(billet|voyage|trajet)\b",
            ],
            "Sport": [
                r"\b(sport|match|équipe|joueur|entraîneur)\b",
                r"\b(terrain|stade|gymnase|club)\b",
                r"\b(compétition|championnat|tournoi)\b",
            ],
        }

        # Analyser chaque contexte
        for theme, patterns in contextual_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                score += matches

            # Seuil adaptatif basé sur la longueur du texte
            min_score = max(1, len(chunks) // 20)
            if score >= min_score:
                themes.append(theme)

        return themes

    def _extract_recurring_themes(self, chunks: List[str]) -> List[str]:
        """Identifie les concepts qui reviennent dans plusieurs chunks"""
        themes = []

        # Extraire les substantifs significatifs de chaque chunk
        chunk_nouns = []
        for chunk in chunks:
            # Identifier les mots probablement importants (substantifs, noms propres)
            words = re.findall(r"\b[A-ZÀ-Ÿ][a-zà-ÿ]{3,}\b|\b[a-zà-ÿ]{5,}\b", chunk)
            meaningful_words = [
                w.lower()
                for w in words
                if w.lower() not in self.stop_words and len(w) >= 4
            ]
            chunk_nouns.append(set(meaningful_words))

        # Trouver les mots qui apparaissent dans plusieurs chunks
        if len(chunk_nouns) > 1:
            common_words = chunk_nouns[0]
            for noun_set in chunk_nouns[1:]:
                common_words = common_words.intersection(noun_set)

            # Ajouter les mots qui apparaissent dans au moins 30% des chunks
            all_words = []
            for noun_set in chunk_nouns:
                all_words.extend(noun_set)

            word_freq = Counter(all_words)
            min_appearances = max(2, len(chunks) * 0.3)

            for word, freq in word_freq.most_common(10):
                if freq >= min_appearances and len(word) > 4:
                    themes.append(word.capitalize())

        return themes

    def _extract_conceptual_themes(self, chunks: List[str]) -> List[str]:
        """Extrait des thèmes basés sur les relations conceptuelles"""
        themes = []

        # Analyser les bigrammes et trigrammes significatifs
        all_text = " ".join(chunks).lower()

        # Extraire les expressions composées
        compound_expressions = re.findall(
            r"\b[a-zà-ÿ]{4,}\s+[a-zà-ÿ]{4,}\b|\b[a-zà-ÿ]{3,}\s+[a-zà-ÿ]{3,}\s+[a-zà-ÿ]{3,}\b",
            all_text,
        )

        # Filtrer et compter les expressions
        meaningful_expressions = []
        for expr in compound_expressions:
            words = expr.split()
            if all(word not in self.stop_words for word in words):
                meaningful_expressions.append(expr)

        expr_freq = Counter(meaningful_expressions)

        # Prendre les expressions les plus fréquentes et les transformer en thèmes
        for expr, freq in expr_freq.most_common(5):
            if freq >= 2:
                # Prendre le mot le plus long ou le plus spécifique
                words = expr.split()
                theme_word = max(words, key=len)
                if len(theme_word) > 4:
                    themes.append(theme_word.capitalize())

        return themes


class HTMLGenerator:
    def __init__(self):
        self.color_map = {}

    def generate_color_palette(self, themes: List[str]) -> Dict[str, str]:
        """Génère une palette de couleurs harmonieuse pour les thèmes"""
        colors = {}

        # Utilise HSV pour générer des couleurs distinctes
        for i, theme in enumerate(themes):
            hue = (i * 360 / len(themes)) % 360
            saturation = 0.6 + (i % 3) * 0.1  # Variation de saturation
            value = 0.8 + (i % 2) * 0.1  # Variation de luminosité

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

        for chunk in chunks:
            stats = theme_stats[chunk.theme]
            stats["count"] += 1
            stats["total_confidence"] += chunk.confidence
            stats["text_length"] += len(chunk.text)

        insights = {}
        total_chunks = len(chunks)

        for theme, stats in theme_stats.items():
            insights[theme] = {
                "percentage": (stats["count"] / total_chunks) * 100,
                "avg_confidence": stats["total_confidence"] / stats["count"],
                "avg_length": stats["text_length"] / stats["count"],
                "chunk_count": stats["count"],
            }

        return insights

    def generate_html(
        self, chunks: List[ThematicChunk], title: str = "Analyse Thématique"
    ) -> str:
        """Génère le HTML final avec le texte coloré"""

        # Extraire les thèmes uniques
        themes = list(set(chunk.theme for chunk in chunks))
        colors = self.generate_color_palette(themes)
        insights = self.generate_insights(chunks)

        # Générer le CSS
        css = self._generate_css(colors)

        # Générer le contenu HTML
        content_html = ""
        for chunk in chunks:
            confidence_class = self._get_confidence_class(chunk.confidence)
            content_html += f"""
            <span class="chunk theme-{self._sanitize_theme_name(chunk.theme)} {confidence_class}" 
                  title="Thème: {chunk.theme} (Confiance: {chunk.confidence:.2f})">
                {self._escape_html(chunk.text)}
            </span>"""

        # Générer la légende
        legend_html = self._generate_legend(colors, insights)

        # Assembler le HTML complet
        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Analyse automatique des thématiques</p>
        </header>
        
        {legend_html}
        
        <main class="content">
            {content_html}
        </main>
        
        <footer>
            <p>Généré automatiquement • {len(chunks)} segments analysés • {len(themes)} thématiques identifiées</p>
        </footer>
    </div>
</body>
</html>"""

        return html

    def _generate_css(self, colors: Dict[str, str]) -> str:
        """Génère le CSS avec les couleurs thématiques"""

        theme_styles = ""
        for theme, color in colors.items():
            sanitized_name = self._sanitize_theme_name(theme)
            # Couleur de fond légère, bordure plus foncée
            bg_color = color + "20"  # Transparence 20%
            theme_styles += f"""
            .theme-{sanitized_name} {{
                background-color: {bg_color};
                border-left: 3px solid {color};
            }}
            .theme-{sanitized_name}:hover {{
                background-color: {color}40;
                box-shadow: 0 2px 8px {color}30;
            }}"""

        return f"""
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            header {{
                text-align: center;
                margin-bottom: 30px;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }}
            
            h1 {{
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }}
            
            .subtitle {{
                color: #7f8c8d;
                font-size: 1.1em;
            }}
            
            .legend {{
                position: fixed;
                top: 20px;
                right: 20px;
                width: 320px;
                max-height: 70vh;
                overflow-y: auto;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
                z-index: 1000;
                transition: all 0.3s ease;
            }}
            
            .legend:hover {{
                background: rgba(255, 255, 255, 0.98);
                box-shadow: 0 12px 40px rgba(0,0,0,0.2);
                transform: translateY(-2px);
            }}
            
            .legend::before {{
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                background: linear-gradient(45deg, rgba(66, 165, 245, 0.3), rgba(156, 39, 176, 0.3), rgba(244, 67, 54, 0.3));
                border-radius: 17px;
                z-index: -1;
                opacity: 0;
                transition: opacity 0.3s ease;
            }}
            
            .legend:hover::before {{
                opacity: 1;
            }}
            
            /* Scrollbar personnalisée pour la légende */
            .legend::-webkit-scrollbar {{
                width: 6px;
            }}
            
            .legend::-webkit-scrollbar-track {{
                background: rgba(0,0,0,0.1);
                border-radius: 3px;
            }}
            
            .legend::-webkit-scrollbar-thumb {{
                background: rgba(0,0,0,0.3);
                border-radius: 3px;
            }}
            
            .legend::-webkit-scrollbar-thumb:hover {{
                background: rgba(0,0,0,0.5);
            }}
            
            .legend h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.5em;
                font-weight: 400;
            }}
            
            .legend-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                padding: 12px;
                border-radius: 8px;
                transition: transform 0.2s ease;
            }}
            
            .legend-item:hover {{
                transform: translateY(-2px);
            }}
            
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 4px;
                margin-right: 12px;
                flex-shrink: 0;
            }}
            
            .legend-text {{
                flex-grow: 1;
            }}
            
            .legend-name {{
                font-weight: 600;
                color: #2c3e50;
            }}
            
            .legend-stats {{
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            
            .content {{
                background: white;
                padding: 30px;
                padding-right: 360px; /* Espace pour la légende flottante */
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                font-size: 1.1em;
                line-height: 1.8;
            }}
            
            .chunk {{
                padding: 4px 8px;
                margin: 2px;
                border-radius: 6px;
                transition: all 0.3s ease;
                cursor: help;
                display: inline;
            }}
            
            .confidence-high {{
                opacity: 1;
                font-weight: 500;
            }}
            
            .confidence-medium {{
                opacity: 0.8;
            }}
            
            .confidence-low {{
                opacity: 0.6;
                font-style: italic;
            }}
            
            {theme_styles}
            
            footer {{
                text-align: center;
                margin-top: 30px;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                
                h1 {{
                    font-size: 2em;
                }}
                
                .legend {{
                    position: relative;
                    width: 100%;
                    max-height: none;
                    margin-bottom: 20px;
                    right: 0;
                    top: 0;
                }}
                
                .content {{
                    padding-right: 30px;
                }}
            }}
            
            @media (max-width: 1400px) {{
                .legend {{
                    width: 280px;
                }}
                
                .content {{
                    padding-right: 320px;
                }}
            }}
            
            /* Animation d'entrée pour la légende */
            @keyframes slideInRight {{
                from {{
                    transform: translateX(100%);
                    opacity: 0;
                }}
                to {{
                    transform: translateX(0);
                    opacity: 1;
                }}
            }}
            
            .legend {{
                animation: slideInRight 0.6s ease-out;
            }}
        </style>"""

    def _generate_legend(self, colors: Dict[str, str], insights: Dict[str, any]) -> str:
        """Génère la légende avec les statistiques"""
        legend_items = ""

        # Trier par pourcentage décroissant
        sorted_themes = sorted(
            insights.items(), key=lambda x: x[1]["percentage"], reverse=True
        )

        for theme, stats in sorted_themes:
            color = colors[theme]
            legend_items += f"""
            <div class="legend-item theme-{self._sanitize_theme_name(theme)}">
                <div class="legend-color" style="background-color: {color};"></div>
                <div class="legend-text">
                    <div class="legend-name">{theme}</div>
                    <div class="legend-stats">
                        {stats["percentage"]:.1f}% du texte • 
                        {stats["chunk_count"]} segments • 
                        Confiance: {stats["avg_confidence"]:.2f}
                    </div>
                </div>
            </div>"""

        return f"""
        <div class="legend">
            <h2>📊 Répartition thématique</h2>
            <div class="legend-grid">
                {legend_items}
            </div>
        </div>"""

    def _sanitize_theme_name(self, theme: str) -> str:
        """Nettoie le nom du thème pour l'utiliser en CSS"""
        return re.sub(r"[^a-zA-Z0-9]", "-", theme.lower())

    def _get_confidence_class(self, confidence: float) -> str:
        """Retourne la classe CSS selon le niveau de confiance"""
        if confidence >= 0.7:
            return "confidence-high"
        elif confidence >= 0.4:
            return "confidence-medium"
        else:
            return "confidence-low"

    def _escape_html(self, text: str) -> str:
        """Échappe les caractères HTML"""
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
        description="Analyse thématique de texte avec génération HTML"
    )
    parser.add_argument("input_file", help="Fichier texte à analyser")
    parser.add_argument("-o", "--output", help="Fichier HTML de sortie (optionnel)")
    parser.add_argument(
        "--max-themes", type=int, default=5, help="Nombre maximum de thèmes (défaut: 5)"
    )

    args = parser.parse_args()

    # Vérifier que le fichier existe
    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier '{args.input_file}' n'existe pas.")
        return

    # Lire le fichier
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return

    if len(text.strip()) == 0:
        print("Erreur: Le fichier est vide.")
        return

    print(f"Analyse du fichier: {args.input_file}")
    print(f"Taille du texte: {len(text)} caractères")

    # Initialiser l'analyseur
    try:
        analyzer = ThematicAnalyzer()
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        return

    # Traiter le texte
    try:
        chunks = analyzer.process_text(text)
        print(f"✅ Analyse terminée: {len(chunks)} chunks analysés")

        # Générer le HTML
        generator = HTMLGenerator()
        title = f"Analyse de {os.path.basename(args.input_file)}"
        html = generator.generate_html(chunks, title)

        # Déterminer le nom du fichier de sortie
        if args.output:
            output_file = args.output
        else:
            base_name = os.path.splitext(args.input_file)[0]
            output_file = f"{base_name}_analyse.html"

        # Sauvegarder le HTML
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ Fichier HTML généré: {output_file}")
        print(
            f"🎨 Thèmes identifiés: {', '.join(set(chunk.theme for chunk in chunks))}"
        )

    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
