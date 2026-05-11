# 🧠 Self-Architecting RL Agent (Architecte de Soi)

Bienvenue dans le projet **Self-Architecting RL Agent**. Ce projet vise à construire un agent d'Apprentissage par Renforcement (RL) modulaire capable de modifier *sa propre architecture de réseau de neurones* en temps réel pendant son entraînement.

Plutôt que d'avoir une architecture rigide fixée avant l'apprentissage, cet agent possède un "Architecte" (ou Configurateur) basé sur le Meta-RL. L'Architecte observe les performances et les gradients, et décide d'ajouter de nouvelles couches (Depth) ou d'élargir des couches existantes (Widen) pour augmenter la capacité de l'agent si nécessaire, tout en préservant ses connaissances existantes via un transfert de poids intelligent.

## 🚀 Fonctionnalités Clés

- **Moteur de Graphe Dynamique** : L'architecture de l'agent est définie par un DAG (Directed Acyclic Graph) généré et compilé à la volée.
- **Encodeur JEPA** : Utilise une architecture Vision Transformer (ViT) inspirée de I-JEPA pour la perception.
- **World Model Récurrent** : Modélise la dynamique de l'environnement latent via un RNN/GRU, à l'image de Dreamer.
- **PPO Inner-Loop** : L'apprentissage des actions est assuré par un algorithme PPO complet avec GAE, permettant une adaptation très stable.
- **Mutations Architecturales à Chaud** : L'agent peut s'allonger (Depth), s'élargir (Width) ou ajouter des **Skip Connections** en cours d'épisode sans détruire ses poids.
- **Pruning Structuré** : L'architecte peut supprimer des canaux de sortie avec un recalcul automatique des connexions suivantes.
- **Scheduler Prune/Grow** : Le contrôleur méta alterne des phases de prune et de croissance selon le cycle méta.
- **Meta-RL Multi-Tâches** : Entraînement sur une rotation d'environnements MiniGrid pour forcer une généralisation structurelle.
- **Support Distribué (Ray)** : Parallélisation de la collecte d'expériences via Ray pour accélérer massivement l'entraînement.
- **Observabilité WandB** : Suivi en temps réel de l'évolution de la topologie du DAG et des métriques de performance.

---

## 🛠️ Structure du Projet

- `modules/` : Les briques fondamentales de l'agent (Perception, World Model, Policy, Value) et le moteur de graphe (`graph.py`).
- `configurator/` : L'intelligence de l'architecte. Contient la Meta-Policy et le `graph_modifier.py` qui applique les mutations et effectue les transferts de tenseurs.
- `envs/` : Les wrappers d'environnement (MiniGrid par défaut) pour s'assurer que les observations sont compatibles PyTorch.
- `train_inner.py` : Entraînement PPO de l'agent sur une architecture fixe.
- `train_outer.py` : La boucle de Meta-RL globale qui invoque l'Architecte.

---

## 💻 Installation

1. **Cloner le projet**
2. **Créer un environnement virtuel (optionnel mais recommandé)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou `venv\Scripts\activate` sur Windows
   ```
3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optionnel) Cloner les dépendances externes dans `external/`**
   Ce projet s'inspire ou peut requérir certains dépôts officiels (MiniGrid, simple-ijepa). Assurez-vous d'avoir les environnements configurés si nécessaire.

5. **Lancer l'entraînement**
   ```bash
   python train_outer.py
   ```

---

## 🤝 Comment Contribuer ?

Toute contribution est la bienvenue ! Voici comment vous pouvez aider :

1. **Nouveaux Nœuds de Graphe** : Dans `modules/graph.py`, vous pouvez enregistrer de nouveaux types de couches PyTorch (e.g., `LayerNorm`, `AttentionBlock`).
2. **Nouvelles Mutations** : Modifiez `configurator/graph_modifier.py` pour ajouter des mutations complexes (ex: ajout de *Skip Connections*, élagage/Pruning).
3. **Méta-Objectifs (Losses)** : Améliorez la fonction de récompense du configurateur dans `train_outer.py` pour mieux pénaliser la latence ou la consommation de mémoire (FLOPs).

Pour proposer une modification, merci d'ouvrir une *Issue* pour en discuter, puis de soumettre une *Pull Request*.

---

## 🔮 Le Futur du Projet (Roadmap)

Le projet MVP est désormais robuste. Les prochaines étapes visent à passer à l'échelle supérieure :

- [x] **Élagage Actif (Pruning)** : Suppression intelligente de neurones basée sur l'importance des canaux (L1-Norm).
- [x] **Scheduler Prune/Grow** : Alternance cyclique des phases de mutation pour stabiliser l'apprentissage.
- [x] **Mutations Topologiques (Skip Connections)** : Ajout dynamique de chemins résiduels avec couches de projection automatiques.
- [x] **PPO Distribué (Ray)** : Passage à l'échelle via Ray pour une collecte d'expérience multi-workers.
- [x] **Observabilité WandB** : Dashboarding complet des métriques de graphe (profondeur, largeur, paramètres).
- [ ] **Méta-Généralisation avec XLand-MiniGrid** : Entraîner l'agent sur des milliers de tâches procédurales pour forcer l'émergence de structures méta-plastiques.
- [ ] **Accélération JAX** : Portage de l'Inner Loop sur JAX/XLand pour multiplier par 100 la vitesse de collecte.
- [ ] **Méta-Objectifs Hardware-Aware** : Intégration d'une pénalité directe sur la latence (ms) et les FLOPs dans la récompense de l'architecte.
- [ ] **Pruning par Sensibilité (Taylor Expansion)** : Utilisation des gradients pour un élagage plus précis que la simple norme des poids.

N'hésitez pas à forker le projet et à expérimenter vos propres idées de neuro-évolution !
