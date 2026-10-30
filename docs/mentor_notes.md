# Notes du mentor

> Tenu par l'agent IA (voir `AGENTS.md` §3). Aucune solution ici : seulement ce qui est observé.

## État actuel

- **Module en cours** : mise en place du nouveau projet (repart de zéro, sans le code généré par IA).
- **Prochaine étape** :
  1. Faire passer `pytest` (erreur d'import attendue : package `nbeats` introuvable). Il doit trouver
     lui-même pourquoi, avec le « src layout » et l'installation éditable comme pistes.
  2. Faire `git init`, premier commit, commencer `docs/journal.md`. (Venv fait : Python 3.14 + torch.)
  3. Puis préparer `losses` (sMAPE).
- **Questions en attente de réponse** :
  - Test de `block` : quelles propriétés vérifier alors que les poids sont aléatoires ? (indice donné :
    checklist de revue d'AGENTS.md)
  - Test de `model` : comment vérifier exactement le résidu et la somme des ŷ malgré l'aléatoire ?
    (indice donné : récupérer les sorties intermédiaires et refaire le calcul)
  - Test de `train` : quel comportement de la loss attendre en surapprenant une seule petite série ?
    Sa réponse actuelle (« les poids se mettent à jour ») est trop faible.
  - sMAPE : 3 cas de test calculés à la main (cas normal, prédiction parfaite, y = ŷ = 0), plus un
    4ᵉ sur l'invariance d'échelle.
  - Masque pour 0/0 : comment le vectoriser sans boucle, et le piège du gradient dans la branche non
    choisie. Pas encore abordé de son côté.
  - Quel fichier liste les dépendances d'un projet, pour la reproductibilité ?

---

## Historique

### 2026-10-27 — Venv recréé (outillage, à sa demande)

- Son venv était en Python 3.15, version pour laquelle torch n'a pas encore de wheel. pip disait alors
  « from versions: none ». À sa demande, l'agent a recréé `.venv` en Python 3.14 et installé torch 2.14.1 (CPU).
- Il n'a pas diagnostiqué lui-même le problème de compatibilité. Question pour plus tard : comment
  lire un « No matching distribution » de pip, et comment l'éviter dans la config du projet ?
- NumPy n'est pas installé (torch affiche un avertissement). C'est à lui de décider quand l'ajouter.

### 2026-10-27 — Diagnostic initial et architecture du projet

**Profil** : vise un poste MLE / Research Engineer. A appris avec Keras, débute en PyTorch. Il a
reconnu de lui-même que l'ancien projet était surtout généré par IA, et il a décidé de repartir de zéro.
Bonne lucidité.

**Code existant de sa main** : `dataset.py` (générateur synthétique + `TimeSeriesDataset` à fenêtre
glissante) avec des tests corrects (shape, dtype, causalité sur une fenêtre). `block.py` commencé,
non fonctionnel (erreurs signalées sans correction : appel à `super`, indentation de `forward`).

**Acquis confirmés**
- Double résidu : il explique bien que le block suivant reçoit le signal moins ce que le block a
  expliqué, et que la prédiction finale est la somme des ŷ.
- Différence entre loss (optimisée pendant l'entraînement) et métrique (évaluation).
- Une loss doit être dérivable. Avec mon aide, il a admis que |x| n'est pas dérivable en 0.
- La boucle d'entraînement, une fois guidée : batch → forward → loss → `backward()` → `step()` →
  `zero_grad()`, répétée sur les batchs puis sur les epochs.
- 0/0 donne NaN. Il a proposé epsilon, puis **de lui-même** un masque (erreur = 0 quand y = ŷ = 0),
  en argumentant qu'epsilon dégrade la mesure s'il y a beaucoup de vrais zéros. Bon réflexe critique.
- Venv : il cite le conflit entre projets. Il n'a pas encore cité la reproductibilité.
- Ordre de codage choisi : dataset → losses → metrics → block → model → train. Il identifie que
  dataset, losses, metrics et block ne dépendent d'aucun autre fichier du projet.

**Idées fausses repérées**
- « Association de plusieurs modèles » au lieu de « plusieurs blocks » : il dit que c'est un lapsus
  de dictée vocale. À revérifier quand l'ensembling du papier sera abordé.
- Pensait que le fichier de loss « définit comment les poids sont mis à jour ». **Corrigé.**
- Attribuait la backpropagation à l'optimiseur. **Corrigé** : il sépare maintenant `backward()` sur
  la loss et `step()` sur l'optimiseur.
- Avait oublié deux étapes de la boucle : le calcul des gradients et la remise à zéro. **Corrigé**
  après indices.

**Fragilités**
- PyTorch : ne connaît pas encore les noms des API. Il faut lui donner la correspondance avec Keras,
  avec des exemples sans rapport avec N-BEATS.
- Stratégie de test : bonne intuition pour les fonctions mathématiques, floue pour les modules à poids
  aléatoires et pour l'entraînement.
- Propagation des NaN : il a reçu l'explication, mais ne l'a pas reformulée lui-même.

**Niveaux d'indice atteints**
- Boucle d'entraînement : niveau 2 (pistes vers la doc `torch.optim` et autograd).
- Parade au 0/0 : niveau 1. Il a trouvé seul.

**À revérifier plus tard**
- Qu'il sait reformuler la boucle d'entraînement sans aide, au moment de coder `train`.
- Points du papier encore non lus par lui : grilles temporelles du backcast et du forecast, ensembling,
  et la question de savoir si le sMAPE du papier utilise vraiment un stop-gradient.
