# Notes du mentor

> Tenu par l'agent IA (voir `AGENTS.md` §3). Aucune solution ici : seulement ce qui est observé.

## État actuel

- **Module en cours** : `losses` terminé, CI verte sur 3.10 → 3.14 (2026-11-12). Prochain : fermer
  le point 5, puis `metrics`.
- **Point 5 de `losses` fermé (2026-11-12)** : test unitaire du gradient de `divide_no_nan` sur les
  deux entrées, vérifié comme capable d'échouer sur l'ancienne version.
- **`metrics` premier jet (2026-11-12)**, en NumPy, écrit séparément des losses. Constats remontés :
  lint, format et mypy en échec ; `np.ndarray` nu (même erreur que celle déjà vue en 3.10) ; conventions
  de division par 0 qui diffèrent d'une métrique à l'autre ; axe du MASE sur une entrée batchée ;
  le test du MAPE avec y = 0 ne vérifie pas la valeur.
- **Questions en attente (metrics)** : (a) ce que le dénominateur du MASE doit couvrir selon la
  section 2 ; (b) pourquoi deux conventions différentes pour la division par 0.
- **`metrics` deuxième jet (2026-11-12)** : réécriture rapide, au style très différent du premier jet
  (helper, docstrings, conversions de dtype). Alias de type dans un module dédié : bon choix. ND et
  NRMSE gardés par préférence personnelle, la normalisation de NRMSE est documentée.
  Constat : le MASE annonce gérer un batch, mais sur un batch de 2 séries il donne 0.109 alors que
  la loss donne 0.55 (vérifié). Niveau d'indice : 1. Lui demander de l'expliquer avec ses mots :
  c'est aussi le test pour savoir s'il maîtrise ce jet.
- **2026-11-17** : MASE par série corrigé (0.55 vérifié), test à deux séries ajouté. Il croyait
  d'abord que `np.mean` et `torch.mean` se comportaient différemment : idée fausse corrigée (même
  règle, seul l'argument d'axe manquait). Il a demandé lui-même quelle version était fausse : niveau 2
  atteint (localisation + tableau chiffré). Choix fait : une série d'historique constant rend tout le
  batch `inf`. nox + pre-commit ajoutés par lui, la CI appelle nox (source unique des vérifications).
  Point restant : la version de ruff est fixée dans pre-commit mais pas dans `pyproject.toml`.
- **Consigne de l'auteur (2026-10-31)** : pas de questions banales (outillage, détails secondaires).
  Creuser quand la question apporte une meilleure compréhension de la discipline (ML/DL).
- **Consigne (2026-10-31)** : une seule liste de points ouverts, ci-dessous. Au maximum une question
  nouvelle par réponse. Refermer ou mettre de côté un point avant d'en ouvrir un autre.
- **Points ouverts, module `losses`** (état au 2026-11-06, vérifié en exécutant le code) :
  1. ✅ MASE : historique en argument, normalisation par série, axe du temps. La valeur est correcte.
  2. ✅ `divide_no_nan` réécrit en torch, typé : gradients finis en 0/0 sur les trois losses.
  3. ✅ sMAPE : stop-gradient en place (gradient mesuré conforme à la version avec stop-gradient).
  4. ✅ MAPE : plus d'inf quand y = 0, quel que soit le signe de l'erreur.
  5. ⚠️ Un seul test de gradient (sMAPE en 0/0). `test_ops` ne vérifie toujours pas le gradient de
     `divide_no_nan`, qui est pourtant l'endroit où le bug vivait.
- **Clarifié (2026-10-31)** : il a recopié `ops.py` du code officiel, et il l'assume. Point clos. Les
  commentaires de `test_losses.py` ont été traduits par l'agent, en « mode solution autorisé ».
- **Mis de côté (à rouvrir au module concerné, pas avant)** :
  - `dataset` : `__len__` négatif si la série est trop courte ; padding et masques au passage sur M4.
  - Tests de `block`, `model` et `train` (propriétés à vérifier avec des poids aléatoires,
    surapprentissage d'une seule série).

---

## Historique

### 2026-11-12 — CI en place et verte

**Acquis confirmés**
- CI écrite par lui : torch en CPU, matrice 3.10 → 3.14, ruff + format + mypy + pytest.
- Il a corrigé seul l'erreur mypy révélée par le job 3.10 (stubs numpy plus anciens), après un
  indice de niveau 2. Il a noté dans son journal pourquoi un type plus large était refusé par
  mypy : raisonnement de typage juste et écrit avec ses mots.

**Fragilités**
- Bornes de versions de `pyproject.toml` : expliquées, à revérifier qu'il les a ajustées.
- Messages de commit moins réguliers (« CI workflow », espace avant les deux-points).

### 2026-11-06 — Correctif du gradient, début de la CI

**Acquis confirmés**
- Il a compris que backward est dérivé du forward par autograd, et qu'il faut empêcher la valeur
  interdite d'entrer dans l'opération, pas la remplacer après. Correctif écrit seul après un indice
  de niveau 2, qui règle d'un coup les points 2 et 4.
- Stop-gradient ajouté au bon endroit (dans la loss, pas dans l'utilitaire).
- Il est sorti du code officiel quand celui-ci ne convenait pas à ses données : bon réflexe de RE.

**Fragilités**
- Toujours peu de tests de gradient (point 5). À rouvrir avant `block`, où ils compteront plus.
- Il n'a pas encore lancé `ruff format` ni mypy avant de dire « c'est fait ».

**Niveaux d'indice atteints**
- NaN dans le gradient de `divide_no_nan` : niveau 2.

### 2026-10-31 — Losses : théorie du sMAPE, premier jet, revue

**Acquis confirmés**
- Lien entre l'échelle de la loss et le learning rate sous SGD (calcul juste : lr × 200), et
  normalisation par Adam. Il a d'abord sous-estimé la conséquence (« moins efficace » au lieu de
  divergence).
- Il a lu la section 5.2 seul : rôle du stop-gradient (le dénominateur au carré disparaît). Il a
  compris que l'explosion près de 0 est atténuée, pas supprimée, une fois que c'était précisé.
- Biais de surestimation du sMAPE trouvé à partir de la formule. Il a fait de lui-même le lien avec
  les coûts asymétriques, et a noté la pinball loss comme hypothèse à tester après la baseline.
- Bonne démarche de reproduction : il est allé lire le code officiel pour le cas 0/0, et il a
  documenté ses choix dans son journal.
- MASE : il a corrigé seul la signature et l'axe, après des questions de niveau 1.

**Fragilités**
- Il teste les valeurs mais pas les gradients, alors que c'était signalé deux fois. Les bugs de
  gradient passent donc inaperçus.
- Il reprend du code de référence sans le réécrire : voir le point « À clarifier » ci-dessus.

**Niveaux d'indice atteints**
- Gradient NaN derrière un masque : niveau 2 (`torch.where` et la branche non choisie, NaN × 0).
- Stop-gradient : niveau 2 (section 5.2).

### 2026-10-30 — Packaging, git, premiers commits

**Acquis confirmés**
- L'intuition sur l'erreur d'import était juste (le chemin de recherche des modules). Il préfère
  installer le package plutôt que bricoler le chemin, mais sa justification reste à approfondir.
- Il a trouvé seul le fichier de configuration du projet, après un indice de niveau 2 (installation
  éditable). La question en attente sur le fichier de dépendances est donc résolue.
- Différence entre dépendances déclarées et environnement figé (`pip freeze`) : expliquée. Il a
  séparé les dépendances d'exécution et de développement.
- Git : `.gitignore` correct, PDF du papier exclu, `block.py` non fonctionnel sorti du commit.
  Deux commits séparés (outillage, puis `dataset`), avec des messages au format Conventional Commits.
  Bonne application des retours.

**Fragilités**
- Ses premières réponses sur le « src layout » et `sys.path` étaient vagues (« une liste de lieux »,
  « plus structuré »). Il n'a pas reformulé le « pourquoi » lui-même.
- Sa première version faisait passer les tests via `pythonpath` sans installer le package. Il ne
  s'en est pas rendu compte avant que l'agent le signale.
- Le `pyproject.toml` contient une config ruff/mypy assez avancée : vérifier qu'il la comprend.
- Détail : AGENTS.md, LICENSE et docs sont dans le commit `feat(data)` plutôt que dans `chore`.

**Niveaux d'indice atteints**
- Import du package : niveau 2.

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
