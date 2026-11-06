# Journal d'apprentissage N-BEATS

## Où j'en suis
- Module en cours :
- Prochaine étape :
- Questions ouvertes :

---

## 27/10/2026 

### Ce que j'ai compris
- Etapes d'un NN
    * Boucle epoch
        1. batch
        2. zero_grad() _ mettre les grad à zero car pytorch les sommes de base_
        3. forward
        4. losse
        5. backward() _ permet le calcul des gradient _
        6. step() _ descente de gradient pour metre à jour les params _

- Pour qu'une metrique puisse être utilisé comme losse, elle doit être derivable presque partout (par necessairement delivrable partout)
    Exemple avec |x| qui n'est pas derivable en zero mais le grad doit retourner quelque chose et ici il retourne 0.
    Derivé à gauche -1, à droite 1, pytorch prend le milieu 0 et poru les loss ça ne pose pas de soucis d'avoir des |x| car si la prediction est egale au reel bah avoir un gard à 0 est logique
- L'odre d'implementation des modules se fera par qui depend de qui
    * dataset
    * metrics
    * losses
    * block
    * model
    * train

---

## 31/10/2026 

### idée à tester
 Si au lieu du smape on utilise plutôt le pinball loss pour pouvoir definir le penalisation de la sous etimation par a la surestimation
 Par exemple pour de la gestion du stock on p eut preferer avoir du surplus de stock qu'un manque ou inversement selon le besoin

### Arbritage
cas 0/0 → 0, comme divide_no_nan dans le code officiel
padding + masques à prévoir au passage sur M4