# Corriger la valeur prédictive positive d'un modèle ECG sur un site externe avec peu d'étiquettes locales

Traduction française de [PPV.md](PPV.md).

## Résumé

**Contexte.** Un site qui adopte un modèle ECG reçoit une valeur prédictive positive (VPP) mesurée dans la population du développeur et l'ajuste à sa propre prévalence par le théorème de Bayes. Cet ajustement à la prévalence suppose que la sensibilité et la spécificité d'un seuil fixe sont transportables d'une population à l'autre. Une mesure antérieure a montré qu'un déplacement du point de fonctionnement violait cette hypothèse entre hôpitaux et lieux de soins. Un site peut à la place mettre à jour le modèle sur des étiquettes locales, mais le nombre d'étiquettes qu'une telle mise à jour demande n'a pas été mesuré.

**Méthodes.** Cinq méthodes de mise à jour, qui donnent la VPP sur un site externe, ont été comparées dans 105 transferts, chacun un modèle calibré dans une population et appliqué à une autre, à Columbia, Beth Israel Deaconess et Chongqing. Deux méthodes n'utilisaient aucune étiquette ; trois utilisaient des budgets de 25 à 400 ECG, tirés par patient, avec 200 tirages par budget. Le critère de jugement principal était l'erreur absolue entre la VPP annoncée et la VPP observée chez les patients cibles non étiquetés. La proportion de tirages dans lesquels un seuil réajusté sur les étiquettes signalait tous les patients a été ajoutée comme critère co-principal après la lecture des résultats.

**Résultats.** Sans étiquette, la correction de prévalence se trompait d'une médiane de 13,1, 11,4 et 51,7 points de pourcentage sur les trois sites. Avec 100 étiquettes, qui contenaient en moyenne 9,9, 13,2 et 14,8 cas, la recalibration logistique se trompait de 3,1, 5,7 et 3,8 points. Le seuil réajusté signalait tous les patients dans 79 %, 84 % et 71 % des tirages où il était placé, et 99 %, 99 % et 98 % des cas. Son erreur était de 2,6, 2,2 et 3,5 points sur ces tirages, et de 5,1, 4,3 et 5,5 points sur les tirages où son seuil était fini. Un seuil certifié par des tests binomiaux exacts pour tenir la VPP source a été trouvé dans 16,8 %, 25,1 % et 1,0 % des tirages ; parmi ces tirages, sa VPP observée tombait sous la VPP source dans 13,2 %, 5,7 % et 100 %.

**Conclusions.** Avec 100 étiquettes locales, la recalibration donnait la VPP à quelques points près sur trois sites externes. Un seuil réajusté sur 100 étiquettes signalait le plus souvent tous les patients, et sa faible erreur à ce budget ne traduisait pas un seuil qui sépare les cas des non-cas. Ces résultats suggèrent qu'un seuil réajusté demande plus d'étiquettes que la recalibration, assez pour placer neuf cas dans la moitié qui sert à l'ajuster.

## 1. Introduction

Un site qui adopte un modèle ECG reçoit une VPP mesurée dans la population du développeur. La pratique courante est l'ajustement à la prévalence : la VPP est recalculée à la prévalence locale par le théorème de Bayes. L'ajustement suppose que la sensibilité et la spécificité du seuil du modèle sont transportables, de sorte que seule la prévalence diffère entre les deux populations. Un changement de sévérité ou de présentation des cas, ou des non-cas, viole cette hypothèse.

Dans une mesure rétrospective [1], la VPP ajustée ne prédisait pas de façon fiable la VPP observée dans d'autres hôpitaux et lieux de soins, parce que le point de fonctionnement d'un seuil fixe se déplaçait d'une population à l'autre. Un site peut à la place mettre à jour le modèle sur des étiquettes locales, des ECG dont le site a confirmé le diagnostic : il peut recalibrer les probabilités du modèle, réajuster son seuil, ou certifier un seuil par un test statistique. Une VPP constante d'un site à l'autre demande un seuil qui suit le site, et un seuil plus haut échange de la sensibilité contre de la VPP. Le nombre d'étiquettes qu'une telle mise à jour demande, et la sensibilité qu'elle conserve, n'ont pas été mesurés.

Nous avons comparé cinq méthodes de mise à jour par l'accord entre la VPP que chacune annonce et la VPP observée sur le site externe, en fonction du nombre d'étiquettes locales.

## 2. Méthodes

### 2.1 Sites et transferts

L'unité d'analyse est le transfert, d'une population source vers une population cible. Trois sites externes forment l'analyse principale. À Columbia, les quatre modèles EchoNext, calibrés sur des patients hospitalisés, ont été appliqués aux ECG des urgences et des consultations (76 transferts). À Beth Israel Deaconess, les modèles de la rotation à sept sources ont été appliqués à MIMIC-IV-ECG (28 transferts), dont les étiquettes sont les comptes rendus de l'appareil ECG, si bien que la VPP observée y mesure l'accord avec l'appareil et non avec un examen de référence clinique. À Chongqing, le modèle d'infarctus du myocarde de PTB-XL a été appliqué aux ECG de l'hôpital (1 transfert). Shandong, les autres corpus de la rotation, CODE-15%, le jeu de test CODE et ZZU forment une réserve secondaire, rapportée dans [`results/ppv_repairs_reserve.json`](results/ppv_repairs_reserve.json). Les transferts où un modèle est évalué dans sa propre population source servent de témoins dans la distribution.

### 2.2 VPP cible et méthodes de mise à jour

Pour chaque transfert, la VPP cible v est la VPP atteinte dans la population source au seuil qui y donne 90 % de sensibilité. Elle a été fixée à partir des seuls scores source, avant toute méthode, et le fichier de résultats enregistre l'empreinte de la table cible. Les cinq méthodes de mise à jour étaient :

1. la correction de prévalence sans étiquette, dans laquelle la prévalence cible était estimée à partir des scores du modèle par espérance-maximisation, une estimation itérative du maximum de vraisemblance des proportions de classes [2], et la VPP au seuil source était recalculée par le théorème de Bayes ;
2. la recalibration logistique sur n étiquettes, qui réajuste l'ordonnée à l'origine et la pente du logit du modèle sur les étiquettes, ou l'ordonnée à l'origine seule quand la classe la plus rare compte moins de dix événements ; la VPP annoncée est la probabilité recalibrée moyenne parmi les patients signalés ;
3. le réajustement du seuil sur des étiquettes partagées, dans lequel le seuil était fixé sur une moitié des n étiquettes pour donner 90 % de sensibilité, et la VPP annoncée était la VPP observée à ce seuil dans l'autre moitié, avec sa borne unilatérale de Clopper-Pearson, une borne de confiance binomiale exacte ;
4. la certification du seuil par Learn then Test [3], dans laquelle les seuils d'une grille fixée par les scores source étaient testés en séquence fixe du plus haut au plus bas, chacun par un test binomial exact de VPP ≤ v à δ = 0,1, et le plus bas seuil rejeté avant le premier non-rejet était retenu ; la VPP annoncée est v. La garantie est marginale : sur des échantillons étiquetés répétés, la probabilité qu'un seuil soit retenu dont la VPP est inférieure ou égale à v est au plus δ. Elle ne borne pas cette probabilité parmi les échantillons dans lesquels un seuil est retenu ;
5. la sélection conforme pondérée sans étiquette ([4], algorithme 1 avec élagage déterministe), qui signale des patients de sorte que le taux de fausses découvertes, la proportion attendue de faux positifs parmi les patients signalés, reste sous q = 1 − v ; des poids de décalage de covariables, le rapport estimé des densités cible et source du score du modèle, ajustent les patients source à la population cible.

Les méthodes 3 et 4 placent les seuils par la règle conforme utilisée à la source : parmi m cas, le seuil est le score du cas de rang ⌈(m + 1) × 0,9⌉ en partant du plus haut. Quand une moitié d'échantillon contient moins de neuf cas, ce rang dépasse m, le seuil est infini et tous les patients sont signalés. Un seuil qui signale tous les patients annonce et observe la prévalence, si bien que son erreur est faible qu'il sépare ou non les cas des non-cas.

### 2.3 Échantillonnage et critères de jugement

Les étiquettes ont été tirées par patient aux budgets n = 25, 50, 100, 200 et 400, avec 200 tirages par budget. Les patients cibles non tirés formaient l'ensemble d'évaluation ; les méthodes 1 et 5 utilisaient toute la cible. Le protocole nommait comme critère principal l'erreur absolue, en points de pourcentage, entre la VPP annoncée et la VPP observée dans l'ensemble d'évaluation, et comme critères secondaires la sensibilité et le bénéfice net. La sensibilité dans la cible était moyennée sur les tirages où une méthode renvoyait un seuil, puis sur les transferts d'un site. Le bénéfice net est le nombre de vrais positifs par patient moins le nombre de faux positifs par patient pondéré par la cote de la probabilité seuil, ici 10 %. Pour la méthode 4, le taux de violation est la proportion de tous les tirages dans lesquels un seuil a été certifié et sa VPP observée est tombée sous v ; pour la méthode 5, la proportion de fausses découvertes est la part réalisée de faux positifs parmi les patients signalés. La discrimination (AUROC) et le bénéfice net sont rapportés par transfert dans [`results/ppv_repairs.json`](results/ppv_repairs.json).

### 2.4 Analyses post hoc

Quatre mesures ont été ajoutées après que le protocole a été fixé : le nombre moyen de cas dans chaque échantillon étiqueté, la proportion de tirages dans lesquels le seuil réajusté signalait tous les patients, l'erreur et la sensibilité du seuil réajusté sur les tirages où il était fini, et deux références de bénéfice net par transfert, signaler tous les patients et le propre seuil de la cible pour 90 % de sensibilité, ajusté sur toutes ses étiquettes. La seconde référence est un point de fonctionnement, non le bénéfice net le plus haut atteignable. La proportion de tirages signalant tous les patients a été ajoutée comme critère co-principal. Les intervalles des erreurs médianes regroupées sont les centiles 2,5 et 97,5 de 200 rééchantillonnages bootstrap, qui tirent avec remise les transferts d'un site puis les tirages de chaque transfert ; l'intervalle de la médiane sans étiquette rééchantillonne aussi les patients, comme pour les critères de réfutation. Avec 200 rééchantillonnages, chaque borne est approchée à environ une décimale. Les références sont lues sur toute la cible, les méthodes sur les patients d'évaluation de chaque tirage, et le bénéfice net du seuil réajusté sur les tirages où il était placé.

### 2.5 Critères de réfutation

Trois critères ont été fixés avec le protocole. La correction sans étiquette serait jugée suffisante hors de Columbia si la borne supérieure de l'intervalle bootstrap de son erreur médiane, obtenu en rééchantillonnant avec remise les transferts et les patients, restait sous deux points à Beth Israel et à Chongqing. La certification serait jugée invalide sous un échantillonnage réaliste si son taux de violation dépassait 15 % à n = 100 sur un site. Le budget serait jugé insuffisant pour certifier v si un seuil était renvoyé dans moins de la moitié des tirages.

## 3. Résultats

### 3.1 Sans étiquette, la correction de prévalence se trompait de plus de dix points

La correction de prévalence se trompait d'une médiane de 13,1 points à Columbia, 11,4 points à Beth Israel et 51,7 points à Chongqing. Elle sous-estimait la VPP observée dans chaque transfert de Columbia et la surestimait à Chongqing. Au seuil source, que cette correction conserve, la sensibilité dans la cible était de 82 % à Columbia, 75 % à Beth Israel et 84 % à Chongqing, contre 90 % dans la source. L'estimation sans étiquette publiée pour Columbia dans [1], sur la sélection de transferts de cette étude, se trompait de 12,6 points. Chez les témoins dans la distribution, la correction se trompait de 2,4 points, et avec la vraie prévalence à la place de l'estimation, le théorème de Bayes se trompait de 0,5 points, contre 0,2 points sur les témoins EchoNext de [1]. Sur les huit sites de réserve, elle se trompait de 12,2 à 93,3 points.

### 3.2 Avec 100 étiquettes, le seuil réajusté signalait le plus souvent tous les patients

Avec 25 étiquettes, la recalibration se trompait de 5,8, 8,5 et 9,3 points à Columbia, Beth Israel et Chongqing, et le réajustement du seuil de 6,0, 5,7 et 6,6 points. Avec 100 étiquettes, les deux erreurs étaient de 3,1 et 2,6 points à Columbia, 5,7 et 2,2 points à Beth Israel, et 3,8 et 3,5 points à Chongqing ; avec 400 étiquettes, elles étaient de 1,9 et 1,6 points, 3,5 et 1,7 points, et 1,8 et 2,6 points. À partir de 100 étiquettes, la VPP annoncée dépassait la VPP observée dans 44 % à 55 % des tirages ; avec moins d'étiquettes, la recalibration la surestimait dans jusqu'à 72 % des tirages.

La recalibration conserve le seuil source, et sa sensibilité était égale à celle de la correction de prévalence à chaque budget. Les seuils réajustés signalaient 99 %, 99 % et 98 % des cas avec 100 étiquettes, et 94 %, 95 % et 92 % avec 400, pour une cible de 90 %. Le tableau 1 donne le seuil réajusté par budget. Avec 100 étiquettes, il signalait tous les patients dans 79 %, 84 % et 71 % des tirages où il était placé. Sur les tirages où son seuil était fini, son erreur était de 5,1, 4,3 et 5,5 points, au-dessus de celle de la recalibration à Columbia et à Chongqing, et sa sensibilité de 92 %, 94 % et 93 %. La part de tirages signalant tous les patients passait sous la moitié à partir de 200 étiquettes à Chongqing et de 400 étiquettes à Columbia et Beth Israel. Chez les témoins dans la distribution, le seuil réajusté signalait tous les patients dans 75 % des tirages avec 100 étiquettes. À la prévalence médiane des transferts, une moitié étiquetée devrait contenir neuf cas à partir de 272 étiquettes à Columbia, 314 à Beth Israel et 121 à Chongqing.

Tableau 1. Seuil réajusté par budget d'étiquettes sur les trois sites et chez les témoins dans la distribution. Les erreurs sont les erreurs absolues médianes de la VPP annoncée, en points de pourcentage. L'erreur sur les seuils finis regroupe les tirages où le seuil était fini, ce qui donne plus de poids aux transferts de prévalence plus haute ; son intervalle à 95 % vient du bootstrap de la section 2.4. La sensibilité sur les seuils finis est moyennée sur les transferts qui en avaient un, chaque transfert comptant une fois. Les nombres moyens de cas dépassent ceux attendus à la prévalence médiane parce que quelques transferts de forte prévalence les relèvent. Avec 25 étiquettes, la valeur de Columbia repose sur 18 tirages et aucun seuil de Chongqing n'était fini ; avec 50 étiquettes, la valeur de Chongqing repose sur 3 tirages.

| Site | Étiquettes | Cas en moyenne | Tirages avec un seuil réajusté | Dont signalant tous les patients | Erreur, tirages avec un seuil | Erreur, seuil fini | Sensibilité, seuil fini |
|---|---|---|---|---|---|---|---|
| Columbia | 25 | 2,5 | 59 % | 100 % | 6,0 | 17,8 (9,8 à 21,0) | 92 % |
| Columbia | 50 | 5,0 | 75 % | 93 % | 3,4 | 7,4 (6,7 à 8,5) | 91 % |
| Columbia | 100 | 9,9 | 89 % | 79 % | 2,6 | 5,1 (4,8 à 5,4) | 92 % |
| Columbia | 200 | 19,9 | 97 % | 60 % | 1,8 | 3,5 (3,3 à 3,7) | 91 % |
| Columbia | 400 | 39,8 | 100 % | 32 % | 1,6 | 2,3 (2,1 à 2,5) | 91 % |
| Beth Israel | 25 | 3,3 | 53 % | 87 % | 5,7 | 10,3 (8,7 à 11,5) | 92 % |
| Beth Israel | 50 | 6,6 | 71 % | 80 % | 2,7 | 6,7 (5,8 à 7,7) | 94 % |
| Beth Israel | 100 | 13,2 | 87 % | 84 % | 2,2 | 4,3 (3,7 à 5,1) | 94 % |
| Beth Israel | 200 | 26,4 | 96 % | 77 % | 1,8 | 3,4 (2,9 à 4,6) | 92 % |
| Beth Israel | 400 | 52,7 | 100 % | 35 % | 1,7 | 2,9 (2,3 à 3,6) | 92 % |
| Chongqing | 25 | 3,8 | 89 % | 100 % | 6,6 | – | – |
| Chongqing | 50 | 7,6 | 98 % | 98 % | 5,9 | 11,0 (8,3 à 12,4) | 94 % |
| Chongqing | 100 | 14,8 | 100 % | 71 % | 3,5 | 5,5 (4,2 à 6,9) | 93 % |
| Chongqing | 200 | 30,0 | 100 % | 3 % | 3,0 | 3,2 (2,6 à 3,6) | 93 % |
| Chongqing | 400 | 59,7 | 100 % | 0 % | 2,6 | 2,6 (2,2 à 2,9) | 92 % |
| Témoins dans la distribution | 25 | 3,1 | 59 % | 96 % | 6,3 | 6,5 (5,2 à 13,0) | 90 % |
| Témoins dans la distribution | 50 | 6,2 | 74 % | 88 % | 3,8 | 5,8 (4,5 à 7,2) | 91 % |
| Témoins dans la distribution | 100 | 12,4 | 87 % | 75 % | 2,4 | 5,0 (4,3 à 5,7) | 92 % |
| Témoins dans la distribution | 200 | 24,8 | 94 % | 54 % | 1,9 | 3,5 (3,2 à 3,8) | 91 % |
| Témoins dans la distribution | 400 | 49,6 | 98 % | 32 % | 1,6 | 2,5 (2,3 à 2,8) | 91 % |

### 3.3 La certification tenait sa garantie marginale mais renvoyait rarement un seuil

À n = 100, le taux de violation était de 2,2 % à Columbia, 1,4 % à Beth Israel et 1,0 % à Chongqing, tous sous δ, et un seuil était certifié dans 16,8 %, 25,1 % et 1,0 % des tirages. Parmi les tirages où un seuil était certifié, sa VPP observée tombait sous v dans 13,2 %, 5,7 % et 100 % ; à Columbia, cette proportion dépassait δ à chaque budget, de 19,2 % avec 25 étiquettes à 13,6 % avec 400. La proportion de Chongqing vient de deux tirages. Pour la plupart des cibles, aucun seuil de la grille n'atteint la VPP source. Là où un seuil était certifié, la sensibilité était de 43 % à Columbia, 87 % à Beth Israel et 38 % à Chongqing.

### 3.4 La sélection conforme pondérée dépassait son niveau nominal à Chongqing

La proportion moyenne de fausses découvertes était de 0,75 à Chongqing pour un niveau nominal de 0,39, de 0,64 pour 0,77 à Columbia et de 0,46 pour 0,60 à Beth Israel. La sensibilité était de 64 %, 68 % et 86 % sur les trois sites.

### 3.5 Sites de réserve et bénéfice net

Le tableau 2 donne les erreurs et le tableau 3 le bénéfice net à n = 100 pour chaque site, réserve secondaire et témoins dans la distribution compris. Avec 100 étiquettes, les échantillons étiquetés contenaient en moyenne moins de deux cas à Shandong, CODE-15% et ZZU, et moins de 16 sur chaque autre site. Le seuil réajusté signalait tous les patients dans 71 % à 100 % des tirages où il était placé, et dans tous à Shandong, CODE-15%, le jeu de test CODE et ZZU. Son bénéfice net restait à 0,006 près de celui de signaler tous les patients sur chaque site sauf PTB-XL, où il était inférieur de 0,009. À Columbia, son bénéfice net était de 0,001, contre −0,001 pour signaler tous les patients et 0,044 pour la recalibration. Le seuil source, que la correction de prévalence et la recalibration conservent, avait un bénéfice net plus haut que de signaler tous les patients sur chaque site.

Tableau 2. Erreur absolue médiane de la VPP annoncée, en points de pourcentage, avec intervalles à 95 %, à n = 100 étiquettes.

| Site | Transferts | Cas dans 100 étiquettes | Correction de prévalence, sans étiquette | Recalibration | Seuil réajusté | Tirages avec un seuil réajusté | Dont signalant tous les patients |
|---|---|---|---|---|---|---|---|
| Columbia | 76 | 9,9 | 13,1 (10,7 à 17,1) | 3,1 (2,9 à 3,5) | 2,6 (2,3 à 2,8) | 89 % | 79 % |
| Beth Israel | 28 | 13,2 | 11,4 (6,7 à 30,3) | 5,7 (4,3 à 7,8) | 2,2 (1,8 à 2,3) | 87 % | 84 % |
| Chongqing | 1 | 14,8 | 51,7 (50,4 à 52,8) | 3,8 (3,3 à 4,3) | 3,5 (3,1 à 4,3) | 100 % | 71 % |
| Shandong | 25 | 1,6 | 22,3 | 4,9 (2,7 à 7,4) | 1,2 (1,0 à 1,3) | 48 % | 100 % |
| Chapman-Ningbo | 28 | 5,0 | 17,3 | 5,1 (3,8 à 7,0) | 1,9 (1,8 à 2,2) | 75 % | 88 % |
| CPSC | 28 | 10,3 | 12,2 | 5,1 (3,9 à 6,6) | 2,9 (2,5 à 3,2) | 93 % | 80 % |
| Georgia | 28 | 6,9 | 20,0 | 5,6 (4,3 à 7,5) | 2,2 (2,1 à 2,5) | 91 % | 93 % |
| PTB-XL | 28 | 15,4 | 14,8 | 4,7 (3,1 à 6,8) | 2,6 (2,6 à 2,7) | 83 % | 82 % |
| CODE-15% | 24 | 1,7 | 93,3 | 3,3 (2,2 à 5,4) | 1,6 (1,5 à 1,6) | 58 % | 100 % |
| Jeu de test CODE | 21 | 3,7 | 54,0 | 5,1 (3,5 à 8,0) | 1,9 (1,7 à 2,0) | 85 % | 100 % |
| ZZU | 14 | 1,2 | 22,5 | 6,9 (2,5 à 13,0) | 1,3 (0,8 à 1,6) | 44 % | 100 % |
| Témoins dans la distribution | 77 | 12,4 | 2,4 | 3,0 (2,5 à 3,4) | 2,4 (2,2 à 2,8) | 87 % | 75 % |

Tableau 3. Bénéfice net à une probabilité seuil de 10 %, à n = 100 étiquettes. La recalibration conserve le seuil source et partage la colonne de la correction de prévalence ; le seuil certifié est moyenné sur les tirages où un seuil a été trouvé.

| Site | Signaler tous les patients | Propre seuil de la cible | Correction de prévalence et recalibration | Seuil réajusté | Seuil certifié | Sélection conforme pondérée |
|---|---|---|---|---|---|---|
| Columbia | −0,001 | 0,032 | 0,044 | 0,001 | 0,033 | 0,035 |
| Beth Israel | 0,036 | 0,101 | 0,090 | 0,031 | 0,118 | 0,073 |
| Chongqing | 0,055 | 0,081 | 0,085 | 0,060 | 0,050 | 0,084 |
| Shandong | −0,093 | 0,003 | 0,006 | −0,093 | 0,009 | −0,014 |
| Chapman-Ningbo | −0,055 | 0,013 | 0,026 | −0,052 | 0,027 | −0,001 |
| CPSC | 0,004 | 0,057 | 0,055 | 0,009 | 0,067 | 0,048 |
| Georgia | −0,034 | 0,037 | 0,037 | −0,032 | 0,034 | 0,005 |
| PTB-XL | 0,061 | 0,124 | 0,088 | 0,052 | 0,120 | 0,047 |
| CODE-15% | −0,092 | −0,077 | −0,041 | −0,092 | −0,036 | −0,032 |
| Jeu de test CODE | −0,070 | −0,009 | 0,000 | −0,070 | −0,009 | −0,011 |
| ZZU | −0,098 | −0,015 | −0,001 | −0,098 | 0,004 | −0,012 |
| Témoins dans la distribution | 0,027 | 0,070 | 0,069 | 0,027 | 0,051 | 0,049 |

### 3.6 Critères de réfutation

Les bornes supérieures de l'erreur médiane sans étiquette étaient de 30,3 points à Beth Israel et de 52,8 points à Chongqing ; avec un seul transfert, la borne de Chongqing ne reflète que le rééchantillonnage des patients. Le taux de violation de la certification ne dépassait 15 % sur aucun site. Un seuil était certifié dans moins de la moitié des tirages sur les trois sites.

## 4. Discussion

### 4.1 Principaux résultats

Sans étiquette, la correction de prévalence ne donnait la VPP sur aucun des trois sites, et le critère de réfutation indique que le besoin d'étiquettes locales n'est pas propre à Columbia. Avec 100 étiquettes, la recalibration la donnait à quelques points près en conservant le seuil source et sa sensibilité. Le seuil réajusté atteignait une erreur plus faible au même budget, mais dans la plupart des tirages il signalait tous les patients, si bien que la VPP annoncée et la VPP observée approchaient toutes deux la prévalence. Sur les tirages où son seuil était fini, son erreur était plus grande que celle de la recalibration sur deux des trois sites. Les témoins dans la distribution montraient une proportion proche de tirages signalant tous les patients, ce qui indique qu'elle tient au nombre de cas et non au décalage entre sites. La certification tenait sa garantie marginale, mais 100 étiquettes ne certifiaient pas la VPP source pour ces cibles, et à Columbia les seuils certifiés trouvés tombaient sous v plus souvent que δ.

### 4.2 Comparaison avec les travaux antérieurs

Sans étiquette, l'erreur de la correction de prévalence à Columbia, 13,1 points, était proche des 12,6 points de l'estimation sans étiquette publiée dans [1] sur la sélection de transferts de cette étude. Dans [1], un seuil recalibré sur 858 patients ambulatoires distincts de Columbia atteignait une sensibilité de 86,3 % ; la présente étude mesure combien d'étiquettes chaque méthode de mise à jour demande pour donner la VPP.

### 4.3 Forces et limites

La cible v a été fixée à partir des seuls scores source, avant toute méthode, et les témoins dans la distribution séparent l'effet du nombre de cas de celui du décalage entre sites. La comparaison repose sur trois sites, avec des transferts corrélés au sein de chacun, et les étiquettes de Beth Israel sont des comptes rendus d'appareil. Chongqing apporte un seul transfert. La VPP observée de chaque tirage est estimée sur un ensemble d'évaluation fini, ce qui ajoute de la variance d'échantillonnage à l'erreur mesurée et au taux de violation. Les tests binomiaux de la certification traitent les ECG comme indépendants alors que plusieurs peuvent venir d'un même patient, si bien que sa garantie ne tient qu'approximativement. La méthode 5 utilise le score du modèle comme seule covariable et suppose donc que le diagnostic ne dépend de l'ECG qu'à travers ce score ; la proportion de fausses découvertes à Chongqing suggère que cette hypothèse n'y tient pas. La cible v est la VPP source, et un site qui fixe une autre cible obtiendrait d'autres taux de certification et d'autres sensibilités. Les courbes de calibration ne sont pas montrées ; la calibration du modèle recalibré à 100 étiquettes est dans [`results/repairs.json`](results/repairs.json).

### 4.4 Implications

Un site qui dispose d'environ 100 étiquettes peut donner sa VPP par recalibration au seuil source. Un seuil réajusté demande neuf cas dans la moitié qui sert à l'ajuster, soit de 121 à 314 étiquettes à la prévalence médiane de ces transferts, et son erreur n'est informative que si la proportion de tirages signalant tous les patients est rapportée avec elle.

## Déclarations

Les déclarations de [1] s'appliquent à cette étude.

## Références

1. Abbou R. Predicting an ECG model's positive predictive value from a clinic's prevalence: a retrospective measurement across hospitals and care settings. Septembre 2026. [REPORT.md](REPORT.md), traduit dans [REPORT.fr.md](REPORT.fr.md), avec son supplément, [SUPPLEMENT.md](SUPPLEMENT.md).
2. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Comput* 2002;14(1):21-41. doi:10.1162/089976602753284446.
3. Angelopoulos AN, Bates S, Candès EJ, Jordan MI, Lei L. Learn then Test: calibrating predictive algorithms to achieve risk control. arXiv:2110.01052, 2021.
4. Jin Y, Candès EJ. Model-free selective inference under covariate shift via weighted conformal p-values. arXiv:2307.09291, 2023.
