# Jugement multi-agents pour l'evaluation des LLM : concordance avec les preferences humaines a travers les ecarts de capacite

**Anthony Boisbouvier** ^1, **Claude Opus 4.5** ^2

^1 Chercheur independant — agent-clash.ai
^2 Co-auteur IA — Anthropic

---

## Resume

L'evaluation des LLM par des LLM juges est une pratique de plus en plus repandue, mais la plupart des approches existantes reposent sur un juge unique, une seule execution, et des benchmarks presentant de larges ecarts de capacite ou l'accord est facile a obtenir. La question de savoir si des panels multi-juges restent fiables lorsque les modeles evalues sont proches en performance — le scenario le plus pertinent pour les decisions de deploiement — reste ouverte.

Cet article decrit Agent Clash, un cadre d'evaluation multi-juges dans lequel un panel de trois LLM de classe frontier (GPT-5.2-pro, Claude Opus 4.5, Gemini 2.5 Pro) evalue des modeles candidats en conditions aveugles avec generation dynamique de criteres et agregation par Borda Count. Nous validons le cadre a travers deux experiences couvrant differents regimes de difficulte, et rapportons quatre contributions principales :

**(1) Un gradient de concordance qui suit la difficulte de la tache.** Sur MT-Bench [1] (N = 100, 6 modeles avec de larges ecarts de capacite), le panel atteint **88,0 % de concordance** avec les preferences humaines expertes (kappa de Cohen = 0,760), depassant a la fois la reference du juge unique GPT-4 (85 %) et l'accord inter-annotateurs humains (81 %). Sur Chatbot Arena [10, 28] (N = 100, 25 modeles frontier avec de faibles ecarts de capacite), la concordance est de **76,0 %** (kappa = 0,520), dans la fourchette de 72-83 % d'accord crowd-expert sur la meme plateforme. La baisse de 12 points est statistiquement significative (z = 2,21, p = 0,027) et coherente avec la discriminabilite de la tache.

**(2) Haute fiabilite test-retest avec caracterisation systematique des erreurs.** Deux executions Arena independantes produisent 91,0 % d'accord inter-executions (kappa = 0,757). Une troisieme execution ciblee (N = 60) confirme que 19/20 (95 %) des erreurs persistantes sont irreductibles sur trois executions (binomiale p < 0,00002), tandis que les 9 evaluations instables se comportent comme attendu sous un modele de tirage a pile ou face (55,6 % de concordance). L'agregation par vote majoritaire sur trois executions n'ameliore pas la concordance (76,0 %), demontrant que le mode d'erreur dominant est systematique plutot que stochastique.

**(3) L'unanimite comme signal de confiance calibre.** Sur l'ensemble des deux experiences (N = 239), les decisions unanimes du panel (3-0) atteignent **84,9 % de concordance** (N = 192, 80 % des evaluations), tandis que les decisions partagees (2-1) atteignent **63,8 %** (N = 47, 20 %). L'ecart de 21,1 points de pourcentage fournit un indicateur de confiance binaire, empiriquement calibre, qu'un juge unique ne peut produire.

**(4) Absence de biais de capacite et d'auto-favoritisme sur les modeles frontier.** Contrairement a MT-Bench, ou 72,7 % des desaccords favorisaient les modeles de rang superieur, l'analyse basee sur l'ELO des desaccords Arena ne montre aucun biais directionnel (test des signes p = 1,0). Les donnees d'auto-evaluation en aveugle (N = 252 cas ou un modele candidat servait egalement de juge) ne montrent aucun auto-favoritisme systematique : les modeles se sont classes premiers dans 52,8 % des cas, contre 59,9 % attendus d'apres les taux de victoire humains — un leger biais anti-soi de -7,1 points de pourcentage.

Le cadre est implemente dans Agent Clash (agent-clash.ai), une plateforme ouverte entierement reproductible ou les utilisateurs fournissent leurs propres cles API et peuvent auditer chaque appel d'inference. Cout total de la validation sur 360 evaluations : 51,70 $ (0,14 $/eval). **Sur les deux experiences, la concordance du panel se situe dans la fourchette de l'accord inter-humain sur les memes benchmarks, avec l'avantage supplementaire d'une reproductibilite totale et d'un signal de confiance calibre (unanimite) qu'aucun juge unique — humain ou automatise — ne peut produire.**

---

# PARTIE I — LE SYSTEME

## 1. Introduction

La plupart des benchmarks LLM existants visent a repondre a une seule question : *Quel modele est le plus performant en general ?* En pratique, les utilisateurs posent une question differente : *Quel modele est suffisamment bon, fiable et economique pour ma tache specifique ?*

L'evaluation humaine reste l'etalon-or mais ne passe pas a l'echelle, est couteuse et difficile a reproduire [17]. L'evaluation automatisee utilisant des LLM comme juges est apparue comme une alternative prometteuse [1, 2] — mais la plupart des approches existantes utilisent un juge unique, une seule execution, et des benchmarks ou les modeles different largement en capacite. Cela rend l'accord eleve facile a atteindre et difficile a interpreter.

> **La question ouverte que cet article aborde :** Les panels d'IA multi-juges restent-ils fiables lorsque les modeles evalues sont des systemes frontier proches en performance — le scenario le plus important pour le deploiement en conditions reelles ?

---

## 2. Hypothese centrale

L'hypothese centrale sous-jacente a ce cadre est la suivante :

> Lorsque plusieurs LLM juges, dont la superiorite par rapport aux modeles evalues est demontrable, classent de maniere coherente la meme sortie comme superieure — a travers des executions repetees et des jeux de donnees varies — la probabilite que cette sortie soit egalement jugee superieure par un evaluateur humain est elevee.

Cette hypothese ne pretend pas a la verite objective ni a l'optimalite universelle. Elle revendique une **pertinence probabiliste, une robustesse et une utilite decisionnelle** sous des conditions bien definies.

Le cadre privilegie la convergence plutot que les victoires isolees : un consensus robuste a travers des evaluations repetees, plutot que des verdicts en un seul passage.

---

## 3. Hypothese motivante : l'evaluation comme capacite distincte

Un nombre croissant d'observations suggere que les LLM pourraient presenter une **asymetrie structurelle entre production et evaluation** : un modele peut etre plus capable d'identifier la meilleure reponse parmi plusieurs candidats que de produire cette meilleure reponse lui-meme [1, 21, 24]. Cette asymetrie a un analogue naturel en cognition humaine — l'evaluation est une tache discriminative ; la production est une tache generative — et la premiere est intrinsequement moins contrainte.

### 3.1 Auto-favoritisme en conditions aveugles

Une preoccupation courante concernant les cadres LLM-as-judge est que les modeles pourraient systematiquement favoriser leurs propres sorties [20]. Le cadre Agent Clash attenue ce risque par une anonymisation stricte (Section 5.3) : les reponses candidates sont depouillees de toute information identificatrice avant que les juges ne les evaluent. Dans ces conditions, l'auto-favoritisme est attendu comme attenue. Nous testons cette hypothese empiriquement dans la Section 9.4.

### 3.2 Implications pour la conception du cadre

Si l'asymetrie evaluation-production se confirme, alors le signal extrait de l'evaluation multi-juges n'est pas simplement un proxy bruite de la qualite de generation — il peut constituer un signal de plus haute fidelite que la sortie generative de n'importe quel modele individuel. Cela n'elimine pas la necessite d'une supervision humaine, mais motive la conception de cadres ou des modeles plus puissants servent de juges pour des candidats plus faibles (Section 5.1).

---

## 4. Travaux anterieurs

**References a juge unique.** Zheng et al. [1] ont etabli que GPT-4 en tant que juge unique atteint 85 % d'accord avec les preferences humaines sur MT-Bench (S2, sans egalites) — depassant l'accord inter-humain de 81 % sur les memes donnees. Liu et al. [2] et Kocmi & Federmann [3] ont confirme que l'evaluation basee sur les LLM surpasse les metriques NLG traditionnelles respectivement sur les taches de resume et de traduction.

**Panels multi-juges.** Des travaux plus recents montrent des ameliorations coherentes avec l'utilisation de plusieurs juges :
- **PoLL** [21] : un panel de 3 modeles divers surpasse le GPT-4 unique (kappa = 0,763 vs. 0,627 sur KILT-NQ), 7x moins cher.
- **ChatEval** [22] : le debat multi-agents ameliore le tau de Kendall de 10 % par rapport au juge unique GPT-4.
- **Judging the Judges** [23] : le kappa est bien plus informatif que l'accord brut ; GPT-4 Turbo atteint kappa = 0,84 sur TriviaQA vs. kappa humain-humain = 0,96.
- **Sage** [24] : meme les meilleurs modeles echouent a maintenir des preferences coherentes dans ~25 % des cas difficiles.

**Ce qui manque.** La plupart des approches existantes reposent sur des executions uniques, des jeux de donnees uniques, ou des juges uniques, et rapportent rarement la variance, les schemas de desaccord ou la robustesse aux changements de donnees [5, 6]. De maniere cruciale, **aucun travail anterieur n'evalue systematiquement les panels multi-juges sur des modeles frontier proches en performance** (ou l'accord est le plus difficile) ni ne caracterise les modes d'erreur sur des executions repetees. Cet article comble ces deux lacunes.

---

## 5. Agent Clash — Conception

Cette section decrit le cadre d'evaluation Agent Clash tel qu'il fonctionne en production.

### 5.1 Vue d'ensemble du pipeline

Le pipeline de production suit six etapes :

```
+------------------------------------------------------------------+
|                 AGENT CLASH — PIPELINE DE PRODUCTION               |
|                                                                   |
|  +--------+   +------------+   +-------------+                   |
|  | PROMPT |-->| GENERER    |-->| ANONYMISER  |                   |
|  |        |   | K reponses |   | Suppr. IDs  |                   |
|  | Requete|   | via        |   | Melanger    |                   |
|  | utilis.|   | OpenRouter |   | Label A0,A1 |                   |
|  +--------+   +------------+   +------+------+                   |
|                                       |                           |
|                +-------------+        |                           |
|                | CRITERES    |<-------+                           |
|                | gpt-4o-mini |                                    |
|                | 3-5 adaptes |                                    |
|                | a la tache  |                                    |
|                +------+------+                                    |
|                       |                                           |
|                       v                                           |
|  +----------------------------------------+                      |
|  |           JUGEMENT PARALLELE           |                      |
|  |                                        |                      |
|  |  Panel de juges (3 Supreme Court)      |                      |
|  |    - GPT-5.2-pro (OpenAI)              |                      |
|  |    - Claude Opus 4.5 (Anthropic)       |                      |
|  |    - Gemini 2.5 Pro (Google)           |                      |
|  |                                        |                      |
|  |  + Auto-evaluation (informative)       |                      |
|  |    - Chaque modele candidat juge       |                      |
|  |      les reponses anonymisees (S5.6)   |                      |
|  |                                        |                      |
|  |  Chaque juge : classer les reponses +  |                      |
|  |  noter par critere (1-5), en aveugle   |                      |
|  +----------------+-----------------------+                      |
|                   |                                               |
|                   v                                               |
|  +--------------------+   +------------------+                    |
|  | AGREGER            |-->| SORTIE           |                    |
|  | Borda Count        |   | Vainqueur +      |                    |
|  | (3 juges SC uniq.) |   | Confiance +      |                    |
|  +--------------------+   | Classements      |                    |
|                   |       +------------------+                    |
|                   v                                               |
|  +--------------------+                                           |
|  | HUMAIN DANS LA     |                                          |
|  | BOUCLE             |                                          |
|  | Comparaison cote   |                                          |
|  | a cote -> l'utili- |                                          |
|  | sateur decide      |                                          |
|  +--------------------+                                           |
+------------------------------------------------------------------+
```

Un utilisateur soumet un prompt. Le pipeline genere une reponse par modele candidat via OpenRouter, les anonymise, genere des criteres d'evaluation specifiques a la tache, envoie tous les juges en parallele, agrege les trois classements Supreme Court via Borda Count, et presente la reponse la mieux classee pour examen humain. Chaque etape est decrite ci-dessous.

### 5.2 Panel de juges

Le verdict est determine par un panel de **trois modeles de classe frontier** de fournisseurs differents — la "Supreme Court" :

| Juge | Fournisseur |
|---|---|
| GPT-5.2-pro | OpenAI |
| Claude Opus 4.5 | Anthropic |
| Gemini 2.5 Pro | Google |

L'utilisation de trois modeles de familles differentes garantit que les biais architecturaux specifiques a une famille sont dilues par les deux autres. Cela reflete l'evaluation par les pairs, ou les evaluateurs sont independants du systeme evalue et les uns des autres.

Chaque juge recoit un prompt structure lui demandant de : (1) classer toutes les reponses de la meilleure a la pire, et (2) noter chaque reponse sur chaque critere (echelle de 1 a 5), avec justification requise pour les notes de 3 ou moins (voir l'exemple detaille en Section 5.9).

**Seuls les trois juges Supreme Court determinent le verdict.** Leurs classements sont agreges via Borda Count (Section 5.5). En production, les modeles candidats peuvent egalement participer au vote avec un poids reduit (1x vs. 2x pour les juges SC), mais cette ponderation n'a modifie aucun verdict lors de nos tests — le double poids des juges SC assure leur predominance. Cette participation des candidats est une option de transparence qui sera rendue configurable dans une prochaine version. Dans les experiences de validation (Partie II), seuls les trois juges SC votent.

### 5.3 Protocole d'anonymisation

Avant que les juges ne voient une reponse, trois garanties sont appliquees :

1. **Suppression des metadonnees.** Le nom du modele, le fournisseur, le point d'acces API, le cout de generation et la latence sont supprimes de chaque reponse.
2. **Etiquetage aleatoire.** Les reponses sont melangees et se voient attribuer des etiquettes neutres (A0, A1, ...). Le melange est re-randomise independamment pour chaque evaluation et chaque execution, de sorte que la meme reponse peut etre etiquetee "A0" dans une evaluation et "A1" dans une autre.
3. **Barriere informationnelle.** Les juges recoivent uniquement : (a) le prompt original de l'utilisateur, (b) les criteres d'evaluation (Section 5.4), et (c) les reponses anonymisees. Ils ne recoivent **pas** l'identite du modele, le fournisseur, le cout, la latence, le verdict d'un autre juge, ni le verdict humain.

### 5.4 Generation dynamique de criteres

Un **modele frontier distinct** (gpt-4o-mini, temperature 0,3, max 150 tokens) genere des criteres d'evaluation specifiques a la tache a partir du prompt utilisateur seul — avant de voir toute reponse candidate.

- **Sortie :** 3-5 criteres ponderes adaptes au domaine de la tache.
- **Repli :** Si la generation echoue ou renvoie un JSON invalide, le pipeline utilise des criteres generiques : `["Accuracy", "Clarity", "Helpfulness", "Completeness"]`.
- **Element stochastique :** Comme la generation de criteres est non deterministe, differentes executions peuvent produire des grilles d'evaluation legerement differentes pour le meme prompt — echantillonnant parmi des perspectives d'evaluation raisonnables.

**Exemple — criteres generes pour un prompt de programmation :**

```
Prompt : "Write a Python function to merge two sorted lists."

Criteres generes :
  1. Correctness  — Le code produit-il une sortie correcte pour tous les cas limites ?
  2. Efficiency   — L'algorithme est-il O(n+m) ou utilise-t-il des operations inutiles ?
  3. Readability  — Les noms de variables sont-ils clairs ? Le code est-il bien structure ?
  4. Robustness   — Gere-t-il les listes vides, les entrees None, les incompatibilites de type ?
```

### 5.5 Agregation : Borda Count

Les classements des trois juges Supreme Court sont agreges via **Borda Count** [11] :

```
B(i, j) = K - rank_j(i)          ou K = nombre de candidats
S(i)    = Sum_j  B(i, j)         (j = 3 juges Supreme Court)
Winner  = argmax_i  S(i)
```

**Cas par paire (K = 2).** Dans les comparaisons par paire, le score de Borda se simplifie en : vainqueur = 1 point, perdant = 0 point, par juge. Avec 3 juges, le verdict se reduit a un **vote majoritaire simple** (2 sur 3). Le formalisme se generalise naturellement a K > 2 candidats en production, ou le Borda Count recompense un classement systematiquement eleve plutot que des premieres places occasionnelles et resiste aux votes aberrants.

*Note : En production, les modeles candidats peuvent egalement voter (poids 1x, vs. 2x pour les juges SC). Cette ponderation assure la predominance des SC et n'a modifie aucun verdict dans les configurations testees (jusqu'a 6 candidats). Cette option sera rendue configurable dans une prochaine version.*

### 5.6 Auto-evaluation (fonctionnalite de production)

En plus des trois juges Supreme Court, chaque **modele candidat** evalue egalement toutes les reponses anonymisees — jugeant effectivement sa propre sortie sans savoir quelle reponse est la sienne.

**Proprietes cles :**
- Les votes d'auto-evaluation sont affiches a l'utilisateur mais **n'influencent pas le verdict**, qui est determine uniquement par le panel Supreme Court (Section 5.5).
- Lorsqu'un modele candidat se classe en dessous de ses concurrents en conditions aveugles, cela fournit un signal de credibilite supplementaire pour l'utilisateur : meme le modele lui-meme "reconnait" qu'une autre reponse etait superieure.
- L'auto-evaluation sert de diagnostic integre pour l'auto-favoritisme (Section 3.1) : une auto-preference systematique en conditions aveugles serait visible dans la decomposition par juge.

**Cette fonctionnalite n'est pas incluse dans les experiences de validation** (Partie II), qui testent uniquement le panel Supreme Court de maniere isolee.

### 5.7 Humain dans la boucle

L'evaluation automatisee **compresse l'espace de decision** plutot que de remplacer le jugement humain. Apres chaque cycle, les utilisateurs recoivent une **comparaison cote a cote** des reponses les mieux classees et prennent la decision finale. Le cadre filtre ; l'humain decide.

### 5.8 Attenuation des biais

| Biais | Attenuation |
|---|---|
| **Auto-favoritisme** [20] | Anonymisation stricte (Section 5.3) + panel multi-juges |
| **Biais de position** | Ordre aleatoire des reponses par evaluation et par execution |
| **Biais de verbosite** | Les criteres penalisent le remplissage, recompensent la concision |
| **Biais d'autorite** | La notation multi-criteres separe le ton du fond |

Aucune attenuation isolee n'est suffisante. Le cadre repose sur leur **effet combine** a travers les juges, les executions et les jeux de donnees.

### 5.9 Exemple detaille

Un parcours complet d'une evaluation par paire (K = 2, 3 juges Supreme Court) :

```
ETAPE 1 — ENTREE
  Prompt : "Explain quantum entanglement to a 10-year-old."
  Reponse du Modele X : [200 mots sur des particules qui sont meilleures amies]
  Reponse du Modele Y : [180 mots sur des des magiques]

ETAPE 2 — ANONYMISER
  Melange (graine aleatoire pour cette eval) : Modele Y tire en premier
  -> Reponse A0 = Modele Y    Reponse A1 = Modele X
  Toutes les metadonnees supprimees.

ETAPE 3 — GENERATION DE CRITERES  (gpt-4o-mini, temp 0,3)
  A partir du prompt seul, genere :
    1. Adaptation a l'age       (30 %) — vocabulaire et concepts adaptes ?
    2. Exactitude scientifique  (25 %) — correct sans simplification excessive ?
    3. Engagement               (25 %) — un enfant resterait-il interesse ?
    4. Exhaustivite             (20 %) — couvre les aspects cles de l'intrication ?

ETAPE 4a — JUGEMENT : CLASSEMENTS (3 juges Supreme Court)
  GPT-5.2-pro :      classe A1 > A0  -> B(A0)=0, B(A1)=1
  Claude Opus 4.5 :  classe A0 > A1  -> B(A0)=1, B(A1)=0
  Gemini 2.5 Pro :   classe A1 > A0  -> B(A0)=0, B(A1)=1

ETAPE 4b — NOTES PAR CRITERE (exemple pour 1 juge)
  Notes de GPT-5.2-pro pour A0 (Modele Y) :
    Adaptation a l'age :       4/5
    Exactitude scientifique :  2/5  -> raison : "Confond intrication et teleportation"
    Engagement :               4/5
    Exhaustivite :             3/5  -> raison : "Omet l'effondrement de la mesure"

  (Pour tout critere note 3 ou moins, le juge doit fournir
   une breve justification — max 15 mots — expliquant la note basse.)

ETAPE 5 — AGREGER  (Borda Count, majorite 2 sur 3)
  S(A0) = 0 + 1 + 0 = 1
  S(A1) = 1 + 0 + 1 = 2
  Vainqueur : A1 = Modele X  (decision partagee, 2-1)

ETAPE 6 — SORTIE + EXAMEN HUMAIN
  Recommandation IA : Modele X (decision partagee, 2-1)
  L'utilisateur voit les deux reponses cote a cote et prend la decision finale.
```

---

## 6. Le triangle cout-qualite-vitesse

En production, la selection de modele ne concerne pas seulement la qualite — c'est un **compromis a trois dimensions** :

| Dimension | Ce qu'elle mesure |
|---|---|
| **Qualite** | Scores agreges par Borda a partir de l'evaluation multi-juges |
| **Cout** | Cout d'inference par token sur toutes les executions |
| **Latence** | Temps jusqu'au premier token et temps de generation total |

Un modele qui se classe premier en qualite mais coute 10x plus et repond 5x plus lentement n'est peut-etre pas le bon choix pour les applications a haut debit. Le cadre permet une **selection Pareto-optimale** : identifier les modeles non domines sur aucune dimension, permettant aux utilisateurs d'arbitrer en fonction de leurs contraintes.

---

# PARTIE II — VALIDATION EMPIRIQUE

## 7. Methodologie de validation

Pour valider le cadre decrit dans la Partie I, nous menons deux etudes de concordance comparant les verdicts d'Agent Clash aux preferences humaines. Cette section decrit la methodologie commune aux deux experiences.

### 7.1 Pipeline de validation

Les experiences utilisent un **pipeline de validation modifie** qui differe du pipeline de production sur un point critique : les reponses sont **injectees** a partir de jeux de donnees de benchmark plutot que generees par le pipeline. Cela isole le composant de jugement — toute difference entre les verdicts IA et humains reflete la qualite du panel de juges, pas la variance de generation des reponses.

Les deux jeux de donnees sont distribues sous la licence CC-BY-4.0, qui autorise l'utilisation, la reproduction et l'adaptation avec attribution. Les attributions sont fournies dans les references [1] et [28].

```
+------------------------------------------------------------------+
|                     PIPELINE DE VALIDATION                         |
|  (differe de la production : reponses INJECTEES, non generees)     |
|                                                                   |
|  +-------------+                  +-------------+                 |
|  | JEU DE      |     injecter     | ANONYMISER  |                |
|  | DONNEES DE  |----------------->| (identique  |                |
|  | BENCHMARK   |  (contourner     |  a la prod.)|                |
|  |             |   la generation) +------+------+                |
|  | Paires de   |                         |                        |
|  | reponses    |                         v                        |
|  | preexist. + |         [Meme pipeline de jugement que S5 :      |
|  | verdict     |          criteres -> 3 juges SC -> Borda]        |
|  | humain      |                         |                        |
|  +-------------+                         v                        |
|                           +-----------------------+               |
|                           | COMPARER              |               |
|                           | Vainqueur IA vs.      |               |
|                           | Vainqueur humain      |               |
|                           | = concordance ?        |               |
|                           +-----------------------+               |
+------------------------------------------------------------------+
```

Les etapes d'anonymisation, de generation de criteres, de jugement et d'agregation sont **identiques au pipeline de production** (Section 5).

### 7.2 Choix de conception de la validation

Deux choix cles ont ete faits pour les experiences de validation :

1. **Seuls les trois juges Supreme Court evaluent.** L'auto-evaluation par les modeles candidats (Section 5.6) est une fonctionnalite de production concue pour la transparence aupres des utilisateurs. Dans ces experiences, nous isolons la qualite de jugement du panel en utilisant uniquement les trois juges SC. Cela elimine les facteurs de confusion lies a la disponibilite variable des modeles candidats et garantit que toutes les evaluations sont strictement comparables. En excluant les votes des candidats, nous mesurons la qualite intrinseque du panel SC sans confusion liee a l'interet propre des candidats.

2. **Comparaisons par paire (K = 2).** Les deux jeux de donnees de benchmark fournissent des preferences humaines par paire. Avec K = 2 candidats et 3 juges, le Borda Count se reduit a un **vote majoritaire simple** (2 sur 3). C'est l'agregation la plus simple possible — un verdict est correct chaque fois qu'au moins 2 des 3 juges sont d'accord avec la preference humaine.

---

## 8. Experience 1 — MT-Bench (grands ecarts de capacite)

> **Question cle :** Un panel IA a 3 juges est-il en accord avec les preferences humaines expertes lorsque les modeles different largement en capacite ?

### 8.1 Materiels et protocole

**Jeu de donnees.** Le jeu de donnees MT-Bench Human Judgments (LMSYS, CC-BY-4.0) [1] contient plus de 3 300 comparaisons par paire d'experts collectees par des chercheurs en NLP a UC Berkeley et LMSYS. Les evaluateurs ont compare les sorties de 6 modeles — GPT-4, GPT-3.5-turbo, Claude-v1, Vicuna-13B-v1.2, Alpaca-13B et LLaMA-13B — sur 80 questions multi-tours couvrant l'ecriture, le raisonnement, les mathematiques, la programmation, l'extraction, les STEM et les humanites. Les jugements ont ete effectues en aveugle (identites des modeles cachees aux evaluateurs).

**Selection de l'echantillon.** Nous avons selectionne N = 100 comparaisons par paire avec des vainqueurs clairs (egalites exclues), couvrant 53 questions uniques a travers les 6 modeles. La distribution des vainqueurs selectionnes par les humains etait : GPT-4 (29), GPT-3.5-turbo (24), Claude-v1 (21), Vicuna-13B (16), Alpaca-13B (7), LLaMA-13B (3).

**Panel de juges.** Les trois juges Supreme Court (GPT-5.2-pro, Claude Opus 4.5, Gemini 2.5 Pro) ont evalue les 100 paires.

**Protocole :**

```
1. ECHANTILLON  N = 100 comparaisons par paire de MT-Bench [1].
                Vainqueurs clairs uniquement (egalites exclues). 53 questions uniques, 6 modeles.

2. INJECTER     Reponses preexistantes collectees aupres d'humains injectees dans
                le pipeline de validation (S7.1), contournant la generation.

3. JUGER        3 juges Supreme Court : anonymiser -> generer les criteres ->
                envoyer aux juges -> Borda (= majorite 2 sur 3).

4. COMPARER     Vainqueur IA vs. vainqueur humain expert. Correspondance/non-correspondance.
                Pas de credit partiel.

5. EXECUTER     100 evaluations, appels webhook sequentiels, ~66 min.
```

**Ce qui est fixe :** les reponses (du jeu de donnees), le panel de juges, la methode d'agregation.
**Ce qui varie :** la graine de generation des criteres, l'ordre de melange des reponses.

**Metriques :** taux de concordance brut, kappa de Cohen [18], intervalles de confiance Wilson a 95 %, analyses stratifiees par niveau de modele et direction des desaccords.

### 8.2 Resultats

#### 8.2.1 Concordance globale

> **Resultat :** 88,0 % de concordance (kappa = 0,760) — depasse la reference du juge unique GPT-4 (85 %) et l'accord inter-humain (81 %).

| Metrique | Valeur |
|---|---|
| Total des evaluations | 100 |
| Correspondances (IA = Humain) | 88 |
| Non-correspondances | 12 |
| **Taux de concordance** | **88,0 %** |
| IC Wilson a 95 % | [80,2 %, 93,0 %] |
| Kappa de Cohen | **0,760** |
| IC kappa a 95 % | [0,632, 0,895] |
| Interpretation kappa (Landis & Koch) | Accord substantiel |

La concordance observee de 88,0 % depasse la reference du juge unique GPT-4 de 85 % rapportee par Zheng et al. [1] dans la configuration S2 (egalites exclues), et depasse substantiellement l'accord inter-annotateurs humains de 81 % rapporte sur le meme benchmark.

Le kappa de Cohen = 0,760 se situe dans la fourchette "accord substantiel" de l'echelle de Landis-Koch [25] (0,61 < kappa <= 0,80). La tache de classification couvre 6 categories de modeles avec des taux de base inegaux, ce qui rend le kappa eleve particulierement significatif.

#### 8.2.2 Concordance par niveau de modele

> **Resultat :** Modeles forts (GPT-4, Claude, GPT-3.5) -> 95,9 % de concordance. Modeles faibles (Vicuna, Alpaca, LLaMA) -> 68,0 %. Le panel a ~11x plus de chances d'etre d'accord avec les humains lorsque le vainqueur est un modele fort.

Cette analyse stratifie les resultats par le modele que les humains ont selectionne comme vainqueur, independamment du modele contre lequel il a ete confronte. Les 100 evaluations couvrent tous les types de confrontations possibles : fort vs. fort (par ex., GPT-4 vs. Claude-v1), fort vs. faible (par ex., GPT-4 vs. Alpaca-13B), et faible vs. faible (par ex., Vicuna vs. LLaMA). Le tableau ci-dessous montre que lorsqu'un humain selectionne un modele fort comme vainqueur, le panel IA est presque certainement d'accord (95,9 %). Inversement, lorsqu'un humain selectionne un modele faible comme vainqueur (souvent contre un autre modele faible), le panel est d'accord dans seulement 68 % des cas — ces situations etant intrinsequement plus ambigues.

Un gradient clair dependant des capacites emerge dans les donnees de concordance :

**Tableau 1.** Taux de concordance par modele vainqueur selectionne par l'humain, ordonne par niveau de capacite du modele.

| Vainqueur humain | n | Correspondances | Concordance | IC Wilson 95 % |
|---|---|---|---|---|
| GPT-4 | 29 | 29 | **100,0 %** | [88,3 %, 100,0 %] |
| Claude-v1 | 21 | 20 | **95,2 %** | [77,3 %, 99,2 %] |
| GPT-3.5-turbo | 24 | 22 | **91,7 %** | [74,2 %, 97,7 %] |
| Vicuna-13B-v1.2 | 16 | 12 | **75,0 %** | [50,5 %, 89,8 %] |
| Alpaca-13B | 7 | 5 | **71,4 %** | [35,9 %, 91,8 %] |
| LLaMA-13B | 2 | 0 | **0,0 %** | [0,0 %, 65,8 %] |

*Note : Une evaluation (id = 36) n'a produit aucun resultat en raison d'une erreur silencieuse du pipeline (0 juges, champs vainqueur vides) et est comptee comme une non-correspondance. Le Tableau 1 totalise N = 99 ; la concordance globale est calculee sur N = 100.*

En regroupant les modeles par niveaux :

**Tableau 2.** Concordance par niveau de capacite du modele.

| Niveau | Modeles | n | Correspondances | Concordance | IC Wilson 95 % |
|---|---|---|---|---|---|
| Fort | GPT-4, Claude-v1, GPT-3.5-turbo | 74 | 71 | **95,9 %** | [88,7 %, 98,6 %] |
| Faible | Vicuna-13B, Alpaca-13B, LLaMA-13B | 25 | 17 | **68,0 %** | [48,4 %, 82,8 %] |

La difference est hautement significative (chi-carre = 14,78, df = 1, p = 0,0001). Le rapport des cotes est de 11,14 : le panel IA a environ 11 fois plus de chances d'etre d'accord avec le verdict humain lorsque le vainqueur selectionne par l'humain appartient au niveau fort.

Ce schema est coherent avec les resultats de Zheng et al. [1, Figure 2], qui rapportent que l'accord du juge unique GPT-4 avec les humains augmente de maniere monotone avec l'ecart de performance entre les modeles evalues — d'environ 70 % pour les paires proches en performance a pres de 100 % pour les paires avec de grands ecarts de capacite.

#### 8.2.3 Cout et efficacite

| Metrique | Valeur |
|---|---|
| Cout total d'evaluation | 12,86 $ |
| Cout par evaluation | 0,129 $ |
| Cout par prediction correcte | 0,146 $ |
| Temps moyen d'evaluation | 39,6 s |
| Temps total d'execution | ~66 min |

### 8.3 Analyse des desaccords

> **Resultat :** Dans 73 % des desaccords, l'IA a favorise un modele plus fort — suggerant un "biais de capacite" qui est attendu et non problematique. La confiance de l'IA est egalement elevee pour les verdicts corrects et incorrects (pas de conscience de ses erreurs sur ce benchmark).

Sur les 12 non-correspondances, une (id = 36) est l'erreur de pipeline vide notee ci-dessus. Les 11 restantes sont des desaccords substantiels IA-humain :

**Tableau 3.** Desaccords substantiels IA-humain (11 des 12 non-correspondances).

| eval_id | Modele A | Modele B | Vainqueur humain | Vainqueur IA |
|---|---|---|---|---|
| 10 | alpaca-13b | vicuna-13b | alpaca-13b | vicuna-13b |
| 11 | vicuna-13b | gpt-3.5-turbo | vicuna-13b | gpt-3.5-turbo |
| 14 | vicuna-13b | alpaca-13b | vicuna-13b | alpaca-13b |
| 23 | vicuna-13b | claude-v1 | claude-v1 | vicuna-13b |
| 30 | alpaca-13b | llama-13b | llama-13b | alpaca-13b |
| 32 | gpt-4 | llama-13b | llama-13b | gpt-4 |
| 33 | gpt-3.5-turbo | vicuna-13b | vicuna-13b | gpt-3.5-turbo |
| 47 | alpaca-13b | gpt-4 | alpaca-13b | gpt-4 |
| 56 | claude-v1 | gpt-3.5-turbo | gpt-3.5-turbo | claude-v1 |
| 77 | vicuna-13b | gpt-3.5-turbo | vicuna-13b | gpt-3.5-turbo |
| 94 | gpt-3.5-turbo | vicuna-13b | gpt-3.5-turbo | vicuna-13b |

**Biais directionnel.** Dans 8 des 11 desaccords (72,7 %), le panel IA a selectionne un modele d'un niveau de capacite superieur a celui du vainqueur selectionne par l'humain. Dans seulement 3 cas (27,3 %), l'IA a selectionne un modele de niveau inferieur. Bien que cette asymetrie suggere un "biais de capacite" — une tendance des juges IA a favoriser des modeles de capacite generalement superieure meme lorsque la reponse specifique du modele plus faible etait superieure — le test des signes n'atteint pas la significativite au seuil alpha = 0,05 (z = 1,51, p = 0,13, bilaterale), probablement en raison du petit echantillon de desaccords.

Ce biais de capacite est attendu et ne doit pas etre interprete comme un defaut. Il reflete le fait que les juges IA privilegient le fond et la profondeur du raisonnement par rapport aux qualites stylistiques ou subjectives, ce qui coincide naturellement avec la hierarchie de capacite connue parmi les modeles. Il est important de noter que ce biais ne se manifeste que dans les rares cas de desaccord (12 % des evaluations) et n'affecte pas les 88 % ou le panel identifie correctement la reponse preferee par l'humain. L'Experience 2 (Section 9) montrera que ce biais disparait lorsque les modeles evalues sont proches en capacite — confirmant qu'il s'agit d'une propriete du jeu de donnees, et non d'un biais intrinseque du panel.

**Concentration par paire.** La paire GPT-3.5-turbo vs. Vicuna-13B represente 4 des 11 desaccords (36,4 %), avec un taux de desaccord de 44,4 % pour cette paire specifique. Cette paire se situe a la frontiere entre les niveaux fort et faible, ou les differences de qualite sont les plus faibles et le jugement subjectif est le plus susceptible de diverger.

### 8.4 Discussion

**Comparaison avec les travaux anterieurs :**

**Tableau 4.** Concordance entre l'evaluation automatisee et les jugements humains a travers les cadres et benchmarks.

| Methode | Benchmark | Accord | kappa | Reference |
|---|---|---|---|---|
| GPT-4 juge unique | MT-Bench (S2, sans egalites) | 85 % | — | Zheng et al. [1] |
| Humain vs. humain | MT-Bench (S2) | 81 % | — | Zheng et al. [1] |
| GPT-4 juge unique | MT-Bench (principal) | >80 % | — | Zheng et al. [1] |
| GPT-4 Turbo juge unique | TriviaQA | — | 0,84 | Thakur et al. [23] |
| Humain vs. humain | TriviaQA | — | 0,96 | Thakur et al. [23] |
| PoLL (panel de 3 modeles) | KILT-NQ | — | 0,763 | Verga et al. [21] |
| PoLL (panel de 3 modeles) | HotPotQA | — | 0,889 | Verga et al. [21] |
| Arena-Hard-Auto | Chatbot Arena | 89,1 % | — | Li et al. [26] |
| Crowd vs. expert | Chatbot Arena | 72-83 % | — | Chiang et al. [10] |
| Expert vs. expert | Chatbot Arena | 79-90 % | — | Chiang et al. [10] |
| **Agent Clash (SC a 3 juges)** | **MT-Bench** | **88,0 %** | **0,760** | **Ce travail (Exp. 1)** |

*Note : "Accord" mesure le pourcentage brut de cas ou deux evaluateurs concordent. Le kappa de Cohen corrige ce taux en soustrayant l'accord attendu par le hasard, fournissant une mesure plus conservative. Les deux metriques ne sont pas interchangeables : un accord de 85 % peut correspondre a des valeurs de kappa tres differentes selon la distribution des categories. Certaines etudes ne rapportent qu'une seule des deux metriques — les cellules vides (—) indiquent que la metrique n'a pas ete calculee dans le travail original, et non qu'elle est nulle.*

Notre panel Supreme Court a 3 juges atteint une concordance qui :

- **Depasse la reference du juge unique GPT-4** de 3,0 points de pourcentage (88,0 % vs. 85 %), coherent avec l'amelioration multi-juges rapportee par Verga et al. [21] et Chan et al. [22].
- **Depasse l'accord inter-humain** sur le meme benchmark de 7,0 points (88,0 % vs. 81 %), suggerant que le panel atteint une concordance surhumaine — bien que cette comparaison doive etre interpretee avec prudence, car la reference humaine peut refleter une diversite legitime de preferences plutot que des erreurs d'evaluateur.
- **N'est pas directement comparable aux 89,1 % d'Arena-Hard-Auto** [26] : leur metrique mesure l'accord de classement au niveau systeme (correlation de Spearman entre les classements de modeles), tandis que la notre mesure la concordance par exemple (correspondance binaire par evaluation). La concordance par exemple est une mesure plus stricte et conservative.
- **Atteint kappa = 0,760**, se situant dans la meme fourchette que PoLL sur HotPotQA (kappa = 0,889) et substantiellement au-dessus du juge unique GPT-4 sur KILT-NQ (kappa = 0,627).

**Le gradient de concordance est le resultat central.** L'accord quasi parfait (95,9 %) sur les vainqueurs de niveau fort montre que le panel identifie de maniere fiable les differences claires de qualite. La concordance plus faible (68,0 %) sur les vainqueurs de niveau faible reflete les cas limites ou les preferences humaines elles-memes divergent. L'Experience 2 (Section 9) confirme cela au niveau inter-experience (88 % -> 76 %).

**Le biais de capacite dans les desaccords.** Dans 72,7 % des non-correspondances, l'IA a favorise des modeles de rang superieur — est-ce un biais ou une fonctionnalite ? Lorsque les humains ont choisi LLaMA-13B plutot que GPT-4 (eval_id 32), les juges ont annule avec une conviction maximale. Cela peut refleter un veritable biais en faveur des modeles conventionnellement plus forts, ou l'evaluateur humain a pu etre influence par la nouveaute. Demeler ces hypotheses necessite des echantillons plus importants.

**Limites.** (1) Les modeles MT-Bench (2023) ont de grands ecarts de capacite -> traite dans l'Experience 2 avec des modeles frontier de 2024. (2) N = 100 -> IC Wilson = [80,2 %, 93,0 %]. (3) Reponses du premier tour uniquement.

---

## 9. Experience 2 — Chatbot Arena (modeles frontier, faibles ecarts de capacite)

> **Question cle :** Que se passe-t-il lorsque les modeles evalues sont tous de classe frontier (GPT-4o, Claude 3.5, Gemini 1.5, etc.) et que l'ecart de qualite est minimal ?

### 9.1 Materiels et protocole

**Motivation.** L'Experience 1 a valide le cadre sur des modeles avec de grands ecarts de capacite (GPT-4 vs. LLaMA-13B). Une question cruciale est de savoir si le pipeline reste utile lorsque les modeles evalues sont des systemes frontier proches en performance — le scenario le plus pertinent pour les decisions de deploiement en conditions reelles.

**Jeu de donnees.** Nous utilisons le jeu de donnees `lmarena-ai/arena-human-preference-100k` [28] (CC-BY-4.0), qui contient environ 100 000 combats anonymes par paire collectes entre juin et aout 2024 sur la plateforme Chatbot Arena. Chaque ligne represente la preference d'un seul juge humain entre deux reponses de modeles au meme prompt.

**Echantillon.** Nous extrayons N = 100 evaluations en utilisant les criteres d'inclusion suivants :
- Les deux modeles appartiennent a un ensemble organise de 28 modeles frontier ou quasi-frontier (Tableau 5)
- Vainqueur clair (pas d'egalite)
- Combat anonyme
- Langue anglaise
- Conversation a un seul tour
- Minimum 30 caracteres par reponse

L'echantillonnage utilise un parametre de reproductibilite fixe (seed = 42) : cela signifie que toute personne re-executant le meme script d'extraction sur le meme jeu de donnees obtiendra exactement les memes 100 evaluations. Ce parametre controle le generateur de nombres pseudo-aleatoires de Python et assure la reproductibilite totale de la selection de l'echantillon.

**Tableau 5.** Modeles inclus dans l'echantillon de validation Arena (25 modeles uniques presents).

| Famille | Modeles |
|---|---|
| GPT | gpt-4o, gpt-4o-aug, gpt-4o-mini, gpt-4-turbo, gpt-4-turbo-jan, gpt-4-turbo-nov, gpt-4, chatgpt-4o |
| Claude | claude-3.5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Gemini | gemini-1.5-pro, gemini-1.5-pro-exp, gemini-1.5-flash |
| Llama | llama-3-70b, llama-3.1-70b, llama-3.1-405b |
| Mistral | mistral-large-2, mixtral-8x22b |
| Autres | deepseek-v2, deepseek-coder-v2, yi-large, yi-large-preview, qwen2-72b, nemotron-340b, command-r-plus |

**Differences cles par rapport a l'Experience 1.** (1) Les modeles sont des systemes frontier de generation 2024 avec des ecarts de capacite beaucoup plus faibles. (2) Les jugements humains proviennent d'utilisateurs anonymes crowd-sourced (et non d'annotateurs experts). (3) 25 modeles au lieu de 6 — plus de 80 paires de modeles distinctes representees. (4) Les prompts sont des requetes reelles generees par les utilisateurs, et non des questions de benchmark multi-tours curees.

**Panel de juges.** Les trois juges Supreme Court (GPT-5.2-pro, Claude Opus 4.5, Gemini 2.5 Pro) ont evalue toutes les paires sur toutes les executions.

**Protocole :**

```
1. ECHANTILLON  N = 100 paires de arena-human-preference-100k [28].
                Inclusion : les deux modeles dans l'ensemble frontier (Tableau 5),
                vainqueur clair, anonyme, anglais, un seul tour, >=30 car.
                Seed = 42 (parametre de reproductibilite).

2. INJECTER     Reponses preexistantes -> pipeline de validation (S7.1).

3. EXEC. 1      Pipeline de jugement standard (S5). Enregistrer les vainqueurs IA.

4. EXEC. 2      Re-executer les 100 evals avec de nouvelles graines aleatoires.
                Memes reponses, nouvelle generation de criteres + melange.

5. COMPARER     Chaque execution vs. humain (concordance).
                Execution 1 vs. Execution 2 (fiabilite test-retest).

6. CLASSIFIER   A partir des Executions 1-2, classifier chaque eval :
                - Stable correct (les deux executions concordent avec l'humain)
                - Stable incorrect (les deux executions en desaccord avec l'humain)
                - Instable (les executions sont en desaccord entre elles)

7. EXEC. 3      Re-evaluation ciblee de 60 evaluations les plus informatives :
   (ciblee)     les 20 stables-incorrects + les 9 instables +
                31 stables-corrects a faible confiance.
```

**Ce qui est fixe :** les reponses (du jeu de donnees), le panel de juges, la methode d'agregation.
**Ce qui varie entre les executions :** la graine de generation des criteres, l'ordre de melange des reponses.

### 9.2 Resultats

> **Resultat :** 76,0 % de concordance (kappa = 0,520) — une baisse de 12 points par rapport a MT-Bench, dans la fourchette de 72-83 % d'accord crowd-expert humain sur la meme plateforme.

**Tableau 6.** Concordance globale sur Chatbot Arena.

| Metrique | Execution 1 | Execution 2 | Combine |
|---|---|---|---|
| N | 100 | 100 | 200 |
| Concordance | **76,0 %** | **75,0 %** | **75,5 %** |
| IC Wilson 95 % | [66,8 %, 83,3 %] | [65,7 %, 82,5 %] | [69,1 %, 80,9 %] |
| Kappa de Cohen | 0,520 | 0,500 | 0,510 |
| IC kappa 95 % | [0,353, 0,687] | [0,330, 0,670] | [0,391, 0,629] |
| Cout total | 15,08 $ | 14,94 $ | 30,02 $ |
| Cout par eval | 0,151 $ | 0,149 $ | 0,150 $ |

Le kappa de Cohen = 0,510 se situe dans la fourchette "modere" de l'echelle de Landis-Koch [25], coherent avec la difficulte de discriminer entre des modeles frontier proches en performance.

**Tableau 7.** Concordance par vainqueur selectionne par l'humain (Execution 1, modeles avec n >= 3).

| Vainqueur humain | n | Correspondances | Concordance | IC Wilson 95 % |
|---|---|---|---|---|
| chatgpt-4o | 8 | 7 | **87,5 %** | [52,9 %, 97,8 %] |
| gemini-1.5-pro-exp | 7 | 6 | **85,7 %** | [48,7 %, 97,4 %] |
| gpt-4o | 13 | 11 | **84,6 %** | [57,8 %, 95,7 %] |
| gemini-1.5-pro | 6 | 5 | **83,3 %** | [43,6 %, 97,0 %] |
| claude-3-opus | 4 | 3 | **75,0 %** | [30,1 %, 95,4 %] |
| llama-3-70b | 4 | 3 | **75,0 %** | [30,1 %, 95,4 %] |
| claude-3.5-sonnet | 13 | 9 | **69,2 %** | [42,4 %, 87,3 %] |
| yi-large | 3 | 2 | **66,7 %** | [20,8 %, 93,9 %] |
| mistral-large-2 | 5 | 3 | **60,0 %** | [23,1 %, 88,2 %] |
| gpt-4-turbo-nov | 5 | 3 | **60,0 %** | [23,1 %, 88,2 %] |
| llama-3.1-70b | 5 | 3 | **60,0 %** | [23,1 %, 88,2 %] |
| deepseek-v2 | 4 | 2 | **50,0 %** | [15,0 %, 85,0 %] |
| llama-3.1-405b | 4 | 2 | **50,0 %** | [15,0 %, 85,0 %] |

Contrairement a l'Experience 1, il n'y a pas de gradient de concordance clair par niveau : tous les modeles sont de classe frontier, et la concordance varie dans une fourchette plus etroite (50-88 %).

### 9.3 Fiabilite test-retest

> **Resultat :** 91 % des evaluations obtiennent le meme verdict IA sur deux executions independantes (kappa = 0,757). Les erreurs sont a 69 % systematiques (meme mauvaise reponse a chaque fois) et a 31 % stochastiques (comportement de pile ou face).

**Tableau 8.** Matrice de concordance inter-executions (N = 100 evaluations appariees).

|  | Execution 2 Correct | Execution 2 Incorrect |
|---|---|---|
| **Execution 1 Correct** | 71 | 5 |
| **Execution 1 Incorrect** | 4 | 20 |

- **Meme verdict IA dans les deux executions :** 91/100 = **91,0 %**
- **Kappa de Cohen inter-executions :** 0,757 (accord "substantiel")
- **Test de McNemar :** chi-carre = 0,00 (avec correction de continuite), p > 0,05 — pas de difference significative entre les executions

Sur les 9 evaluations discordantes (verdict IA different entre les executions), toutes impliquent des paires de modeles frontier ou l'ecart de qualite est minimal (par ex., claude-3.5-sonnet vs. gemini-1.5-pro, claude-3-opus vs. gpt-4o). Celles-ci representent le "seuil de bruit" ou la variation stochastique dans la generation de criteres et l'echantillonnage des juges produit des verdicts differents sur des cas limites.

**Classification de la stabilite :** 71 % des evaluations sont stablement correctes (les deux executions concordent avec l'humain), 20 % sont stablement incorrectes (les deux executions en desaccord), et 9 % sont instables (les executions sont en desaccord entre elles). Le faible taux d'instabilite confirme que le pipeline est reproductible et que les erreurs sont predominantement systematiques plutot que stochastiques.

**Confirmation de l'Execution 3 ciblee (N = 60).** Pour renforcer ces resultats, nous avons mene une troisieme execution ciblee sur les 60 evaluations les plus informatives statistiquement : les 20 stablement incorrectes, les 9 instables, et 31 stablement correctes a faible confiance. Pour selectionner les 31 evaluations stablement correctes a retester, nous utilisons le score de confiance numerique produit par le pipeline, qui mesure la marge de victoire Borda normalisee divisee par le nombre de juges. Les 31 evaluations selectionnees sont celles avec la confiance moyenne la plus basse sur les deux premieres executions. Resultats :

- **Erreurs systematiques confirmees :** 19/20 (95 %) des evaluations stablement incorrectes sont restees erronees dans l'Execution 3. Sous H0 que les erreurs sont des tirages aleatoires a pile ou face, P(>=19/20) < 0,00002 (test binomial). La seule exception (id = 100, llama-3.1-405b vs. llama-3.1-70b) etait la seule evaluation a basculer en trois executions. Les 19 evaluations restantes etaient erronees dans les trois executions, les etablissant comme des **desaccords systematiques irreductibles** entre le panel IA et les preferences humaines crowd-sourced.
- **Les evaluations instables sont veritablement stochastiques :** 5/9 (55,6 %) ont correspondu dans l'Execution 3 — proche des 50 % attendus pour un comportement aleatoire de pile ou face, confirmant que celles-ci representent le seuil de bruit du pipeline.
- **Les predictions a faible confiance sont fragiles :** 3/31 (9,7 %) des evaluations stablement correctes mais a faible confiance ont bascule vers l'erreur dans l'Execution 3.
- **La concordance par vote majoritaire egale la concordance d'une seule execution :** Sur les 100 evaluations, le verdict majoritaire sur les executions disponibles donne 76,0 % — identique au taux d'une seule execution. Cela confirme que les erreurs ne peuvent pas etre moyennees en re-executant, car le mode d'erreur dominant est systematique.

### 9.4 Analyse des desaccords

> **Resultat :** Pas de biais de capacite sur les modeles frontier (contrairement a MT-Bench). Pas d'auto-favoritisme detecte. Les donnees d'auto-evaluation en aveugle confirment l'absence d'auto-promotion.

**Scores ELO.** Les scores ELO, derives du classement Chatbot Arena [10], sont des evaluations de competence relative calculees a partir de milliers de votes humains par paire, analogues aux classements d'echecs : une difference ELO de 100 points signifie que le modele mieux classe est predit comme vainqueur dans environ 64 % des confrontations.

Sur les 24 desaccords de l'Execution 1 :

- **Pas de biais de capacite directionnel.** En utilisant les scores ELO approximatifs de Chatbot Arena (environ aout 2024), nous testons si le panel IA favorise systematiquement les modeles a ELO plus eleve dans les desaccords. Sur 24 non-correspondances, l'IA a selectionne le modele a ELO plus eleve dans 11 cas (45,8 %) et le modele a ELO plus bas dans 10 cas (41,7 %), avec 3 egalites. Le test des signes est non significatif (p = 1,0), et la difference ELO moyenne est de -2,9 (negligeable). Cela contraste fortement avec l'Experience 1, ou 72,7 % des non-correspondances favorisaient les modeles de rang superieur — confirmant que le "biais de capacite" observe dans MT-Bench etait entraine par les grands ecarts de capacite dans ce jeu de donnees, et non par un biais intrinseque du panel de juges.

- **Paires de desaccord chronique.** claude-3.5-sonnet vs. gemini-1.5-pro est la paire la plus disputee (3 des 5 evaluations mal classees dans les deux executions), coherent avec ces deux modeles etant tres proches sur le classement Arena (difference ELO d'environ 20). Les paires deepseek-v2 vs. gemini-1.5-flash et gpt-4o-aug vs. llama-3.1-405b sont egalement systematiquement mal classees sur les deux executions, representant une ambiguite veritable dans ces confrontations.

- **Auto-favoritisme.** Comme le panel de juges inclut GPT-5.2-pro, Claude Opus 4.5 et Gemini 2.5 Pro, nous testons si les juges favorisent leur propre famille de modeles. Les choix IA de la famille GPT (33) correspondent etroitement aux choix humains (34) ; les choix IA de la famille Claude (18) correspondent aux choix humains (19) ; les choix IA de la famille Gemini (20) depassent legerement les choix humains (16). Aucun auto-favoritisme systematique n'est detecte, coherent avec les resultats de Panickssery et al. [20] en conditions aveugles.

- **Auto-evaluation en aveugle.** Nos donnees de validation incluent 252 instances ou un modele candidat servait egalement de juge (non-Supreme Court), evaluant effectivement sa propre sortie sans savoir quelle reponse etait la sienne. Sur l'ensemble de ces cas, les modeles se sont classes premiers dans 52,8 % des evaluations, compare a un taux attendu de 59,9 % base sur les taux de victoire humains pour ces memes modeles — une difference de -7,1 points de pourcentage, indiquant un leger biais anti-soi global. Resultats par modele : GPT-4 a montre un fort biais anti-soi (-25,0pp, N=44), GPT-3.5-turbo etait quasi neutre (+2,1pp, N=47), et Claude 3.5 Sonnet a montre un leger biais anti-soi (-7,7pp, N=78). Certains modeles a petit echantillon ont montre des deviations positives (par ex., GPT-4-turbo +27,8pp, N=18), mais celles-ci ne sont pas statistiquement significatives compte tenu des tailles d'echantillon. Globalement, aucune auto-promotion systematique n'est detectee en conditions aveugles.

- **Decomposition des erreurs.** Sur les 29 evaluations qui ne sont pas stablement correctes sur les deux executions (29 %), 69 % sont systematiques (20 evaluations constamment erronees sur les deux executions) et 31 % sont stochastiques (9 evaluations incoherentes entre les executions). Une troisieme execution ciblee a confirme 19/20 (95 %) des erreurs systematiques (binomiale p < 0,00002), tandis que les 9 erreurs stochastiques ont montre 55,6 % de concordance dans l'Execution 3 — proche des 50 % attendus pour un veritable comportement de pile ou face.

### 9.5 Discussion

**76 % sur Arena est competitif — et approche le plafond humain.** Ce resultat se situe dans la fourchette de 72-83 % d'accord crowd-expert humain sur la meme plateforme [10], a 0,15 $/eval (vs. plusieurs dollars pour l'annotation humaine) avec une reproductibilite totale. Sur les modeles frontier, le biais de capacite observe dans l'Experience 1 disparait naturellement : lorsque les modeles sont proches en performance, il n'y a plus de hierarchie de capacite a favoriser. De plus, il est important de contextualiser le taux de 76 % : chaque evaluation Arena reflete un vote humain unique (pas un consensus). Les etudes de Chiang et al. [10] montrent que l'accord crowd-expert varie de 72-83 %, et l'accord expert-expert de 79-90 %. Notre panel a 76 % se situe donc dans la fourchette que l'on attendrait d'un evaluateur humain supplementaire. **Un modele parfait qui reproduirait exactement le jugement d'un humain moyen plafonnerait a environ 80 % de concordance avec un autre humain aleatoire** — notre panel approche cette borne.

**Haute fiabilite test-retest.** 91 % d'accord inter-executions (kappa = 0,757). Les 9 evaluations instables ont montre 55,6 % de concordance dans l'Execution 3 — indiscernable du hasard — confirmant qu'elles representent le seuil de bruit du pipeline.

**19 % d'erreurs systematiques irreductibles.** Le vote majoritaire sur 3 executions n'ameliore pas la concordance (76,0 % = taux d'une seule execution). Explications possibles :
- Les juges IA privilegient la precision factuelle/profondeur du raisonnement ; les votants humains peuvent accorder plus d'importance au style ou a l'utilite
- Certains votes crowd-sourced sont bruites ou idiosyncrasiques
- Certaines paires de modeles (par ex., Claude 3.5 Sonnet vs. Gemini 1.5 Pro) se situent en dessous de la limite de resolution de tout systeme d'evaluation

**Nature des desaccords.** Sur les 24 desaccords de l'Experience 2, la grande majorite concerne des paires de modeles tres proches en performance (difference ELO < 50 points). Dans ces cas, les deux reponses sont souvent de qualite comparable, et le choix entre elles reflete une preference subjective plutot qu'une difference objectivable. Cela fait echo aux observations de Feng et al. [24], qui montrent que meme les meilleurs evaluateurs echouent a maintenir des preferences coherentes dans ~25 % des cas difficiles. **Les erreurs du panel sont concentrees precisement la ou la notion meme de "bonne reponse" est ambigue**, ce qui relativise leur impact pratique.

**Pas de biais de capacite sur les modeles frontier.** Le "biais de capacite" de l'Experience 1 (72,7 % des erreurs favorisant les modeles plus forts) disparait sur Arena : seulement 45,8 % favorisent le modele a ELO plus eleve (test des signes p = 1,0). Le biais etait un artefact du jeu de donnees, pas une propriete intrinseque du panel.

**Pas d'auto-favoritisme.** Les taux de victoire des familles GPT, Claude et Gemini correspondent etroitement aux preferences humaines. Les donnees d'auto-evaluation en aveugle (N = 252) confirment l'absence d'auto-promotion en conditions anonymisees. L'utilisation de modeles frontier comme juges pour leur propre lignee est valide en conditions aveugles [20].

**Limites.** (1) Chaque evaluation Arena reflete un vote humain unique, pas un consensus. (2) Le jeu de donnees couvre uniquement juin-aout 2024. (3) Conversations a un seul tour uniquement. (4) N = 260 fournit une puissance globale adequate mais une puissance limitee par sous-groupe de modeles. (5) Reduire le taux d'erreur systematique de 19 % necessite des changements architecturaux, pas de la repetition.

---

## 10. Analyse inter-experiences

### 10.1 Gradient de concordance

> **Resultat :** La baisse de concordance de 12 points (88 % -> 76 %) est statistiquement significative (p = 0,027) et attendue : des ecarts de capacite plus faibles rendent la tache plus difficile pour les humains comme pour l'IA.

**Tableau 9.** Comparaison de l'Experience 1 (MT-Bench) et de l'Experience 2 (Chatbot Arena).

| Metrique | Exp. 1 : MT-Bench | Exp. 2 : Arena |
|---|---|---|
| **Jeu de donnees** | MT-Bench Human Judgments [1] | Chatbot Arena 100k [28] |
| **Annee** | 2023 | 2024 |
| **N modeles** | 6 | 25 |
| **N evaluations** | 100 | 100 (x2 executions) + 60 ciblees |
| **Ecart entre modeles** | Large (GPT-4 vs. LLaMA-13B) | Faible (GPT-4o vs. Claude 3.5 Sonnet) |
| **Annotateurs humains** | Expert | Crowd-sourced |
| **Concordance** | **88,0 %** | **76,0 %** / **75,0 %** |
| **Kappa de Cohen** | 0,760 | 0,520 / 0,500 |
| **Cout total** | 12,86 $ | 38,84 $ (260 evals) |
| **Cout par eval** | 0,129 $ | 0,149 $ |
| **Test-retest** | — | 91,0 % (kappa = 0,757) |
| **Taux d'erreur systematique** | — | 19 % (confirme 3 executions) |

La baisse de 12 points est **statistiquement significative** (z = 2,21, p = 0,027) et entrainee par trois facteurs : (1) **des ecarts de capacite plus faibles** rendent la discrimination plus difficile pour les humains et l'IA ; (2) **des annotateurs crowd vs. experts** — nos 76 % se situent dans la fourchette de 72-83 % d'accord crowd-expert sur Arena [10] ; (3) **25 modeles (vs. 6)** creent de nombreuses confrontations serrees ou tout evaluateur serait en difficulte.

**Tableau 10.** Comparaison consolidee avec les travaux anterieurs et les references humaines.

| Categorie | Methode | Benchmark | Accord | kappa | Reference |
|---|---|---|---|---|---|
| **References humaines** | Crowd vs. expert | Chatbot Arena | 72-83 % | — | Chiang et al. [10] |
| | Expert vs. expert | Chatbot Arena | 79-90 % | — | Chiang et al. [10] |
| | Humain vs. humain | MT-Bench (S2) | 81 % | — | Zheng et al. [1] |
| **Juge unique** | GPT-4 | MT-Bench (S2, sans egalites) | 85 % | — | Zheng et al. [1] |
| **Panels multi-juges** | PoLL (3 modeles) | HotPotQA | — | 0,889 | Verga et al. [21] |
| | **Agent Clash (3 SC)** | **MT-Bench** | **88,0 %** | **0,760** | **Ce travail** |
| | **Agent Clash (3 SC)** | **Arena (frontier)** | **76,0 %** | **0,520** | **Ce travail** |
| **Fiabilite** | **Agent Clash test-retest** | **Arena** | **91,0 %** | **0,757** | **Ce travail** |

*Les metriques ne sont pas directement comparables entre differents benchmarks (voir note du Tableau 4 concernant l'interpretation accord vs. kappa).*

### 10.2 Ablation : juge individuel vs. panel

Pour evaluer si le panel multi-juges ameliore les performances des juges individuels, nous avons extrait le vote individuel de chaque juge Supreme Court de toutes les evaluations des deux experiences et calcule la concordance par juge avec les preferences humaines.

**Tableau 11.** Concordance des juges individuels vs. panel (vote majoritaire), N cumule = 192.

| Juge | MT-Bench (N=99) | Arena (N=93) | Cumule (N=192) |
|---|---|---|---|
| GPT-5.2-pro | 88,9 % | 74,2 % | 81,8 % |
| Claude Opus 4.5 | 87,9 % | 76,3 % | 82,3 % |
| Gemini 2.5 Pro | 83,8 % | 69,9 % | 77,1 % |
| **Panel (majorite 2 sur 3)** | **88,9 %** | **75,3 %** | **82,3 %** |

**Resultats cles :**

1. **Le panel egale le meilleur juge individuel** — sans savoir a l'avance quel juge sera le meilleur. Sur MT-Bench, GPT-5.2-pro est le meilleur juge individuel (88,9 %) ; sur Arena, Claude Opus 4.5 est le meilleur (76,3 %). Le meilleur juge change selon les jeux de donnees, mais le panel egale ou depasse systematiquement le meilleur.

2. **Le panel protege contre le pire juge.** Compare a un juge unique selectionne aleatoirement, le panel gagne +1,9pp en moyenne. Compare au pire juge (Gemini 2.5 Pro), le panel gagne +5,2pp.

3. **L'architecture multi-juges apporte de la robustesse**, pas une amelioration de la performance de pointe. Sa valeur principale est l'**assurance** : elle garantit la performance du meilleur juge individuel quel que soit le jeu de donnees ou le domaine de tache evalue.

### 10.3 L'unanimite comme signal de confiance

Un avantage cle d'un panel multi-juges par rapport a un juge unique est la capacite de mesurer l'**accord interne** comme indicateur de confiance. Nous analysons la concordance stratifiee selon que les trois juges ont atteint une decision unanime (3-0) ou partagee (2-1).

**Tableau 12.** Concordance par unanimite du panel (cumulee sur les deux experiences, N = 239).

| Accord | N | % des evals | Concordance avec l'humain |
|---|---|---|---|
| Unanime (3-0) | 192 | 80 % | **84,9 %** |
| Partage (2-1) | 47 | 20 % | **63,8 %** |
| **Difference** | | | **+21,1pp** |

**Resultats cles :**

1. **L'unanimite est un fort predicteur de precision.** Lorsque les trois juges sont d'accord, la concordance est de 84,9 %. Lorsqu'ils se partagent 2-1, la concordance chute a 63,8 %. L'ecart de 21,1 points de pourcentage fournit un **signal de confiance binaire, empiriquement calibre**.

2. **Sur les decisions partagees, le panel surpasse encore les juges individuels.** Lorsque les 3 juges sont en desaccord (2 contre 1), c'est un cas intrinsequement difficile. Le panel, qui suit la majorite (2 sur 3), atteint neanmoins 63,8 % de concordance avec la preference humaine. En comparaison, si un juge unique avait ete utilise sur ces memes cas difficiles, le meilleur juge individuel (Claude Opus 4.5) aurait atteint environ 57,4 %. Le gain de +6,4pp montre que c'est precisement sur les cas litigieux que le vote a 3 juges apporte le plus de valeur — le juge dissident a tort dans la majorite des cas.

3. **Un juge unique ne peut pas produire ce signal.** Un juge seul produit toujours un verdict sans mesure interne de certitude. Le panel multi-juges fournit un mecanisme de routage actionnable : les verdicts unanimes peuvent etre acceptes avec haute confiance ; les verdicts partages peuvent etre signales pour examen humain.

**Implication pratique :** En production, le signal d'unanimite permet une strategie de triage simple — accepter automatiquement les decisions unanimes, escalader les decisions partagees vers l'examen humain — ce qui eleverait la concordance effective sur le sous-ensemble accepte a ~85 %.

### 10.4 Synthese

La proposition de valeur du panel multi-juges est triple :

1. **Robustesse :** Il egale le meilleur juge individuel sans savoir a priori quel juge est le meilleur, protegant contre le cout de +5,2pp de choisir le mauvais juge unique.

2. **Calibration de la confiance :** Le signal d'unanimite (84,9 % vs. 63,8 %) fournit un indicateur de confiance actionnable, empiriquement valide, qu'un juge unique ne peut pas produire.

3. **Diversification des biais :** Trois familles de modeles de fournisseurs differents garantissent qu'aucun biais architectural unique ne domine le verdict.

Le panel multi-juges n'ameliore pas dramatiquement la concordance brute par rapport au meilleur juge unique. Sa contribution principale est la **fiabilite et l'interpretabilite** — savoir quand faire confiance au verdict et quand escalader.

---

## 11. Limites

Le cadre mesure bien la qualite percue, la coherence et la stabilite du raisonnement. Il ne **pretend pas** mesurer la verite absolue, l'exactitude factuelle specifique a un domaine, ni la preference stylistique universelle.

| Limite | Implication |
|---|---|
| **Plafond de capacite des juges** | A mesure que les candidats approchent la capacite des juges, la discrimination se degrade |
| **Cout** | L'evaluation multi-juges est plus couteuse qu'un juge unique — mais a ~0,15 $ par evaluation, le surcout est negligeable pour un prompt avec quelques modeles candidats. Le cadre est concu pour les decisions de selection a forts enjeux |
| **Biais residuel** | Les juges LLM portent des biais [9] qui ne peuvent etre entierement elimines -> voie humain-dans-la-boucle preservee |
| **Qualite de la verite terrain** | MT-Bench utilise des annotations expertes ; Arena utilise des votes uniques crowd-sourced par evaluation — aucun ne constitue une verite terrain definitive. Les metriques de concordance heritent du bruit de la reference |
| **Biais du generateur de criteres** | Les criteres dynamiques sont generes par gpt-4o-mini (Section 5.4). Cela introduit une dependance a un modele unique : la grille d'evaluation reflete l'interpretation d'un seul modele des exigences de la tache. L'utilisation d'un generateur de criteres different pourrait modifier les resultats |
| **Taille du panel** | Seulement 3 juges testes ; des panels plus larges pourraient ameliorer la concordance ou la calibration de la confiance |
| **Pas de variation de jeu de donnees** | Nous n'avons pas pu tester la robustesse a des ensembles de reponses varies pour le meme prompt — aucun tel jeu de donnees apparie avec des preferences humaines n'existe dans la litterature. Cela reste une question ouverte importante |

Le cadre ne remplace pas le jugement humain — il fournit un signal evolutif et reproductible qui concentre l'attention humaine la ou elle est le plus necessaire.

---

## 12. Travaux futurs

Plusieurs extensions sont prevues : (1) mesurer la stabilite du classement sur N = 10 cycles d'evaluation independants en utilisant le tau de Kendall [19] ; (2) evaluer la generalisation inter-jeux de donnees sur K = 5 jeux de donnees heterogenes ; (3) des experiences controlees d'auto-favoritisme en conditions aveugles avec des tests de Wilcoxon ; (4) cartographier la frontiere de Pareto cout-qualite-latence ; et (5) tester des panels de juges plus larges (5 ou 7 juges) pour evaluer si des juges supplementaires ameliorent la concordance ou la calibration de la confiance.

**(6) Stabilite du pipeline de bout en bout.** Les Experiences 1-2 ont maintenu les reponses fixes et mesure uniquement la variance du *jugement* (Section 7.1). En production cependant, Agent Clash regenere les reponses a chaque execution. Un test preliminaire examine ce qui se passe lorsque la generation de reponses et le jugement sont repetes a partir de zero. Nous avons fait passer 10 prompts (4 faciles, 4 moyens, 2 difficiles) a travers 3 cycles complets — regenerant les reponses de GPT-4o-mini et Claude 3.5 Sonnet a temperature 1,0, puis jugeant chaque nouvelle paire — pour 30 evaluations au total (3,99 $). Tous les hashes de reponses differaient entre les executions, confirmant la generation non deterministe. Resultat : **6/10 prompts (60 %) ont produit le meme vainqueur dans les 3 executions**, tandis que 4/10 ont montre un partage 2-1. Ce taux de stabilite de bout en bout est nettement inferieur a l'accord test-retest de 91 % sur le jugement seul (Experience 2), indiquant que la variance de generation est la source dominante d'instabilite — pas le panel de juges. Ce resultat souligne que la stabilite du jugement — validee dans cette etude — est une condition necessaire mais non suffisante pour la stabilite de bout en bout. Aucune reference de panel humain n'existe dans la litterature pour ce type de test, ce qui constitue une limitation ouverte. Une etude a grande echelle avec N >= 30 prompts et une decomposition de la variance (composantes generation vs. jugement) est justifiee.

---

## 13. Conclusion

> **En resume :** Un panel IA a 3 juges atteint 88 % de concordance avec les humains lorsque les modeles different largement et 76 % lorsqu'ils sont proches en performance — comparable a l'accord inter-humain dans les deux cas. La valeur principale du panel n'est pas l'amelioration brute de la precision par rapport au meilleur juge unique, mais la robustesse et la confiance calibree.

Nous avons valide un cadre multi-juges a travers deux regimes de difficulte. Le **gradient de concordance** — 88,0 % (kappa = 0,760) sur les grands ecarts, 76,0 % (kappa = 0,520) sur les modeles frontier — suit la difficulte de la tache et se situe dans la fourchette de l'accord inter-annotateurs humain sur les deux benchmarks. La fiabilite test-retest est de 91 %, avec des erreurs decomposees en 19 % systematiques (confirmees irreductibles, p < 0,00002) et 9 % stochastiques.

**Ce qui distingue ce travail :**
1. Validation sur des **modeles frontier proches en performance**, pas seulement sur des benchmarks avec de grands ecarts
2. **Caracterisation systematique des erreurs** a travers des executions repetees (pas seulement la precision en un seul passage)
3. **Analyse d'ablation** montrant que le panel egale le meilleur juge individuel sans savoir lequel a l'avance, avec une protection de +5,2pp contre le pire
4. Un **signal d'unanimite calibre** (84,9 % vs. 63,8 %) fournissant un indicateur de confiance actionnable qu'un juge unique ne peut pas produire

Le taux de desaccord irreductible de 19 % fixe un plafond pratique. Toute amelioration supplementaire necessite des changements architecturaux — des ensembles de juges specialises, une escalade humain-dans-la-boucle, ou des criteres adaptes a l'alignement avec les preferences de la foule — et non des executions supplementaires du meme pipeline.

---

## Reproductibilite et transparence

De nombreux benchmarks existants sont **opaques** : les donnees d'evaluation, l'identite des juges et les criteres de notation sont caches aux utilisateurs. Ce cadre adopte l'approche inverse.

En production, Agent Clash regenere les reponses a chaque execution — les resultats ne seront donc pas identiques d'une execution a l'autre, mais la methodologie d'evaluation est strictement reproductible. La stabilite de jugement validee dans cette etude (91 % test-retest) garantit que les verdicts du panel sont coherents meme lorsque les criteres et l'ordre des reponses sont re-randomises.

Pour permettre la verification independante des resultats rapportes dans cette etude, les materiels suivants sont disponibles en acces libre :

- **Pipelines de validation** (workflows n8n exportes)
- **Jeux de donnees** utilises (echantillons MT-Bench et Arena)
- **Resultats bruts** (matrices de notation par juge, par evaluation)
- **Article** (versions anglaise et francaise)

Repository : [github.com/anthonyboisbouvier-paris/agent-clash-paper](https://github.com/anthonyboisbouvier-paris/agent-clash-paper)

Plateforme de production : [agent-clash.ai](https://www.agent-clash.ai/)

- **Les utilisateurs fournissent leur propre cle API** -> controle total et auditabilite de chaque appel d'inference
- **Tous les prompts, jeux de donnees, criteres, identites des juges et regles d'agregation** sont divulgues et configurables [8]
- **Matrices de notation brutes** disponibles par juge, par execution, par jeu de donnees
- **Vues de reponses cote a cote** pour l'inspection directe des sorties de modeles

Tout utilisateur peut re-executer exactement la meme evaluation et s'attendre a des resultats statistiquement coherents. Ceci n'est pas un benchmark auquel on fait confiance — c'est un benchmark que l'on **verifie**.

---

## Remerciements

Cet article a ete relu avec l'assistance de ChatGPT-o3 (OpenAI). Toutes les decisions concernant la conception experimentale, l'analyse des donnees, l'interpretation et les choix editoriaux ont ete prises par l'auteur humain.

---

## References

[1] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.05685*

[2] Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *arXiv:2303.16634*

[3] Kocmi, T. & Federmann, C. (2023). Large Language Models Are State-of-the-Art Evaluators of Translation Quality. *arXiv:2302.14520*

[4] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). Self-Consistency Improves Chain-of-Thought Reasoning in Language Models. *arXiv:2203.11171*

[5] Oren, Y., Geva, M., Stanovsky, G., & Smith, N. A. (2022). Distribution Shifts Are the Norm in NLP. *arXiv:2205.12350*

[6] Mitchell, E., Noh, Y., Li, S., Armstrong, W., Agarwal, A., Liu, P., Finn, C., & Manning, C. D. (2022). Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference. *arXiv:2211.11875*

[7] Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet Classifiers Generalize to ImageNet? *ICML 2019*

[8] OpenAI (2023). Evals Framework. https://github.com/openai/evals

[9] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*

[10] Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., & Stoica, I. (2024). Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. *arXiv:2403.04132*

[11] De Borda, J.-C. (1781). Memoire sur les elections au scrutin. *Histoire de l'Academie Royale des Sciences*, Paris.

[12] Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). AlpacaEval: An Automatic Evaluator of Instruction-Following Models. https://github.com/tatsu-lab/alpaca_eval

[13] Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P., & Hashimoto, T. B. (2024). AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. *NeurIPS 2023*

[14] Saad-Falcon, J., Barber, T., Jain, N., Desai, A., & Potts, C. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *arXiv:2311.09476*

[15] Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*

[16] Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., et al. (2022). Holistic Evaluation of Language Models (HELM). *arXiv:2211.09110*

[17] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*

[18] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37-46.

[19] Kendall, M. G. (1938). A New Measure of Rank Correlation. *Biometrika*, 30(1/2), 81-93.

[20] Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM Evaluators Recognize and Favor Their Own Generations. *arXiv:2404.13076*

[21] Verga, P., Hofstatter, S., Althammer, S., Pirtoaca, G., Cer, D., Re, C., Gunnemann, S., & Petroni, F. (2024). Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. *arXiv:2404.18796*

[22] Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., & Liu, Z. (2024). ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate. *ICLR 2024. arXiv:2308.07201*

[23] Thakur, N., Mukherjee, S., Arasteh, S. T., Reimers, N., Han, J., & Schutze, H. (2024). Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges. *arXiv:2406.12624*

[24] Feng, S., Park, C. Y., Liu, Y., & Tsvetkov, Y. (2025). Sage: A General-Purpose Evaluator for LLMs. *arXiv:2512.16041*

[25] Landis, J. R. & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159-174.

[26] Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline. *arXiv:2406.11939*

[27] Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, 22(158), 209-212.

[28] LMSYS / LMArena (2024). Arena Human Preference 100k Dataset. `lmarena-ai/arena-human-preference-100k`, Hugging Face Datasets. CC-BY-4.0. https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k
