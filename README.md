# Générateur Kanban

Ce dépôt inclut une automatisation légère qui transforme chaque commit git en carte Kanban. La mécanique met à jour deux fichiers :

- `kanban_cards.json` – historique append-only des cartes générées
- `kanban_export.csv` – export compatible Trello régénéré à chaque commit

## Installation

Lancez une fois le script d’installation pour activer les hooks :

```bash
python tools/install_hooks.py
```

Le script configure `core.hooksPath` sur `.githooks` et s’assure que le hook post-commit est exécutable.

## Utilisation

1. Réalisez vos modifications locales.
2. Indexez les fichiers : `git add .`.
3. Lancez l’assistant de commit : `python tools/commit.py` (le script pose les questions type/scope/difficulté puis construit le message).
4. Après un commit réussi, le hook `post-commit` exécute silencieusement `tools/kanban_generate.py`.

Le générateur lit le dernier commit, détermine les labels et éléments de DoD selon le type, puis produit les fichiers `kanban_cards.json` et `kanban_export.csv`. Chaque carte inclut :

- Le hash du commit
- Le type / scope / sujet analysé
- Le corps du commit (s’il existe)
- La liste des fichiers modifiés

Vous pouvez versionner ces fichiers dans le même commit ou dans un commit de suivi afin de garder votre board synchronisé avec l’historique git.
