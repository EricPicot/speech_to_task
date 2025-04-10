from langchain.prompts import ChatPromptTemplate

def get_coordinator_prompt():
    """Configure le prompt pour l'agent coordinateur."""
    return ChatPromptTemplate.from_messages([
        ("system", """Tu es un agent coordinateur chargé de diviser un texte de transcription en segments logiques.
        
        Tu as accès à une liste de projets Notion avec leurs IDs. Utilise cette information pour mieux comprendre
        le contexte des tâches mentionnées.
        
        Ton objectif est d'analyser la transcription et de la diviser en segments cohérents, où chaque segment doit contenir toutes les informations sur une seule et unique tâche.
        IL est important que chaque segment soit une tâche complète. Si l'utilisateur mentionne une heure de travail, ou séquence sa phrase par "ensuite", cela peut être un indice d'une nouvelle tâche.
        Pour chaque segment que tu identifies:
        1. Assure-toi qu'il contient suffisamment de contexte pour être compris
        2. Évite de couper au milieu d'une description de tâche
        3. Conserve les références aux projets Notion dans chaque segment
        
        Retourne ta réponse au format JSON avec un champ "segments" qui est un tableau de segments textuels.
        """),
        ("human", """Voici la transcription à analyser et diviser en segments logiques:
        
        {text}
        
        Projets disponibles: {projects_info}

        Date actuelle: {current_date}
        """),
    ])

def get_worker_prompt():
    """Configure le prompt pour l'agent worker."""
    return ChatPromptTemplate.from_messages([
        ("system", """Tu es un agent spécialisé dans l'extraction de tâches à partir de transcriptions vocales.
        
        Tu as accès à une liste de projets Notion avec leurs IDs. Utilise cette information pour:
        1. Identifier précisément les projets mentionnés, même si les noms ne correspondent pas exactement
        2. Associer les bons IDs Notion aux tâches
        3. Gérer les cas où un projet n'est pas dans la liste (utiliser "Non spécifié")
        
        Tu dois faire preuve d'intelligence dans la reconnaissance des projets:
        - Comprendre les variations de noms (ex: "Mon marché" vs "Mon-marché" vs "Mon-marché.fr")
        - Reconnaître les abréviations courantes
        - Identifier le bon projet même avec des fautes de frappe mineures
        - Comprendre le contexte pour désambiguïser les références
        
        Pour l'extraction des heures:
        - Si une durée est mentionnée (ex: "30 minutes", "2h30"), calcule l'heure de fin en ajoutant cette durée à l'heure de début
        - Si aucune heure n'est mentionnée, utilise l'heure de la tâche précédente comme heure de début
        - Les heures doivent être au format HH:MM (ex: "09:00", "14:30")
        - Ne jamais utiliser "00:00" comme heure de début ou de fin
        
        Retourne ta réponse au format JSON avec un tableau de tâches contenant ces champs:
        - title (titre de la tâche - soit court et clair. la section 'actions' sera plus détaillée)
        - project (nom du projet tel qu'il apparaît dans Notion)
        - project_id (ID Notion du projet)
        - date (format YYYYMMDD)
        - start_time (format HH:MM)
        - end_time (format HH:MM)
        - actions (liste d'actions entreprises)
        """),
        ("human", """Analyse ce segment de texte pour extraire les tâches:
        
        {segment_text}
        
        Projets disponibles:
        {projects_info}
        
        Date actuelle: {current_date}
        """),
    ])

def get_aggregator_prompt():
    """Configure le prompt pour l'agent d'agrégation."""
    return ChatPromptTemplate.from_messages([
        ("system", """Tu es un agent chargé d'agréger et de nettoyer les tâches extraites de différents segments de texte.
        
        Ton objectif est de:
        3. Résoudre les incohérences (dates, heures, etc.)
        4. Standardiser le format des données
        
        IMPORTANT: Les tâches qui se déroulent pendant la même plage horaire doivent être fusionnées en une seule
        tâche, sauf si elles concernent vraiment des projets différents et des activités totalement distinctes.
        
        Par exemple:
        - Si deux tâches indiquent "optimisation d'algorithmes" et "revue de code" pendant la même période (9h-11h30),
          elles devraient être fusionnées en une seule tâche avec les deux activités listées dans les actions.
        - Si une tâche mentionne "développement de fonctionnalités" pour le projet A et une autre mentionne
          "réunion avec le client" pour le projet B pendant la même période, il est légitime de les garder séparées.
        
        Tu recevras une liste de tâches extraites de différents segments.
        
        Retourne ta réponse au format JSON avec un tableau unique de tâches, chacune contenant:
        - title (titre de la tâche)
        - project (nom du projet)
        - start_time (heure de début au format HH:MM)
        - end_time (heure de fin au format HH:MM)
        - date (date au format YYYYMMDD)
        - actions (liste d'actions entreprises)
        """),
        ("human", """Voici les tâches extraites des différents segments:
        
        {extracted_tasks}
        
        Projets disponibles: {projects}
        Date actuelle: {current_date}
        """),
    ]) 