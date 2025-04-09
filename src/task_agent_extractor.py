"""
Task Agent Extractor

Un système qui utilise LangGraph pour extraire des tâches à partir de transcriptions vocales de manière parallélisée.
Il divise la transcription en segments, extrait les tâches de chaque segment en parallèle, puis agrège les résultats.

Author: Eric Picot
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, TypedDict, Annotated, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator

# Importations LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Importations LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send

load_dotenv()

class TaskWithTime(BaseModel):
    """Modèle pour représenter une tâche extraite avec les heures de début et de fin."""
    description: str = Field(description="Description détaillée de la tâche réalisée")
    project: str = Field(description="Projet associé à la tâche")
    start_time: str = Field(description="Heure de début de la tâche (format HH:MM)")
    end_time: str = Field(description="Heure de fin de la tâche (format HH:MM)")
    date: str = Field(description="Date à laquelle la tâche a été réalisée (format YYYY-MM-DD)")
    actions: List[str] = Field(description="Liste des actions spécifiques réalisées dans le cadre de cette tâche")
    
    @property
    def duration_minutes(self) -> int:
        """Calcule la durée de la tâche en minutes."""
        try:
            start = datetime.strptime(self.start_time, "%H:%M")
            end = datetime.strptime(self.end_time, "%H:%M")
            
            # Gérer le cas où la tâche se termine le jour suivant
            if end < start:
                end += timedelta(days=1)
                
            delta = end - start
            return int(delta.total_seconds() / 60)
        except Exception:
            # En cas d'erreur, retourner 0
            return 0

def resolve_date_reference(query: str) -> str:
    """
    Résout une référence temporelle relative comme 'hier', 'avant-hier', etc. en une date absolue.
    
    Args:
        query: La référence temporelle à résoudre (ex: 'hier', 'aujourd'hui', 'lundi dernier')
        
    Returns:
        La date absolue au format YYYY-MM-DD
    """
    today = datetime.now()
    
    # Normaliser le texte de la requête
    query_lower = query.lower()
    
    # Références simples
    if any(term in query_lower for term in ["aujourd'hui", "maintenant"]):
        return today.strftime("%Y-%m-%d")
    elif "hier" in query_lower:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif any(term in query_lower for term in ["avant-hier", "avant hier"]):
        return (today - timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Jours de la semaine
    days_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    
    # Rechercher un jour de la semaine dans la requête
    for i, day in enumerate(days_fr):
        if day in query_lower:
            # Convertir l'indice du jour (0 = lundi dans notre liste, 0 = lundi dans Python)
            target_weekday = i
            current_weekday = today.weekday()
            
            # Calculer la différence de jours
            if any(term in query_lower for term in ["dernier", "passé", "précédent"]):
                # C'est un jour de la semaine dernière
                days_diff = current_weekday - target_weekday
                if days_diff <= 0:
                    days_diff += 7
            else:
                # C'est un jour de cette semaine
                days_diff = current_weekday - target_weekday
                if days_diff < 0:
                    # Le jour est plus tard dans la semaine
                    days_diff = 0
            
            return (today - timedelta(days=days_diff)).strftime("%Y-%m-%d")
    
    # Par défaut, retourner aujourd'hui
    return today.strftime("%Y-%m-%d")

# Segment pour le traitement parallèle
class Segment(BaseModel):
    """Segment de transcription à traiter par un worker."""
    id: int = Field(description="Identifiant unique du segment")
    text: str = Field(description="Texte du segment à analyser")

# État du graphe principal
class TaskExtractionState(TypedDict):
    """
    État global du graphe d'extraction de tâches.
    Contient les entrées, l'état interne et la sortie du processus d'extraction.
    """
    # Entrées
    transcript: str                           # Transcription complète
    projects: List[str]                       # Liste des projets disponibles
    current_date: str                         # Date actuelle
    
    # État interne
    segments: List[Segment]                   # Segments de la transcription
    extracted_tasks: Annotated[List[Dict], operator.add]  # Tâches extraites par tous les workers
    
    # Sortie finale
    final_tasks: List[TaskWithTime]           # Tâches agrégées et formatées

# État pour les workers
class WorkerState(TypedDict):
    """
    État pour un worker traitant un segment spécifique.
    Contient les informations nécessaires pour extraire les tâches d'un segment.
    """
    segment: Segment                          # Segment à traiter
    projects: List[str]                       # Projets disponibles
    current_date: str                         # Date actuelle
    extracted_tasks: Annotated[List[Dict], operator.add]  # Pour collecter les tâches

class TaskAgentExtractor:
    """
    Extracteur de tâches basé sur LangGraph avec traitement parallèle.
    
    Cette classe utilise LangGraph pour orchestrer l'extraction de tâches à partir de transcriptions
    vocales en utilisant un traitement parallèle. Le processus se déroule en trois étapes principales :
    
    1. Un agent coordinateur divise la transcription en segments logiques
    2. Des agents travailleurs extraient les tâches de chaque segment en parallèle
    3. Un agent agrégateur combine et nettoie les tâches extraites
    
    Attributes:
        projects (List[str]): Liste des projets disponibles pour la classification des tâches
        llm (ChatOpenAI): Instance du modèle de langage configuré pour l'extraction
        graph (StateGraph): Graphe LangGraph compilé pour l'orchestration
    """
    
    def __init__(self, projects=None, api_key=None, model="google/gemini-2.0-flash-001"):
        """
        Initialise l'agent d'extraction de tâches avec LangGraph.
        
        Args:
            projects (List[str], optional): Liste des projets disponibles. Par défaut, liste vide.
            api_key (str, optional): Clé API pour OpenRouter. Par défaut, utilise OPENROUTER_API_KEY de l'environnement.
            model (str, optional): Modèle à utiliser pour l'extraction. Par défaut "google/gemini-2.0-flash-001".
        
        Raises:
            ValueError: Si aucune clé API n'est trouvée.
        """
        self.projects = projects or []
        
        # Récupérer la clé API depuis les arguments ou les variables d'environnement
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Aucune clé API trouvée. Fournissez-la en argument ou définissez la variable d'environnement OPENROUTER_API_KEY."
            )
        
        # Configurer le LLM avec OpenRouter
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Speech to Task Tracker"
            }
        )
        
        # Configurer les différents prompts pour les agents
        self._setup_prompts()
        
        # Construire le graphe LangGraph
        self._build_graph()
        
    def _setup_prompts(self):
        """Configure les prompts pour les différents agents."""
        
        # Prompt pour l'agent coordinateur
        self.coordinator_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un agent coordinateur chargé de diviser un texte de transcription en segments logiques.
            
            Ton objectif est d'analyser la transcription et de la diviser en segments cohérents, où chaque segment contient
            potentiellement des informations sur une ou plusieurs tâches distinctes. Tu dois identifier où il est logique
            de séparer le texte pour un traitement parallèle efficace.
            
            Pour chaque segment que tu identifies:
            1. Assure-toi qu'il contient suffisamment de contexte pour être compris de manière autonome
            2. Évite de couper au milieu d'une description de tâche
            3. Priorise les ruptures naturelles dans le discours
            
            Retourne ta réponse au format JSON avec un champ "segments" qui est un tableau de segments textuels.
            
            Si le texte est court et ne nécessite pas de division, retourne-le comme un seul segment.
            """),
            ("human", """Voici la transcription à analyser et diviser en segments logiques:
            
            {transcript}
            
            Projets disponibles: {projects}
            Date actuelle: {current_date}
            """),
        ])
        
        # Prompt pour les agents travailleurs (extraction de tâches)
        self.worker_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un agent spécialisé dans l'extraction de tâches à partir de transcriptions vocales.
            
            Ton objectif est d'identifier toutes les tâches mentionnées dans le segment de texte et d'extraire :
            1. Une description claire de chaque tâche
            2. Le projet associé à chaque tâche (si non spécifié, indiquer "Non spécifié")
            3. L'heure de début et de fin de chaque tâche (au format HH:MM)
            4. La date à laquelle chaque tâche a été réalisée (au format YYYY-MM-DD)
            5. Les actions spécifiques réalisées
            
            IMPORTANT: Considère toutes les activités réalisées dans la même plage horaire comme une SEULE tâche,
            même si elles concernent plusieurs projets. Si plusieurs activités sont mentionnées pour la même plage horaire 
            mais concernent clairement des projets différents, alors tu peux créer des tâches séparées.
            
            Par exemple:
            - "De 9h à 11h30, j'ai optimisé des algorithmes et fait une revue de code" devrait être considéré comme
              une SEULE tâche avec deux actions: "optimisation d'algorithmes" et "revue de code".
            - "De 9h à 11h30, j'ai optimisé des algorithmes et eu une réunion avec l'équipe marketing" pourrait être
              deux tâches distinctes si elles concernent des projets clairement différents.
            
            Il se peut que l'heure de début et de fin ne soient pas précisées dans le texte mais que seule la durée soit mentionnée.
            Dans ce cas, invente une heure de début et de fin théorique en fonction de la durée de la tâche.
            De même, si la date n'est pas précisée, utilise la date actuelle fournie.
            
            Si aucun projet n'est mentionné, utilise le projet "Non spécifié".
            
            Ton objectif est de comprendre les intentions de l'utilisateur même si le texte n'est pas parfaitement structuré.
            
            Retourne ta réponse au format JSON avec un tableau de tâches contenant ces champs:
            - description (description de la tâche)
            - project (nom du projet)
            - start_time (heure de début au format HH:MM)
            - end_time (heure de fin au format HH:MM)
            - date (date au format YYYY-MM-DD)
            - actions (liste d'actions entreprises)
            """),
            ("human", """Analyse ce segment de texte pour extraire les tâches:
            
            {segment_text}
            
            Projets disponibles: {projects}
            Date actuelle: {current_date}
            """),
        ])
        
        # Prompt pour l'agent d'agrégation
        self.aggregator_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un agent chargé d'agréger et de nettoyer les tâches extraites de différents segments de texte.
            
            Ton objectif est de:
            1. Fusionner les tâches similaires ou identiques
            2. Éliminer les doublons
            3. Résoudre les incohérences (dates, heures, etc.)
            4. Standardiser le format des données
            5. Compléter les informations manquantes si possible
            
            IMPORTANT: Les tâches qui se déroulent pendant la même plage horaire doivent être fusionnées en une seule
            tâche, sauf si elles concernent vraiment des projets différents et des activités totalement distinctes.
            
            Par exemple:
            - Si deux tâches indiquent "optimisation d'algorithmes" et "revue de code" pendant la même période (9h-11h30),
              elles devraient être fusionnées en une seule tâche avec les deux activités listées dans les actions.
            - Si une tâche mentionne "développement de fonctionnalités" pour le projet A et une autre mentionne
              "réunion avec le client" pour le projet B pendant la même période, il est légitime de les garder séparées.
            
            Tu recevras une liste de tâches extraites de différents segments.
            
            Retourne ta réponse au format JSON avec un tableau unique de tâches, chacune contenant:
            - description (description de la tâche)
            - project (nom du projet)
            - start_time (heure de début au format HH:MM)
            - end_time (heure de fin au format HH:MM)
            - date (date au format YYYY-MM-DD)
            - actions (liste d'actions entreprises)
            """),
            ("human", """Voici les tâches extraites des différents segments:
            
            {extracted_tasks}
            
            Projets disponibles: {projects}
            Date actuelle: {current_date}
            """),
        ])
    
    def _build_graph(self):
        """Construit le graphe LangGraph pour l'extraction de tâches en parallèle."""
        
        # Fonction pour le nœud coordinateur
        def coordinator(state: TaskExtractionState) -> TaskExtractionState:
            """
            Divise la transcription en segments logiques pour traitement parallèle.
            
            Args:
                state: L'état actuel du graphe contenant la transcription
                
            Returns:
                État mis à jour avec les segments identifiés
            """
            # Extraire les valeurs de l'état
            transcript = state["transcript"]
            projects = state["projects"]
            current_date = state["current_date"]
            
            print("Coordinateur: division de la transcription en segments...")
            
            # Invoquer le LLM avec le prompt du coordinateur
            response = self.llm.invoke(
                self.coordinator_prompt.format(
                    transcript=transcript,
                    projects=", ".join(projects),
                    current_date=current_date
                )
            )
            
            # Extraire les segments du JSON dans la réponse
            segments_texts = []
            try:
                # Récupérer le contenu de la réponse
                content = response.content
                print(f"Réponse du coordinateur: {content[:100]}...")
                
                # Rechercher une structure JSON dans le texte
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    # JSON trouvé, essayer de l'extraire
                    json_str = json_match.group(0)
                    print(f"JSON extrait: {json_str[:100]}...")
                    
                    # Parser le JSON et récupérer les segments
                    data = json.loads(json_str)
                    segments_texts = data.get("segments", [])
                    print(f"Segments extraits du JSON: {len(segments_texts)}")
                else:
                    # Aucun JSON trouvé
                    print("Aucun JSON trouvé, utilisation de la transcription complète comme segment unique")
                    segments_texts = [transcript]
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON: {str(e)}")
                segments_texts = [transcript]
            except Exception as e:
                print(f"Erreur lors de l'extraction des segments: {str(e)}")
                segments_texts = [transcript]
            
            # Garantir qu'il y a au moins un segment
            if not segments_texts:
                print("Aucun segment trouvé, utilisation de la transcription complète comme segment unique")
                segments_texts = [transcript]
            
            # Créer les objets Segment
            segments = []
            for i, segment_text in enumerate(segments_texts):
                segments.append(Segment(id=i, text=segment_text))
                
            # Afficher des informations sur les segments
            print(f"La transcription a été divisée en {len(segments)} segments")
            for segment in segments:
                print(f"Segment {segment.id}: {len(segment.text)} caractères")
                print(f"Début du segment {segment.id}: {segment.text[:50]}...")
            
            # Retourner un état avec les segments
            return {"segments": segments}
        
        # Fonction pour le nœud travailleur (extraction de tâches)
        def worker(state: WorkerState) -> WorkerState:
            """
            Extrait les tâches d'un segment spécifique de la transcription.
            
            Args:
                state: État contenant le segment à traiter et les métadonnées
                
            Returns:
                État mis à jour avec les tâches extraites
            """
            # Extraire les valeurs de l'état
            segment = state["segment"]
            projects = state["projects"]
            current_date = state["current_date"]
            
            print(f"Worker démarré pour le segment {segment.id}...")
            
            # Invoquer le LLM avec le prompt du travailleur
            print(f"Worker invoquant le LLM pour le segment {segment.id}...")
            response = self.llm.invoke(
                self.worker_prompt.format(
                    segment_text=segment.text,
                    projects=", ".join(projects),
                    current_date=current_date
                )
            )
            print(f"Worker a reçu une réponse du LLM pour le segment {segment.id}")
            
            # Extraire les tâches du JSON dans la réponse
            tasks = []
            try:
                # Récupérer le contenu de la réponse
                content = response.content
                print(f"Worker contenu de la réponse pour segment {segment.id}: {content[:100]}...")
                
                # Chercher un tableau JSON (prioritaire)
                json_array_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
                if json_array_match:
                    json_str = json_array_match.group(0)
                    print(f"Worker a trouvé un tableau JSON dans le segment {segment.id}")
                    tasks = json.loads(json_str)
                else:
                    # Chercher un objet JSON simple
                    json_obj_match = re.search(r'{.*}', content, re.DOTALL)
                    if json_obj_match:
                        json_str = json_obj_match.group(0)
                        print(f"Worker a trouvé un objet JSON simple dans le segment {segment.id}")
                        task_data = json.loads(json_str)
                        if isinstance(task_data, dict):
                            tasks = [task_data]
                    else:
                        print(f"Worker n'a pas trouvé de JSON dans la réponse pour le segment {segment.id}")
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON dans le segment {segment.id}: {str(e)}")
            except Exception as e:
                print(f"Erreur lors de l'extraction des tâches du segment {segment.id}: {str(e)}")
            
            print(f"Le segment {segment.id} a extrait {len(tasks)} tâches")
            
            # Avec Annotated[List, operator.add], on retourne juste les nouvelles tâches
            return {"extracted_tasks": tasks}
        
        # Fonction pour le nœud d'agrégation
        def aggregator(state: TaskExtractionState) -> TaskExtractionState:
            """Agrège et nettoie les tâches extraites de tous les segments."""
            # Extraire les valeurs de l'état
            projects = state["projects"]
            current_date = state["current_date"]
            extracted_tasks = state.get("extracted_tasks", [])
            
            print(f"Agrégateur appelé. Nombre total de tâches: {len(extracted_tasks)}")
            
            # Si aucune tâche n'a été extraite, retourner une liste vide
            if not extracted_tasks:
                print("Aucune tâche à agréger")
                return {"final_tasks": []}
            
            # Pré-traitement des tâches avant de les envoyer au LLM
            # Regrouper les tâches qui se chevauchent dans le temps
            preprocessed_tasks = []
            time_period_groups = {}  # Dictionnaire pour regrouper les tâches par période de temps
            
            for task in extracted_tasks:
                # Normaliser les noms de champs
                task = {
                    "description": task.get("description", ""),
                    "project": task.get("project", task.get("projet", "Non spécifié")),
                    "start_time": task.get("start_time", task.get("heure_debut", "")),
                    "end_time": task.get("end_time", task.get("heure_fin", "")),
                    "date": task.get("date", current_date),
                    "actions": task.get("actions", [])
                }
                
                # Clé de période de temps pour le regroupement
                time_key = f"{task['date']}_{task['start_time']}_{task['end_time']}"
                
                if time_key in time_period_groups:
                    # Si cette période existe déjà, fusionner avec les tâches existantes
                    existing_group = time_period_groups[time_key]
                    # Vérifier si c'est la même tâche ou une tâche différente dans la même période
                    same_project = any(t["project"] == task["project"] for t in existing_group)
                    
                    if same_project:
                        # Même projet dans la même période, fusionner les actions
                        for t in existing_group:
                            if t["project"] == task["project"]:
                                # Fusionner les descriptions si elles sont différentes
                                if task["description"] not in t["description"]:
                                    t["description"] = f"{t['description']}; {task['description']}"
                                # Ajouter les actions qui ne sont pas déjà présentes
                                for action in task["actions"]:
                                    if action not in t["actions"]:
                                        t["actions"].append(action)
                    else:
                        # Différent projet dans la même période, ajouter comme nouvelle entrée
                        existing_group.append(task)
                else:
                    # Nouvelle période de temps
                    time_period_groups[time_key] = [task]
            
            # Aplatir les groupes en une liste
            for group in time_period_groups.values():
                preprocessed_tasks.extend(group)
            
            print(f"Après prétraitement: {len(preprocessed_tasks)} tâches")
            
            # Invoquer le LLM avec le prompt d'agrégation
            response = self.llm.invoke(
                self.aggregator_prompt.format(
                    extracted_tasks=json.dumps(preprocessed_tasks, ensure_ascii=False, indent=2),
                    projects=", ".join(projects),
                    current_date=current_date
                )
            )
            
            # Extraire les tâches agrégées du JSON dans la réponse
            final_tasks = []
            try:
                # Tenter de parser directement le contenu
                content = response.content
                print(f"Réponse de l'agrégateur: {content[:100]}...")
                # Recherche de structure JSON dans le texte
                import re
                
                # Chercher un tableau JSON
                json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    print("Agrégateur: tableau JSON trouvé")
                    tasks_data = json.loads(json_str)
                    
                    # Convertir en objets TaskWithTime
                    for task_data in tasks_data:
                        try:
                            # Normaliser les noms de champs
                            field_mapping = {
                                'description': 'description',
                                'projet': 'project',
                                'project': 'project',
                                'heure_debut': 'start_time',
                                'start_time': 'start_time',
                                'heure_fin': 'end_time',
                                'end_time': 'end_time',
                                'date': 'date',
                                'actions': 'actions'
                            }
                            
                            normalized_task = {}
                            for key, value in task_data.items():
                                if key in field_mapping:
                                    normalized_task[field_mapping[key]] = value
                                else:
                                    normalized_task[key] = value
                            
                            final_tasks.append(TaskWithTime(**normalized_task))
                        except Exception as e:
                            print(f"Erreur lors de la création d'une tâche: {str(e)}")
                else:
                    print("Agrégateur: aucun tableau JSON trouvé")
            except Exception as e:
                print(f"Erreur lors de l'agrégation des tâches: {str(e)}")
                
                # En cas d'erreur, utiliser les tâches prétraitées
                for task_data in preprocessed_tasks:
                    try:
                        final_tasks.append(TaskWithTime(**task_data))
                    except Exception as e:
                        print(f"Erreur lors de la création d'une tâche: {str(e)}")
            
            print(f"Agrégation terminée: {len(final_tasks)} tâches finales extraites")
            
            # Retourner les tâches finales
            return {"final_tasks": final_tasks}
        
        # Fonction pour assigner les travailleurs
        def assign_workers(state: TaskExtractionState):
            """Assigne un travailleur à chaque segment."""
            segments = state.get("segments", [])
            projects = state.get("projects", [])
            current_date = state.get("current_date", datetime.now().strftime("%Y-%m-%d"))
            
            print(f"Assignation des travailleurs pour {len(segments)} segments")
            
            # Utiliser l'API Send pour démarrer des travailleurs en parallèle
            return [
                Send(
                    "worker", 
                    {
                        "segment": segment,
                        "projects": projects,
                        "current_date": current_date
                    }
                ) 
                for segment in segments
            ]
        
        # Créer le graphe d'état
        builder = StateGraph(TaskExtractionState)
        
        # Ajouter les nœuds au graphe
        builder.add_node("coordinator", coordinator)
        builder.add_node("worker", worker)
        builder.add_node("aggregator", aggregator)
        
        # Définir le point d'entrée et les arêtes
        builder.add_edge(START, "coordinator")
        builder.add_conditional_edges("coordinator", assign_workers, ["worker"])
        builder.add_edge("worker", "aggregator")
        builder.add_edge("aggregator", END)
        
        # Compiler le graphe
        self.graph = builder.compile()
        
    def extract_tasks(self, text: str) -> List[TaskWithTime]:
        """
        Extrait les tâches d'un texte transcrit en utilisant le graphe LangGraph.
        
        Cette méthode initialise le graphe LangGraph avec la transcription fournie,
        exécute le processus d'extraction parallèle complet et retourne les tâches extraites.
        
        Args:
            text: Le texte transcrit à analyser
            
        Returns:
            Une liste de tâches extraites (instances de TaskWithTime)
        """
        try:
            # Préparer l'état initial
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            initial_state = {
                "transcript": text,
                "projects": self.projects,
                "current_date": current_date,
                "segments": [],
                "extracted_tasks": [],
                "final_tasks": []
            }
            
            print(f"Initialisation de l'état avec les projets: {', '.join(self.projects)}")
            print(f"Date actuelle: {current_date}")
            
            # Exécuter le graphe
            print("Exécution du graphe LangGraph...")
            result = self.graph.invoke(initial_state)
            
            # Extraire les tâches du résultat
            tasks = result.get("final_tasks", [])
            print(f"État final récupéré, nombre de tâches: {len(tasks)}")
            
            return tasks
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des tâches: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_tasks_from_file(self, transcript_file: str) -> List[TaskWithTime]:
        """
        Extrait les tâches d'un fichier de transcription.
        
        Args:
            transcript_file: Chemin vers le fichier de transcription
            
        Returns:
            Une liste de tâches extraites (instances de TaskWithTime)
        """
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.extract_tasks(text)
        except FileNotFoundError:
            print(f"Fichier non trouvé: {transcript_file}")
            return []
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de transcription: {str(e)}")
            return []
    
    def format_tasks_summary(self, tasks: List[TaskWithTime]) -> str:
        """
        Formate les tâches en un résumé lisible pour l'utilisateur.
        
        Génère un rapport formaté contenant les détails de chaque tâche,
        y compris la description, le projet, la période, la date et les actions.
        
        Args:
            tasks: Liste des tâches à formater
            
        Returns:
            Un résumé formaté des tâches
        """
        if not tasks:
            return "Aucune tâche extraite."
        
        summary = "📋 Tâches extraites:\n\n"
        
        for i, task in enumerate(tasks, 1):
            summary += f"Tâche {i}:\n"
            summary += f"📝 Description: {task.description}\n"
            summary += f"🏢 Projet: {task.project}\n"
            summary += f"⏰ Période: {task.start_time} - {task.end_time} ({task.duration_minutes} minutes)\n"
            summary += f"📅 Date: {task.date}\n"
            
            summary += "🔍 Actions:\n"
            for action in task.actions:
                summary += f"  • {action}\n"
            
            summary += "\n"
        
        return summary
    
    def save_graph_visualization(self, output_path: str = "task_extractor_graph.png") -> str:
        """
        Génère et sauvegarde une visualisation du graphe LangGraph.
        
        Cette méthode crée une représentation visuelle du graphe LangGraph 
        utilisé pour l'extraction des tâches et la sauvegarde au format PNG.
        
        Args:
            output_path: Chemin de sortie pour l'image générée (par défaut: "task_extractor_graph.png")
            
        Returns:
            Le chemin vers l'image générée
        """
        try:
            # Générer la visualisation Mermaid du graphe
            mermaid_png = self.graph.get_graph().draw_mermaid_png()
            
            # Sauvegarder l'image
            with open(output_path, 'wb') as f:
                f.write(mermaid_png)
            
            print(f"Visualisation du graphe sauvegardée dans: {output_path}")
            return output_path
        except Exception as e:
            print(f"Erreur lors de la génération de la visualisation du graphe: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

# Exemple d'utilisation
if __name__ == "__main__":
    """
    Exemple d'utilisation de l'extracteur de tâches.
    
    Si ce fichier est exécuté directement, il crée une instance de TaskAgentExtractor
    et l'utilise pour extraire les tâches d'un exemple de transcription.
    """
    try:
        # Exemple de projets
        projects = [
            "Website Redesign", 
            "Application Mobile", 
            "Base de données", 
            "Marketing", 
            "Formation personnelle", 
            "Mon marché recommandation", 
            "Mon marché Search", 
            "Développement", 
            "Réunion"
        ]
        
        # Créer l'extracteur
        print("Initialisation de l'extracteur...")
        extractor = TaskAgentExtractor(projects=projects)
        extractor.save_graph_visualization("task_extractor_graph.png")
        
        # Exemple de transcription
        test_text = """
        Bonjour, voici mon rapport d'activité pour aujourd'hui. J'ai passé la matinée à travailler sur le projet 
        "Mon marché Search". De 9h à 11h30, j'ai optimisé les algorithmes de recherche et amélioré le filtrage des résultats. 
        J'ai également corrigé quelques bugs liés à l'affichage des résultats de recherche et fait une revue de code 
        pour mon collègue.
        
        Ensuite, de 13h à 14h30, j'ai eu une réunion avec l'équipe marketing pour discuter des nouvelles fonctionnalités 
        à ajouter au site web. Nous avons planifié les prochaines étapes et réparti les tâches.
        
        Finalement, de 15h à 17h, j'ai travaillé sur la "Base de données" en nettoyant et optimisant certaines requêtes SQL 
        qui ralentissaient les performances. J'ai identifié plusieurs problèmes d'indexation et proposé des solutions.

        Qui plus est, vendredi dernier j'ai travaillé sur le projet Mon marché recommandation pendant 2h30 sur notamment la prédiction 
        de la catégorie pour les searchers pour améliorer le moteur de recherche. Pour que la méthode soit plus robuste, 
        j'ai fait une dernière vérification avec un modèle de langage.
        
        Ensuite, hier j'ai travaillé sur la mise en production avec Antoine du projet Mon marché recommandation. 
        Ça a duré toute la journée, environ 6 heures de travail à partir de 9h du matin. Nous avons notamment travaillé 
        sur l'intégration de GCP, amélioré l'approche incrémentale du code pour le moteur de recherche et renforcé 
        la sécurité grâce à une authentification du moteur de recherche pour les appels API.
        """
        
        # Extraire les tâches
        print("Extraction des tâches en cours...")
        tasks = extractor.extract_tasks(test_text)
        
        # Afficher le résumé
        print(extractor.format_tasks_summary(tasks))
    except KeyboardInterrupt:
        print("\nExtraction interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Erreur lors de l'exécution de l'exemple: {str(e)}")
        import traceback
        traceback.print_exc() 