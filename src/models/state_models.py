from typing import TypedDict, List, Dict, Annotated
import operator
from models.data_models import Segment, TaskWithTime

class TaskExtractionState(TypedDict):
    """
    État global du graphe d'extraction de tâches.
    Contient les entrées, l'état interne et la sortie du processus d'extraction.
    """
    # Entrées
    text: str                           # Transcription complète
    projects_dict: Dict[str, str]            # Mapping nom de projet -> ID Notion
    projects_info: str                         # Information sur les projets
    current_date: str                         # Date actuelle
    
    # État interne
    segments: List[Segment]                   # Segments de la transcription
    tasks: Annotated[List[Dict], operator.add]  # Tâches extraites par tous les workers
    
    # Sortie finale
    final_tasks: List[TaskWithTime]           # Tâches agrégées et formatées

class WorkerState(TypedDict):
    """
    État pour un worker traitant un segment spécifique.
    Contient les informations nécessaires pour extraire les tâches d'un segment.
    """
    segment: Segment                          # Segment à traiter
    projects_dict: Dict[str, str]            # Mapping nom de projet -> ID Notion
    current_date: str                         # Date actuelle
    extracted_tasks: Annotated[List[Dict], operator.add]  # Pour collecter les tâches 