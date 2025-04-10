from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

@dataclass
class Segment:
    """Représente un segment de texte pour l'analyse."""
    id: str
    text: str
    tasks: List[Dict] = field(default_factory=list)

@dataclass
class Project:
    """Représente un projet dans Notion."""
    id: str
    name: str
    database_id: str

@dataclass
class TaskWithTime:
    """Représente une tâche avec son timing."""
    description: str
    project_name: str
    project_id: str
    start_time: str
    end_time: str
    date: str
    actions: List[str] 

class TaskWithTime(BaseModel):
    """Modèle pour représenter une tâche extraite avec les heures de début et de fin."""
    title: str = Field(description="Titre de la tâche, court et clair")
    project_name: str = Field(description="Nom du projet associé à la tâche")
    project_id: Optional[str] = Field(description="ID Notion du projet", default=None)
    date: str = Field(description="Date de la tâche (format YYYYMMDD)")
    start_time: str = Field(description="Heure de début de la tâche (format HH:MM)")
    end_time: str = Field(description="Heure de fin de la tâche (format HH:MM)")
    actions: List[str] = Field(description="Liste des actions spécifiques réalisées")
    
    @property
    def duration_minutes(self) -> int:
        """Calcule la durée de la tâche en minutes."""
        try:
            start = datetime.strptime(self.start_time, "%H:%M")
            end = datetime.strptime(self.end_time, "%H:%M")
            if end < start:
                end += timedelta(days=1)
            delta = end - start
            return int(delta.total_seconds() / 60)
        except Exception:
            return 0 
