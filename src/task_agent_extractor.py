import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Importations LangChain pour les agents
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

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
        except:
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
    
    # Références simples
    if query.lower() in ["aujourd'hui", "maintenant"]:
        return today.strftime("%Y-%m-%d")
    elif query.lower() == "hier":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif query.lower() in ["avant-hier", "avant hier"]:
        return (today - timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Jours de la semaine
    days_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    for i, day in enumerate(days_fr):
        if day in query.lower():
            # Convertir l'indice du jour en format Python (0 = lundi dans notre liste, 0 = lundi dans Python)
            target_weekday = i
            current_weekday = today.weekday()
            
            # Calculer la différence de jours
            if "dernier" in query.lower() or "passé" in query.lower():
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

def extract_tasks_from_text(text: str, projects: List[str] = None) -> str:
    """
    Extrait les tâches d'un texte.
    
    Args:
        text: Le texte à analyser
        projects: Liste optionnelle des projets disponibles
        
    Returns:
        Un JSON représentant les tâches extraites
    """
    # Cette fonction est juste un wrapper pour l'agent
    return f"Analyser le texte suivant pour identifier les tâches: {text}"
        
class TaskAgentExtractor:
    """
    Classe qui utilise un agent LangChain pour extraire les tâches.
    Cette approche est plus robuste car l'agent peut raisonner et résoudre
    les problèmes de manière autonome.
    """
    
    def __init__(self, projects=None):
        """
        Initialise l'agent d'extraction de tâches.
        
        Args:
            projects: Liste optionnelle des projets disponibles
        """
        self.projects = projects or []
        
        # Créer les outils avec la nouvelle méthode d'initialisation
        self.date_tool = Tool(
            name="date_resolver",
            func=resolve_date_reference,
            description="Résout une référence temporelle relative comme 'hier', 'avant-hier', etc. en une date absolue"
        )
        
        self.task_tool = Tool(
            name="task_extractor",
            func=lambda text: extract_tasks_from_text(text, self.projects),
            description="Extrait les tâches mentionnées dans un texte, avec leurs détails (projet, heures, date, actions)"
        )
        
        # Configurer le LLM avec OpenRouter
        self.llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Speech to Task Tracker"
            }
        )
        
        # Configurer le prompt pour l'agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un agent spécialisé dans l'extraction de tâches à partir de transcriptions vocales.
            
            Ton objectif est d'identifier toutes les tâches mentionnées dans le texte et d'extraire :
            1. Une description claire de chaque tâche
            2. Le projet associé à chaque tâche (si non spécifié, indiquer "Non spécifié")
            3. L'heure de début et de fin de chaque tâche (au format HH:MM)
            4. La date à laquelle chaque tâche a été réalisée (au format YYYY-MM-DD)
            5. Les actions spécifiques réalisées
            
            Il se peut que l'heure de début et de fin ne soient pas précisées dans le texte mais que seule la durée soit mentionnée.
            Dans ce cas, invente une heure de début et de fin théorique en fonction de la durée de la tâche. 
            De même, si la date n'est pas précisée, utilise la date du jour actuelle.
            
            Si une date vague comme "la semaine dernière" ou "le mois passé" est mentionnée, utilise l'outil date_resolver.
            
            Si aucun projet n'est mentionné, utilise le projet "Non spécifié".
            
            Si plusieurs tâches sont mentionnées, identifie-les toutes.
            Pour les références temporelles comme "hier", "aujourd'hui", "lundi dernier", etc., utilise l'outil date_resolver.
            
            Ton objectif est de comprendre les intentions de l'utilisateur même si le texte n'est pas parfaitement structuré.
            
            Retourne ta réponse au format JSON avec les clés suivantes:
            - description (description de la tâche)
            - project (nom du projet)
            - start_time (heure de début au format HH:MM)
            - end_time (heure de fin au format HH:MM)
            - date (date au format YYYY-MM-DD)
            - actions (liste d'actions entreprises)
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Formater les outils pour l'agent
        tools = [self.date_tool, self.task_tool]
        llm_with_tools = self.llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
        
        # Configurer la mémoire
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Créer l'agent
        self.agent_executor = AgentExecutor(
            agent=create_openai_functions_agent(
                llm=llm_with_tools, 
                prompt=self.prompt,
                tools=tools
            ),
            tools=tools,
            memory=self.memory,
            verbose=True
        )
        
    def extract_tasks(self, text: str) -> List[TaskWithTime]:
        """
        Extrait les tâches d'un texte transcrit.
        
        Args:
            text: Le texte transcrit à analyser
            
        Returns:
            Une liste de tâches extraites
        """
        try:
            # Préparer les variables pour l'agent
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Construire un message qui inclut toutes les informations nécessaires
            input_message = f"""Analyse ce texte pour extraire les tâches:

{text}

Projects disponibles: {", ".join(self.projects)}
Date actuelle: {current_date}"""
            
            # Exécuter l'agent avec un seul paramètre d'entrée
            result = self.agent_executor.invoke({
                "input": input_message
            })
            
            # Extraire les tâches de la réponse de l'agent
            agent_output = result["output"]
            
            print(f"Réponse de l'agent:\n{agent_output}\n")
            
            # Tenter d'extraire un JSON de la réponse
            tasks = self._extract_tasks_from_output(agent_output)
            
            return tasks
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des tâches: {str(e)}")
            return []
    
    def _extract_tasks_from_output(self, output: str) -> List[TaskWithTime]:
        """
        Extrait les tâches structurées de la sortie de l'agent.
        
        Args:
            output: La sortie texte de l'agent
            
        Returns:
            Une liste d'objets TaskWithTime
        """
        tasks = []
        
        # Mappings des champs français vers anglais
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
        
        # Essayer de trouver des structures JSON dans la sortie
        try:
            # Chercher un tableau JSON complet
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"JSON extrait: {json_str}")
                tasks_data = json.loads(json_str)
                for task_data in tasks_data:
                    try:
                        # Normaliser les noms de champs
                        normalized_task = {}
                        for key, value in task_data.items():
                            if key in field_mapping:
                                normalized_task[field_mapping[key]] = value
                            else:
                                normalized_task[key] = value
                                
                        tasks.append(TaskWithTime(**normalized_task))
                    except Exception as e:
                        print(f"Erreur lors de la création d'une tâche: {str(e)}")
                        print(f"Données: {task_data}")
                return tasks
            
            # Si pas de tableau, chercher un objet JSON simple
            json_match = re.search(r'{.*}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"JSON extrait: {json_str}")
                task_data = json.loads(json_str)
                try:
                    # Normaliser les noms de champs
                    normalized_task = {}
                    for key, value in task_data.items():
                        if key in field_mapping:
                            normalized_task[field_mapping[key]] = value
                        else:
                            normalized_task[key] = value
                            
                    tasks.append(TaskWithTime(**normalized_task))
                except Exception as e:
                    print(f"Erreur lors de la création d'une tâche: {str(e)}")
                    print(f"Données: {task_data}")
                return tasks
                
        except Exception as e:
            print(f"Erreur lors de l'extraction du JSON: {str(e)}")
        
        # Si aucun JSON n'a été trouvé, demander à l'agent de reformater sa réponse
        try:
            # Demander à l'agent de retourner les données au format JSON
            result = self.agent_executor.invoke({
                "input": "Reformate ta dernière réponse sous forme de JSON valide. Fournit un tableau de tâches avec les champs suivants: description, project, start_time, end_time, date, actions (liste de chaînes)."
            })
            
            # Tenter d'extraire le JSON reformaté
            reformatted_output = result["output"]
            print(f"Réponse reformatée:\n{reformatted_output}\n")
            
            # Chercher un tableau JSON dans la sortie reformatée
            json_match = re.search(r'\[\s*{.*}\s*\]', reformatted_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"JSON extrait après reformatage: {json_str}")
                tasks_data = json.loads(json_str)
                for task_data in tasks_data:
                    try:
                        # Normaliser les noms de champs
                        normalized_task = {}
                        for key, value in task_data.items():
                            if key in field_mapping:
                                normalized_task[field_mapping[key]] = value
                            else:
                                normalized_task[key] = value
                                
                        tasks.append(TaskWithTime(**normalized_task))
                    except Exception as e:
                        print(f"Erreur lors de la création d'une tâche: {str(e)}")
                        print(f"Données: {task_data}")
                return tasks
                
            # Si pas de tableau, chercher un objet JSON simple
            json_match = re.search(r'{.*}', reformatted_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"JSON extrait après reformatage: {json_str}")
                task_data = json.loads(json_str)
                try:
                    # Normaliser les noms de champs
                    normalized_task = {}
                    for key, value in task_data.items():
                        if key in field_mapping:
                            normalized_task[field_mapping[key]] = value
                        else:
                            normalized_task[key] = value
                            
                    tasks.append(TaskWithTime(**normalized_task))
                except Exception as e:
                    print(f"Erreur lors de la création d'une tâche: {str(e)}")
                    print(f"Données: {task_data}")
                return tasks
                
        except Exception as e:
            print(f"Erreur lors de la reformatage: {str(e)}")
        
        return tasks
    
    def extract_tasks_from_file(self, transcript_file: str) -> List[TaskWithTime]:
        """
        Extrait les tâches d'un fichier de transcription.
        
        Args:
            transcript_file: Chemin vers le fichier de transcription
            
        Returns:
            Une liste de tâches extraites
        """
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.extract_tasks(text)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de transcription: {str(e)}")
            return []
    
    def format_tasks_summary(self, tasks: List[TaskWithTime]) -> str:
        """
        Formate les tâches en un résumé lisible.
        
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

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de projets
    projects = ["Website Redesign", "Application Mobile", "Base de données", "Marketing", 
                "Formation personnelle", "Mon marché recommandation", "Mon marché Search", "Développement", "Réunion"]
    
    # Créer l'extracteur
    extractor = TaskAgentExtractor(projects=projects)
    
    # Exemple de transcription
    text = open("transcripts/transcript_20250408_112130.txt", "r").read()
    # Extraire les tâches
    tasks = extractor.extract_tasks(text)
    
    # Afficher le résumé
    print(extractor.format_tasks_summary(tasks)) 