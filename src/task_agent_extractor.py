"""
Task Agent Extractor with Notion Integration

Un système qui utilise LangGraph pour extraire des tâches à partir de transcriptions vocales de manière parallélisée,
avec intégration Notion pour la gestion des projets et la synchronisation des données.

Author: Eric Picot
"""

import os
import re
import json
import uuid
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, TypedDict, Annotated, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator
import asyncio
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send

from notion_integration import NotionIntegration
from models.data_models import Project, TaskWithTime, Segment
from models.state_models import TaskExtractionState, WorkerState
from prompts import get_coordinator_prompt, get_worker_prompt, get_aggregator_prompt

load_dotenv()

logger = logging.getLogger(__name__)




class TaskAgentExtractor:
    """Extracteur de tâches basé sur LangGraph avec intégration Notion."""
    
    def __init__(self, llm: BaseChatModel, notion: NotionIntegration):
        """
        Initialise l'extracteur de tâches.
        
        Args:
            llm: Le modèle de langage à utiliser
            notion: L'intégration Notion pour la synchronisation des projets
        """
        self.llm = llm
        self.notion = notion
        
        # Initialiser les prompts
        self.coordinator_prompt = get_coordinator_prompt()
        self.worker_prompt = get_worker_prompt()
        self.aggregator_prompt = get_aggregator_prompt()
        
        # Construire le graphe d'état
        builder = StateGraph("task_extraction")
        
        # Ajouter les nœuds
        builder.add_node("coordinator", self.coordinator)
        builder.add_node("worker", self.worker)
        builder.add_node("aggregator", self.aggregator)
        
        # Définir les transitions
        builder.set_entry_point("coordinator")
        
        # Define conditional edges for segment processing
        def should_continue_processing(state):
            """Determine if there are more segments to process."""
            current_index = state.get("current_segment_index", 0)
            segments = state.get("segments", [])
            return "worker" if current_index < len(segments) else "aggregator"
        
        # Add edges with conditions
        builder.add_conditional_edges(
            "coordinator",
            should_continue_processing,
            {
                "worker": "worker",
                "aggregator": "aggregator"
            }
        )
        
        builder.add_conditional_edges(
            "worker",
            should_continue_processing,
            {
                "worker": "worker",
                "aggregator": "aggregator"
            }
        )
        
        # Set the aggregator as the end point
        builder.set_finish_point("aggregator")
        
        # Compile the graph
        self.graph = builder.compile()
    
    async def initialize(self):
        """Initialize Notion integration and sync projects."""
        await self.notion.__aenter__()
        self.projects = await self.notion.sync_projects()
        return self
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.notion.__aexit__(None, None, None)
    
    def coordinator(self, state: Dict) -> Dict:
        """
        Coordonne l'extraction des tâches en divisant le texte en segments logiques.
        """
        try:
            # Extraire les valeurs de l'état
            text = state["text"]
            projects_info = state["projects_info"]
            current_date = state["current_date"]
            
            # Appeler le LLM pour diviser en segments
            response = self.llm.invoke(
                self.coordinator_prompt.format(
                    text=text,
                    projects_info=projects_info,
                    current_date=current_date
                )
            )
            
            # Extraire les segments du JSON
            try:
                # Chercher un JSON dans la réponse
                response_text = response.content if hasattr(response, 'content') else str(response)
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    json_str = match.group()
                    data = json.loads(json_str)
                    segments_texts = data.get("segments", [])
                else:
                    # Aucun JSON trouvé
                    logger.warning("Aucun JSON trouvé, utilisation de la transcription complète comme segment unique")
                    segments_texts = [text]
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON: {str(e)}")
                segments_texts = [text]
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction des segments: {str(e)}")
                segments_texts = [text]
            
            # Garantir qu'il y a au moins un segment
            if not segments_texts:
                logger.warning("Aucun segment trouvé, utilisation de la transcription complète comme segment unique")
                segments_texts = [text]
            
            # Créer les objets Segment
            segments = [
                Segment(
                    id=str(uuid.uuid4()),
                    text=segment_text,
                    tasks=[]  # Initialize empty tasks list
                )
                for segment_text in segments_texts
            ]
            
            # Mettre à jour l'état
            state["segments"] = segments
            state["current_segment_index"] = 0
            state["all_tasks"] = []  # Initialize list to collect all tasks
            
            # Set the first segment as current
            if segments:
                state["segment"] = segments[0]
            
            logger.info(f"Created {len(segments)} segments")
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans le coordinateur: {str(e)}")
            logger.error(traceback.format_exc())
            return state
    
    def worker(self, state: Dict) -> Dict:
        """
        Extrait les tâches d'un segment de texte.
        """
        try:
            # Get current segment and index
            current_index = state.get("current_segment_index", 0)
            segments = state.get("segments", [])
            
            if current_index >= len(segments):
                logger.warning("No more segments to process")
                return state
            
            segment = segments[current_index]
            projects_info = state["projects_info"]
            current_date = state["current_date"]
            
            logger.info(f"Processing segment {current_index + 1}/{len(segments)} (ID: {segment.id})")
            
            # Extract tasks from segment
            response = self.llm.invoke(
                self.worker_prompt.format(
                    segment_text=segment.text,
                    projects_info=projects_info,
                    current_date=current_date
                )
            )
            
            # Parse tasks from response
            tasks = []
            try:
                response_text = response.content if hasattr(response, 'content') else str(response)
                json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                if json_match:
                    tasks = json.loads(json_match.group(0))
                    segment.tasks = tasks
                    state["all_tasks"].extend(tasks)  # Add tasks to global collection
                    logger.info(f"Extracted {len(tasks)} tasks from segment {segment.id}")
                else:
                    logger.warning(f"No tasks found in segment {segment.id}")
                    segment.tasks = []
            except Exception as e:
                logger.error(f"Error parsing tasks from segment {segment.id}: {str(e)}")
                segment.tasks = []
            
            # Update state for next iteration
            state["current_segment_index"] = current_index + 1
            if current_index + 1 < len(segments):
                state["segment"] = segments[current_index + 1]
            
            return state
            
        except Exception as e:
            logger.error(f"Error in worker: {str(e)}")
            logger.error(traceback.format_exc())
            return state
    
    def aggregator(self, state: Dict) -> Dict:
        """
        Agrège les tâches extraites de tous les segments.
        """
        try:
            # Get all collected tasks and projects dictionary
            all_tasks = state.get("all_tasks", [])
            projects_dict = state.get("projects_dict", {})
            
            logger.info(f"Aggregating {len(all_tasks)} tasks from all segments")
            
            # Vérifier et formater les tâches
            final_tasks = []
            for task in all_tasks:
                try:
                    # Check required fields
                    if not all(key in task for key in ["title", "project", "start_time", "end_time", "date"]):
                        logger.warning(f"Task ignored - missing fields: {task}")
                        continue
                    
                    # Récupérer l'ID du projet
                    project_name = task["project"]
                    project_id = projects_dict.get(project_name)
                    
                    if not project_id:
                        logger.warning(f"Projet non trouvé: {project_name}")
                        project_id = "Non spécifié"
                    
                    # Créer la tâche finale
                    final_task = TaskWithTime(
                        title=task["title"],
                        project_name=project_name,
                        project_id=project_id,
                        start_time=task["start_time"],
                        end_time=task["end_time"],
                        date=task["date"],
                        actions=task.get("actions", [])
                    )
                    
                    final_tasks.append(final_task)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la tâche: {str(e)}")
                    continue
            
            # Mettre à jour l'état
            state["final_tasks"] = final_tasks
            logger.info(f"Agrégation terminée: {len(final_tasks)} tâches valides")
            
            return state
            
        except Exception as e:
            logger.error(f"Erreur dans l'agrégateur: {str(e)}")
            logger.error(traceback.format_exc())
            return state
    
    async def extract_tasks(self, text: str, create_timesheet_entries: bool = False) -> List[TaskWithTime]:
        """Extract tasks from text and optionally create timesheet entries in Notion.
        
        Args:
            text: The text to extract tasks from
            create_timesheet_entries: If True, create timesheet entries in Notion
            
        Returns:
            List of extracted tasks
        """
        try:
            # Sync projects first
            projects = await self.notion.sync_projects()
            if not projects:
                logger.error("No projects found in Notion")
                return []
                
            # Format projects info for workers
            projects_info = "\n".join([f"- {p.name} (ID: {p.id})" for p in projects])
            
            # Create projects dictionary for efficient lookup
            projects_dict = {p.name: p.id for p in projects}
            
            # Initialize state with projects information
            initial_state = {
                "text": text,
                "segments": [],
                "tasks": [],
                "projects": projects,
                "projects_dict": projects_dict,
                "projects_info": projects_info,
                "current_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Get extracted tasks
            tasks = result.get("final_tasks", [])
            logger.info(f"Extracted {len(tasks)} tasks")
            # Create timesheet entries if requested
            if tasks and create_timesheet_entries:
                logger.info(f"Creating {len(tasks)} timesheet entries in Notion")
                await self.notion.create_timesheet_entries(tasks)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des tâches: {str(e)}")
            logger.error(traceback.format_exc())
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
    
    async def format_tasks_summary(self, tasks: List[TaskWithTime]) -> str:
        """
        Formate les tâches en un résumé lisible pour l'utilisateur.
        
        Génère un rapport formaté contenant les détails de chaque tâche,
        y compris le titre, le projet, la période et les actions.
        
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
            summary += f"📝 Titre: {task.title}\n"
            summary += f"🏢 Projet: {task.project_name}\n"
            summary += f"⏰ Période: {task.start_time} - {task.end_time} ({task.duration_minutes} minutes)\n"
            
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

async def main():
    """Point d'entrée principal du script."""
    try:
        # Initialize LLM with OpenRouter
        llm = ChatOpenAI(
            temperature=0,
            model="google/gemini-2.0-flash-001",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Notion MCP Agent"
            }
        )
        
        # Initialize Notion integration
        notion = NotionIntegration()
        await notion.__aenter__()
        
        # Sync projects
        projects = await notion.sync_projects()
        print(f"Synced {len(projects)} projects from Notion")
        
        # Test task extraction
        TEST_FILE = 'transcripts/transcript_20250409_124339.txt'
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            test_text = f.read()
        
        # Create task agent
        agent = TaskAgentExtractor(llm=llm, notion=notion)
        # agent.save_graph_visualization()
        
        # Extract tasks
        tasks = await agent.extract_tasks(test_text, create_timesheet_entries=True)
        print(f"Extracted {len(tasks)} tasks")

        # Format and display the tasks summary
        summary = await agent.format_tasks_summary(tasks=tasks)
        print("\n" + summary)  # Print the formatted summary
        
        # Cleanup
        await notion.__aexit__(None, None, None)
        
    except Exception as e:
        logger.error(f"Erreur dans le programme principal: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 




    