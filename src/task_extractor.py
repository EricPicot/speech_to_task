import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Task(BaseModel):
    """Modèle pour représenter une tâche extraite du texte."""
    description: str = Field(description="Description détaillée de la tâche réalisée")
    project: str = Field(description="Projet associé à la tâche")
    duration_minutes: int = Field(description="Durée de la tâche en minutes")
    date: str = Field(description="Date à laquelle la tâche a été réalisée (format YYYY-MM-DD)")
    actions: List[str] = Field(description="Liste des actions spécifiques réalisées dans le cadre de cette tâche")

class TaskExtractor:
    """
    Classe pour extraire les tâches à partir d'un texte transcrit.
    Utilise OpenRouter via LangChain pour l'analyse du texte.
    """
    def __init__(self, projects=None):
        """
        Initialise l'extracteur de tâches.
        
        Args:
            projects: Liste optionnelle des projets disponibles
        """
        self.projects = projects or []
        
        # Configurer le LLM avec OpenRouter
        self.llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",  # Modèle demandé par l'utilisateur
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            # Les en-têtes sont maintenant correctement passés dans la configuration
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Speech to Task Tracker"
            }
        )
        
        # Configurer le parser pour les sorties
        self.parser = PydanticOutputParser(pydantic_object=Task)
        
        # Mapping pour les noms de champs incorrects
        self.field_mapping = {
            'task_description': 'description',
            'task': 'description', 
            'task_project': 'project',
            'task_date': 'date',
            'task_duration': 'duration_minutes',
            'duration': 'duration_minutes',
            'task_actions': 'actions'
        }
        
    def _create_prompt(self):
        """Crée le template de prompt pour l'extraction des tâches."""
        projects_info = ""
        if self.projects:
            projects_info = f"Voici la liste des projets disponibles: {', '.join(self.projects)}."
        
        system_template = """Tu es un assistant spécialisé dans l'extraction de tâches à partir de transcriptions vocales.
        
        Tu dois analyser le texte et identifier toutes les tâches décrites par l'utilisateur.
        
        Pour chaque tâche, tu dois extraire les informations suivantes:
        - description: Une description claire de la tâche
        - project: Le projet associé à cette tâche
        - duration_minutes: La durée de la tâche en minutes (entier)
        - date: La date à laquelle la tâche a été réalisée (au format YYYY-MM-DD)
        - actions: Une liste des actions spécifiques réalisées dans le cadre de cette tâche
        
        {projects_info}
        
        Règles importantes pour la date:
        - Si aucune date n'est spécifiée, utilise la date d'aujourd'hui: {today}
        - Si "hier" est mentionné, utilise cette date: {yesterday}
        - Si "avant-hier" est mentionné, utilise cette date: {day_before_yesterday}
        - Pour les autres références temporelles, détermine la date précise
        
        IMPORTANT: Utilise EXACTEMENT les noms de champs suivants dans ta réponse JSON:
        - description (pas task_description)
        - project (pas task_project)
        - duration_minutes (pas duration)
        - date
        - actions
        
        Identifie toutes les tâches mentionnées dans le texte.
        Si plusieurs tâches sont présentes, retourne-les sous forme de liste JSON.
        
        Important: Assure-toi que ta réponse est un JSON valide. N'inclus pas de backticks (```) ou d'indication de langage comme "json" dans ta réponse.
        Réponds uniquement avec un objet JSON ou un tableau JSON.
        """
        
        human_template = """Voici la transcription vocale à analyser:
        
        {text}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        return prompt
    
    def _clean_llm_response(self, response_text):
        """
        Nettoie la réponse du LLM pour extraire uniquement le JSON valide.
        
        Args:
            response_text: Le texte brut de la réponse du LLM
            
        Returns:
            Le texte JSON nettoyé
        """
        # Supprimer les backticks et les identifiants de langage
        clean_text = re.sub(r'^```(?:json|js)?\s*', '', response_text, flags=re.MULTILINE)
        clean_text = re.sub(r'\s*```$', '', clean_text, flags=re.MULTILINE)
        
        # Supprimer tout texte avant le premier { ou [
        match = re.search(r'[\[\{]', clean_text)
        if match:
            start_index = match.start()
            clean_text = clean_text[start_index:]
        
        return clean_text
    
    def _normalize_task_data(self, task_data):
        """
        Normalise les données de tâche en corrigeant les noms de champs incorrects.
        
        Args:
            task_data: Dictionnaire contenant les données de la tâche
            
        Returns:
            Dictionnaire normalisé avec les bons noms de champs
        """
        normalized_data = {}
        
        # Parcourir les données de la tâche
        for key, value in task_data.items():
            # Si la clé est dans le mapping, utiliser la clé correcte
            if key in self.field_mapping:
                normalized_data[self.field_mapping[key]] = value
            else:
                normalized_data[key] = value
                
        # Vérifier si on a les champs obligatoires
        required_fields = ['description', 'project', 'duration_minutes', 'date', 'actions']
        for field in required_fields:
            if field not in normalized_data:
                # Essayer de dériver le champ à partir d'autres données
                if field == 'actions' and 'description' in normalized_data:
                    # Si pas d'actions mais qu'on a une description, créer une action générique
                    normalized_data[field] = [f"Réaliser la tâche: {normalized_data['description']}"]
                
        return normalized_data
        
    def extract_tasks(self, text: str) -> List[Task]:
        """
        Extrait les tâches d'un texte transcrit.
        
        Args:
            text: Le texte transcrit à analyser
            
        Returns:
            Une liste de tâches extraites
        """
        # Préparer les dates pour le contexte
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        day_before_yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        
        # Créer le prompt
        prompt = self._create_prompt()
        
        try:
            # Préparer les variables pour le prompt
            variables = {
                "text": text,
                "projects_info": "Voici la liste des projets disponibles: " + ", ".join(self.projects) if self.projects else "",
                "today": today,
                "yesterday": yesterday, 
                "day_before_yesterday": day_before_yesterday
            }
            
            # Formater le prompt
            formatted_prompt = prompt.format_messages(**variables)
            
            # Obtenir la réponse du LLM
            llm_response = self.llm.invoke(formatted_prompt)
            
            # Afficher la réponse brute pour le débogage
            print(f"Réponse brute du LLM:\n{llm_response.content}\n")
            
            # Nettoyer la réponse pour avoir un JSON valide
            clean_json = self._clean_llm_response(llm_response.content)
            
            # Afficher le JSON nettoyé pour le débogage
            print(f"JSON nettoyé:\n{clean_json}\n")
            
            # Analyser manuellement le JSON
            try:
                # Essayer de parser directement
                parsed_json = json.loads(clean_json)
                
                # Convertir en objets Task
                tasks = []
                if isinstance(parsed_json, list):
                    # C'est une liste de tâches
                    for task_data in parsed_json:
                        try:
                            # Normaliser les données de la tâche
                            normalized_data = self._normalize_task_data(task_data)
                            
                            # Afficher les données normalisées pour le débogage
                            print(f"Données normalisées: {normalized_data}")
                            
                            # Créer l'objet Task
                            task = Task(**normalized_data)
                            tasks.append(task)
                        except Exception as e:
                            print(f"Erreur lors de la conversion d'une tâche: {str(e)}")
                            print(f"Données d'origine: {task_data}")
                else:
                    # C'est une seule tâche
                    try:
                        # Normaliser les données de la tâche
                        normalized_data = self._normalize_task_data(parsed_json)
                        
                        # Afficher les données normalisées pour le débogage
                        print(f"Données normalisées: {normalized_data}")
                        
                        # Créer l'objet Task
                        task = Task(**normalized_data)
                        tasks.append(task)
                    except Exception as e:
                        print(f"Erreur lors de la conversion d'une tâche: {str(e)}")
                        print(f"Données d'origine: {parsed_json}")
                
                return tasks
                
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON: {str(e)}")
                print(f"JSON reçu: {clean_json}")
                
                # Essayer une approche plus robuste pour réparer le JSON
                # Ce code n'est pas complet, mais illustre une approche possible
                return []
                
        except Exception as e:
            print(f"Erreur lors de l'extraction des tâches: {str(e)}")
            return []
    
    def extract_tasks_from_file(self, transcript_file: str) -> List[Task]:
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
    
    def format_tasks_summary(self, tasks: List[Task]) -> str:
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
            summary += f"⏱️ Durée: {task.duration_minutes} minutes\n"
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
                "Formation personnelle", "Mon marché recommandation", "Développement", "Réunion"]
    
    # Créer l'extracteur
    extractor = TaskExtractor(projects=projects)
    
    # Exemple de transcription
    text = """alors aujourd'hui j'ai travaillé 30 minutes pour récapituler ce que j'ai fait hier sur le projet mon marché recommandation et ensuite j'ai fait 2h30 de travail intensif pour créer une application speech to text qui me permet de plus rapidement traquer les tâches que j'ai fait ça il faut l'ajouter au projet formation personnelle est le résultat de cette étape c'est d'avoir une application radio d'avoir une transcription du speech vers le texte qui fonctionne et qui s'enregistre automatiquement"""
    
    # Extraire les tâches
    tasks = extractor.extract_tasks(text)
    
    # Afficher le résumé
    print(extractor.format_tasks_summary(tasks)) 