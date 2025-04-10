import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient
from models.data_models import Project, TaskWithTime

logger = logging.getLogger(__name__)

class NotionIntegration:
    """Gère l'intégration avec Notion pour la synchronisation des projets et des tâches."""
    
    def __init__(self):
        """Initialise l'intégration Notion."""
        self.mcp_config = {
            "notion": {
                "command": "node",
                "args": ["/Applications/Utilities/notionmcp/build/index.js"],
                "transport": "stdio",
                "env": {
                    "NOTION_API_TOKEN": os.getenv("NOTION_API_TOKEN")
                }
            }
        }
        self.project_database_id = os.getenv("project_database_id")
        self.timesheet_database_id = os.getenv("timesheet_database_id")
        self.projects_cache: Dict[str, Project] = {}
        self.last_sync: Optional[datetime] = None
        self.client = None
        self.tools = None
    
    async def __aenter__(self):
        """Initialize the MCP client and get tools."""
        try:
            self.client = MultiServerMCPClient(self.mcp_config)
            await self.client.__aenter__()
            self.tools = self.client.get_tools()
            
            # Find all required tools
            self.query_database_tool = next(
                (tool for tool in self.tools if tool.name == "notion_query_database"),
                None
            )
            self.create_database_item_tool = next(
                (tool for tool in self.tools if tool.name == "notion_create_database_item"),
                None
            )
            self.append_block_children_tool = next(
                (tool for tool in self.tools if tool.name == "notion_append_block_children"),
                None
            )
            
            # Check all tools are found
            if not all([self.query_database_tool, self.create_database_item_tool, self.append_block_children_tool]):
                missing_tools = [
                    tool_name for tool_name, tool in [
                        ("notion_query_database", self.query_database_tool),
                        ("notion_create_database_item", self.create_database_item_tool),
                        ("notion_append_block_children", self.append_block_children_tool)
                    ] if not tool
                ]
                raise ValueError(f"Could not find required tools: {', '.join(missing_tools)}")
                
            print("Successfully initialized Notion integration")
            return self
            
        except Exception as e:
            print(f"Error initializing Notion integration: {str(e)}")
            if self.client:
                await self.client.__aexit__(None, None, None)
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the MCP client."""
        if self.client:
            try:
                await self.client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
                # Don't re-raise as we're in cleanup
    
    async def sync_projects(self, force: bool = False) -> List[Project]:
        """
        Synchronise les projets depuis Notion.
        
        Args:
            force: Force la synchronisation même si le cache est récent
            
        Returns:
            Liste des projets synchronisés
        """
        # Vérifier si une synchronisation est nécessaire
        now = datetime.now()
        if not force and self.last_sync and (now - self.last_sync).total_seconds() < 300:
            print("Using cached projects...")
            return list(self.projects_cache.values())
        
        try:
            print(f"Fetching projects from database {self.project_database_id}...")
            
            # Use the tool through its run method
            response = await self.query_database_tool.ainvoke({
                "database_id": self.project_database_id
            })
            
            # Parse the JSON response if it's a string
            if isinstance(response, str):
                response = json.loads(response)
            
            print(f"Got response from Notion: {str(response)[:200]}...")
            
            # Traiter chaque projet
            projects = []
            for item in response.get("results", []):
                try:
                    project_id = item.get("id")
                    properties = item.get("properties", {})
                    
                    # Handle potential missing title field more gracefully
                    title_content = (
                        properties.get("Name", {})
                        .get("title", [{}])[0]
                        .get("text", {})
                        .get("content", "Unknown")
                    )
                    
                    if not title_content or title_content == "Unknown":
                        print(f"Warning: Could not extract title for project {project_id}")
                        continue
                    
                    project = Project(
                        id=project_id,
                        name=title_content,
                        database_id=self.project_database_id
                    )
                    projects.append(project)
                    self.projects_cache[title_content] = project
                    print(f"Added project: {title_content} (ID: {project_id})")
                    
                except Exception as e:
                    print(f"Error processing project item: {str(e)}")
                    print(f"Project data: {item}")
                    continue
            
            self.last_sync = now
            print(f"Successfully synced {len(projects)} projects")
            return projects
            
        except Exception as e:
            print(f"Erreur lors de la synchronisation des projets: {str(e)}")
            import traceback
            traceback.print_exc()
            return list(self.projects_cache.values())
    
    def get_project_id(self, project_name: str) -> Optional[str]:
        """Récupère l'ID d'un projet par son nom."""
        project = self.projects_cache.get(project_name)
        return project.id if project else None
    
    async def create_timesheet_entry(self, task: TaskWithTime) -> Optional[str]:
        """Create a single timesheet entry in Notion.
        
        Args:
            task: The task to create a timesheet entry for
            
        Returns:
            The ID of the created page if successful, None otherwise
        """
        try:
            # Format task properties according to Notion schema
            properties = {
                "Name": {
                    "title": [
                        {
                            "text": {
                                "content": "TEST - " + task.title
                            }
                        }
                    ]
                },
                "Project": {
                    "relation": [
                        {
                            "id": task.project_id
                        }
                    ]
                },
                "Start time": {
                    "date": {
                        "start": f"{task.date}T{task.start_time}:00"
                    }
                },
                "End time": {
                    "date": {
                        "start": f"{task.date}T{task.end_time}:00"
                    }
                }
            }
            
            # Only add Project relation if we have a valid project_id
            if task.project_id and task.project_id != "Non spécifié":
                try:
                    # Validate project_id format (should be a UUID)
                    if len(task.project_id) == 36 and task.project_id.count('-') == 4:
                        properties["Project"] = {
                            "relation": [
                                {
                                    "id": task.project_id
                                }
                            ]
                        }
                    else:
                        logger.warning(f"Invalid project_id format: {task.project_id}")
                except Exception as e:
                    logger.warning(f"Error validating project_id: {str(e)}")
            
            # Create the page in the timesheet database
            response = await self.create_database_item_tool.ainvoke({
                "database_id": self.timesheet_database_id,
                "properties": properties
            })
            
            # Parse the response if it's a string
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse response as JSON: {response}")
                    return None
            
            if not isinstance(response, dict):
                logger.error(f"Invalid response type: {type(response)}")
                return None
                
            if "id" not in response:
                logger.error("Failed to create timesheet entry: no page ID returned")
                logger.error(f"Response: {response}")
                return None
                
            page_id = response["id"]
            
            # Add description block with task title and actions
            try:
                description_content = task.title + "\n\nActions réalisées:\n"
                if task.actions:
                    for action in task.actions:
                        if action:  # Skip empty actions
                            description_content += f"• {action}\n"
                else:
                    description_content += "Aucune action spécifiée\n"
                
                blocks = [{
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {
                                "content": description_content
                            }
                        }]
                    }
                }]
                
                # Append blocks to the page
                block_response = await self.append_block_children_tool.ainvoke({
                    "block_id": page_id,
                    "children": blocks
                })
                
                if not block_response:
                    logger.warning("Failed to append description block, but page was created")
                
                return page_id
                
            except Exception as e:
                logger.error(f"Error adding description block: {str(e)}")
                # Return the page_id anyway since the page was created
                return page_id
                
        except Exception as e:
            logger.error(f"Error creating timesheet entry: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def create_timesheet_entries(self, tasks: List['TaskWithTime']) -> List[str]:
        """
        Crée plusieurs entrées dans la base de données timesheet via le serveur MCP.
        
        Args:
            tasks: Liste des tâches à ajouter au timesheet
            
        Returns:
            Liste des IDs des pages créées
        """
        created_ids = []
        for task in tasks:
            if page_id := await self.create_timesheet_entry(task):
                created_ids.append(page_id)
        return created_ids