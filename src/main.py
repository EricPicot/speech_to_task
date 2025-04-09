import os
import gradio as gr
from datetime import datetime
from scipy.io.wavfile import write as write_wav
import numpy as np
import speech_recognition as sr
import time
import sys
from task_extractor import TaskExtractor
from task_agent_extractor import TaskAgentExtractor

# Contrôle du niveau de verbosité des logs
DEBUG_MODE = False

def set_debug_mode(enable_debug):
    """Active ou désactive le mode debug avec logs détaillés."""
    global DEBUG_MODE
    DEBUG_MODE = enable_debug
    return f"Mode debug {'activé' if enable_debug else 'désactivé'}"

# Rediriger les logs stdout vers une chaîne de caractères pour l'interface
class LogCapture:
    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        
    def start(self):
        sys.stdout = self
        
    def stop(self):
        sys.stdout = self.original_stdout
        
    def write(self, text):
        self.logs.append(text)
        self.original_stdout.write(text)
        
    def flush(self):
        self.original_stdout.flush()
        
    def get_logs(self):
        return ''.join(self.logs)
        
    def clear(self):
        self.logs = []

# Créer un capteur de logs
log_capture = LogCapture()

# Assurer que les dossiers nécessaires existent
os.makedirs('audio_input', exist_ok=True)
os.makedirs('transcripts', exist_ok=True)

# Initialiser l'extracteur de tâches avec les projets adaptés à vos besoins réels
PROJECTS = ["Site Web", "Application Mobile", "Base de données", "Marketing", 
           "Documentation", "Support Client", "Développement", "Réunion", "Formation", 
           "Formation personnelle", "Mon marché recommandation"]

# Créer les deux types d'extracteurs
task_extractor = TaskExtractor(projects=PROJECTS)
task_agent_extractor = TaskAgentExtractor(projects=PROJECTS)

def process_audio(audio_data, use_agent=False, enable_debug=False):
    """
    Traite l'audio enregistré :
    1. Sauvegarde le fichier WAV
    2. Transcrit l'audio
    3. Sauvegarde la transcription
    4. Extrait les tâches
    
    Args:
        audio_data: Données audio enregistrées
        use_agent: Booléen indiquant s'il faut utiliser l'agent LangGraph
        enable_debug: Activer les logs détaillés
        
    Retourne un résumé des opérations.
    """
    # Configurer le mode debug
    set_debug_mode(enable_debug)
    
    # Démarrer la capture des logs si le mode debug est activé
    if DEBUG_MODE:
        log_capture.clear()
        log_capture.start()
    
    if audio_data is None:
        return "❌ Aucun audio enregistré"

    try:
        # Afficher un message de démarrage
        status = "🔄 Traitement audio en cours...\n\n"
        yield status
        
        # 1. Sauvegarder l'audio
        status += "⏳ Sauvegarde de l'audio...\n"
        yield status
        
        sample_rate, audio_array = audio_data
        
        # Créer un timestamp unique pour lier l'audio et la transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Préparer le chemin du fichier audio
        filename = f"recording_{timestamp}.wav"
        audio_filepath = os.path.join("audio_input", filename)
        
        # Assurer que l'audio est au format 16 bits pour WAV
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)
            
        # Sauvegarder l'audio
        write_wav(audio_filepath, sample_rate, audio_array)
        status += f"✅ Audio sauvegardé: {audio_filepath}\n\n"
        yield status
        
        # 2. Transcrire l'audio
        status += "⏳ Transcription de l'audio en texte...\n"
        yield status
        
        # Initialiser le recognizer
        recognizer = sr.Recognizer()
        
        # Charger l'audio sauvegardé
        with sr.AudioFile(audio_filepath) as source:
            audio = recognizer.record(source)
            
        try:
            # Obtenir la transcription
            transcript_text = recognizer.recognize_google(audio, language="fr-FR")
            
            # 3. Sauvegarder la transcription
            transcript_filename = f"transcript_{timestamp}.txt"
            transcript_filepath = os.path.join("transcripts", transcript_filename)
            
            with open(transcript_filepath, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            status += f"✅ Transcription réussie et sauvegardée: {transcript_filepath}\n\n"
            status += f"📝 Texte transcrit:\n{transcript_text}\n\n"
            yield status
            
            # Variable pour stocker le résultat de l'extraction des tâches
            tasks_summary = ""
            
            # 4. Extraire les tâches si du texte a été transcrit
            if transcript_text:
                try:
                    # Choisir l'extracteur en fonction du paramètre use_agent
                    if use_agent:
                        # Utiliser l'agent LangGraph
                        status += "⏳ Extraction des tâches avec l'agent LangGraph (parallélisé)...\n"
                        yield status
                        
                        try:
                            start_time = time.time()
                            status += "⏳ Initialisation du graphe LangGraph...\n"
                            yield status
                            
                            tasks = task_agent_extractor.extract_tasks(transcript_text)
                            end_time = time.time()
                            
                            status += f"✅ Traitement terminé en {end_time - start_time:.2f} secondes\n"
                            yield status
                            
                            if tasks:
                                status += f"✅ Extraction réussie! {len(tasks)} tâches trouvées.\n\n"
                                yield status
                                
                                tasks_summary = task_agent_extractor.format_tasks_summary(tasks)
                                status += f"Nombre de tâches extraites: {len(tasks)}\n\n"
                                yield status  # Yield status before adding task summary
                                
                                # Add task summary and yield again to ensure it's displayed
                                status += tasks_summary
                                yield status
                            else:
                                status += "⚠️ Aucune tâche n'a pu être extraite de cette transcription par l'agent.\n\n"
                                yield status
                        except Exception as extract_error:
                            status += f"❌ Erreur spécifique lors de l'extraction des tâches avec l'agent LangGraph: {str(extract_error)}\n"
                            import traceback
                            error_traceback = traceback.format_exc()
                            status += f"Détails de l'erreur:\n{error_traceback}\n\n"
                            yield status
                    else:
                        # Utiliser l'extracteur simple
                        status += "⏳ Extraction des tâches avec l'extracteur simple...\n"
                        yield status
                        
                        start_time = time.time()
                        tasks = task_extractor.extract_tasks(transcript_text)
                        end_time = time.time()
                        
                        if tasks:
                            status += f"✅ Extraction réussie en {end_time - start_time:.2f} secondes\n\n"
                            tasks_summary = task_extractor.format_tasks_summary(tasks)
                            status += f"Nombre de tâches extraites: {len(tasks)}\n\n"
                            yield status  # Yield status before adding task summary
                            
                            # Add task summary and yield again to ensure it's displayed
                            status += tasks_summary
                            yield status
                        else:
                            status += "⚠️ Aucune tâche n'a pu être extraite de cette transcription.\n\n"
                            yield status
                except Exception as e:
                    status += f"❌ Erreur lors de l'extraction des tâches: {str(e)}\n\n"
                    import traceback
                    error_traceback = traceback.format_exc()
                    status += f"Détails de l'erreur:\n{error_traceback}\n\n"
                    yield status
            
            # Retourner un résumé complet
            extraction_method = "agent LangGraph" if use_agent else "extracteur simple"
            final_status = f"""✅ Traitement audio terminé:
• Audio: {audio_filepath}
• Transcription: {transcript_filepath}
• Méthode d'extraction: {extraction_method}

📝 Texte transcrit:
{transcript_text}

{tasks_summary}"""
            
            # Ajouter les logs détaillés si le mode debug est activé
            if DEBUG_MODE:
                log_capture.stop()
                final_status += "\n\n--- Logs de débogage ---\n"
                final_status += log_capture.get_logs()
            
            # Yield one more time before returning the final result
            yield final_status
            return final_status
            
        except sr.UnknownValueError:
            return f"❌ Impossible de comprendre l'audio\nAudio sauvegardé: {audio_filepath}"
        except sr.RequestError as e:
            return f"❌ Erreur de service: {str(e)}\nAudio sauvegardé: {audio_filepath}"
            
    except Exception as e:
        if DEBUG_MODE:
            log_capture.stop()
        error_msg = f"❌ Erreur lors du traitement: {str(e)}"
        import traceback
        error_traceback = traceback.format_exc()
        detailed_msg = f"{error_msg}\n\n{error_traceback}"
        
        if DEBUG_MODE:
            detailed_msg += "\n\n--- Logs de débogage ---\n"
            detailed_msg += log_capture.get_logs()
            
        return detailed_msg

def analyze_transcript(transcript_text, use_agent=False, enable_debug=False):
    """
    Analyse une transcription existante pour extraire les tâches.
    
    Args:
        transcript_text: Texte de la transcription à analyser
        use_agent: Booléen indiquant s'il faut utiliser l'agent LangGraph
        enable_debug: Activer les logs détaillés
        
    Returns:
        Un résumé formaté des tâches extraites
    """
    # Configurer le mode debug
    set_debug_mode(enable_debug)
    
    # Démarrer la capture des logs si le mode debug est activé
    if DEBUG_MODE:
        log_capture.clear()
        log_capture.start()
        
    if not transcript_text:
        return "❌ Aucun texte à analyser"
    
    # Initialiser le statut avec la transcription pour l'afficher immédiatement
    status = f"📝 Texte à analyser:\n{transcript_text}\n\n"
    yield status
    
    try:
        # Choisir l'extracteur en fonction du paramètre use_agent
        if use_agent:
            # Utiliser l'agent LangGraph
            status += "⏳ Extraction des tâches avec l'agent LangGraph (parallélisé)...\n"
            yield status
            
            try:
                start_time = time.time()
                status += "⏳ Initialisation du graphe LangGraph...\n"
                yield status
                
                tasks = task_agent_extractor.extract_tasks(transcript_text)
                end_time = time.time()
                
                status += f"✅ Traitement terminé en {end_time - start_time:.2f} secondes\n"
                yield status
                
                if tasks:
                    status += f"✅ Extraction réussie! {len(tasks)} tâches trouvées.\n\n"
                    yield status
                    
                    tasks_summary = task_agent_extractor.format_tasks_summary(tasks)
                    status += tasks_summary
                    yield status
                else:
                    status += "⚠️ Aucune tâche n'a pu être extraite de cette transcription par l'agent."
                    yield status
            except Exception as extract_error:
                status += f"❌ Erreur spécifique lors de l'extraction des tâches avec l'agent LangGraph: {str(extract_error)}\n"
                import traceback
                error_traceback = traceback.format_exc()
                status += f"Détails de l'erreur:\n{error_traceback}\n\n"
                yield status
        else:
            # Utiliser l'extracteur simple
            status += "⏳ Extraction des tâches avec l'extracteur simple...\n"
            yield status
            
            start_time = time.time()
            tasks = task_extractor.extract_tasks(transcript_text)
            end_time = time.time()
            
            if tasks:
                status += f"✅ Extraction réussie en {end_time - start_time:.2f} secondes\n\n"
                tasks_summary = task_extractor.format_tasks_summary(tasks)
                status += tasks_summary
            else:
                status += "⚠️ Aucune tâche n'a pu être extraite de cette transcription."
                
        # Ajouter les logs détaillés si le mode debug est activé
        if DEBUG_MODE:
            log_capture.stop()
            status += "\n\n--- Logs de débogage ---\n"
            status += log_capture.get_logs()
            
        return status
        
    except Exception as e:
        if DEBUG_MODE:
            log_capture.stop()
        error_msg = f"❌ Erreur lors de l'analyse: {str(e)}"
        status += error_msg
        import traceback
        error_traceback = traceback.format_exc()
        status += f"\n\nDétails de l'erreur:\n{error_traceback}\n\n"
        
        if DEBUG_MODE:
            status += "\n\n--- Logs de débogage ---\n"
            status += log_capture.get_logs()
            
        yield status
        return status

# Interface Gradio
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎤 Enregistreur Audio et Extraction de Tâches")
        gr.Markdown("Enregistrez votre voix pour transcrire et extraire automatiquement vos tâches, ou analysez directement un texte.")
        
        with gr.Row():
            debug_checkbox = gr.Checkbox(
                label="Mode debug", 
                value=False,
                info="Afficher les logs détaillés (utile pour le débogage)"
            )
            debug_status = gr.Textbox(label="Statut du mode debug", value="Mode debug désactivé", interactive=False)
            
        debug_checkbox.change(
            fn=set_debug_mode,
            inputs=[debug_checkbox],
            outputs=[debug_status]
        )
        
        with gr.Tab("Enregistrement Audio"):
            with gr.Row():
                use_agent_audio = gr.Checkbox(
                    label="Utiliser l'agent LangGraph (plus robuste, avec heures de début/fin)", 
                    value=True,
                    info="Recommandé pour une meilleure précision, mais peut prendre plus de temps"
                )
            
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="numpy", 
                label="🎧 Audio",
                elem_id="audio_input"
            )
            
            result = gr.Textbox(
                label="Résultats", 
                lines=20,
                elem_id="result_box",
                show_copy_button=True
            )
            
            # Add a refresh button to manually update the UI if needed
            refresh_btn = gr.Button("Rafraîchir l'affichage")
            
            # Appeler process_audio automatiquement dès qu'un nouvel enregistrement est effectué
            process_event = audio_input.change(
                fn=process_audio, 
                inputs=[audio_input, use_agent_audio, debug_checkbox], 
                outputs=result
            )
            
            # Add a dummy function for refresh
            def refresh_output(current_text):
                return current_text
                
            refresh_btn.click(
                fn=refresh_output,
                inputs=[result],
                outputs=[result]
            )
        
        with gr.Tab("Analyse de Texte"):
            with gr.Row():
                use_agent_text = gr.Checkbox(
                    label="Utiliser l'agent LangGraph (plus robuste, avec heures de début/fin)", 
                    value=True,
                    info="Recommandé pour une meilleure précision, mais peut prendre plus de temps"
                )
                
            text_input = gr.Textbox(
                label="Texte à analyser", 
                lines=5, 
                placeholder="Entrez le texte de votre transcription ici...",
                value="""alors aujourd'hui j'ai travaillé 30 minutes pour récapituler ce que j'ai fait hier sur le projet mon marché recommandation et ensuite j'ai fait 2h30 de travail intensif pour créer une application speech to text qui me permet de plus rapidement traquer les tâches que j'ai fait ça il faut l'ajouter au projet formation personnelle est le résultat de cette étape c'est d'avoir une application radio d'avoir une transcription du speech vers le texte qui fonctionne et qui s'enregistre automatiquement"""
            )
            analyze_btn = gr.Button("Analyser")
            text_result = gr.Textbox(
                label="Tâches Extraites", 
                lines=20,
                show_copy_button=True
            )
            
            analyze_btn.click(
                fn=analyze_transcript, 
                inputs=[text_input, use_agent_text, debug_checkbox], 
                outputs=text_result
            )
            
            # Add refresh button for text analysis
            text_refresh_btn = gr.Button("Rafraîchir l'affichage")
            text_refresh_btn.click(
                fn=refresh_output,
                inputs=[text_result],
                outputs=[text_result]
            )
        
        gr.Markdown("""
        ## À propos
        Cette application utilise :
        - **Speech Recognition** pour la transcription audio
        - **LangChain** et **LangGraph** pour l'extraction parallèle des tâches
        - **Gradio** pour l'interface utilisateur
        
        Le code source est disponible sur GitHub.
        """)
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)