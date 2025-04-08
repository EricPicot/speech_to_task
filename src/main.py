import os
import gradio as gr
from datetime import datetime
from scipy.io.wavfile import write as write_wav
import numpy as np
import speech_recognition as sr
from task_extractor import TaskExtractor

# Assurer que les dossiers nécessaires existent
os.makedirs('audio_input', exist_ok=True)
os.makedirs('transcripts', exist_ok=True)

# Initialiser l'extracteur de tâches avec les projets adaptés à vos besoins réels
PROJECTS = ["Site Web", "Application Mobile", "Base de données", "Marketing", 
           "Documentation", "Support Client", "Développement", "Réunion", "Formation", 
           "Formation personnelle", "Mon marché recommandation"]

# Créer l'extracteur de tâches
task_extractor = TaskExtractor(projects=PROJECTS)

def process_audio(audio_data):
    """
    Traite l'audio enregistré :
    1. Sauvegarde le fichier WAV
    2. Transcrit l'audio
    3. Sauvegarde la transcription
    4. Extrait les tâches
    
    Retourne un résumé des opérations.
    """
    if audio_data is None:
        return "❌ Aucun audio enregistré"

    try:
        # 1. Sauvegarder l'audio
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
        
        # 2. Transcrire l'audio
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
            
            # Variable pour stocker le résultat de l'extraction des tâches
            tasks_summary = ""
            
            # 4. Extraire les tâches si du texte a été transcrit
            if transcript_text:
                try:
                    # Extraire les tâches
                    tasks = task_extractor.extract_tasks(transcript_text)
                    
                    # Formater le résumé des tâches
                    if tasks:
                        tasks_summary = "\n\n" + task_extractor.format_tasks_summary(tasks)
                    else:
                        tasks_summary = "\n\nAucune tâche n'a pu être extraite de cette transcription."
                except Exception as e:
                    tasks_summary = f"\n\n❌ Erreur lors de l'extraction des tâches: {str(e)}"
            
            # Retourner un résumé avec les deux filepaths et la transcription
            return f"""✅ Traitement audio réussi:
• Audio: {audio_filepath}
• Transcription: {transcript_filepath}

Texte: {transcript_text}{tasks_summary}"""
            
        except sr.UnknownValueError:
            return f"❌ Impossible de comprendre l'audio\nAudio sauvegardé: {audio_filepath}"
        except sr.RequestError as e:
            return f"❌ Erreur de service: {str(e)}\nAudio sauvegardé: {audio_filepath}"
            
    except Exception as e:
        return f"❌ Erreur lors du traitement: {str(e)}"

def analyze_transcript(transcript_text):
    """
    Analyse une transcription existante pour extraire les tâches.
    """
    if not transcript_text:
        return "❌ Aucun texte à analyser"
    
    try:
        # Extraire les tâches
        tasks = task_extractor.extract_tasks(transcript_text)
        
        # Formater le résumé
        if tasks:
            return task_extractor.format_tasks_summary(tasks)
        else:
            return "Aucune tâche n'a pu être extraite de cette transcription."
    except Exception as e:
        return f"❌ Erreur lors de l'analyse: {str(e)}"

# Interface Gradio
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎤 Enregistreur Audio et Extraction de Tâches")
        gr.Markdown("Enregistrez votre voix pour transcrire et extraire automatiquement vos tâches, ou analysez directement un texte.")
        
        with gr.Tab("Enregistrement Audio"):
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="🎧 Audio")
            result = gr.Textbox(label="Résultats", lines=12)
            
            # Appeler process_audio automatiquement dès qu'un nouvel enregistrement est effectué
            audio_input.change(fn=process_audio, inputs=audio_input, outputs=result)
        
        with gr.Tab("Analyse de Texte"):
            text_input = gr.Textbox(label="Texte à analyser", lines=5, 
                                    placeholder="Entrez le texte de votre transcription ici...",
                                    value="""alors aujourd'hui j'ai travaillé 30 minutes pour récapituler ce que j'ai fait hier sur le projet mon marché recommandation et ensuite j'ai fait 2h30 de travail intensif pour créer une application speech to text qui me permet de plus rapidement traquer les tâches que j'ai fait ça il faut l'ajouter au projet formation personnelle est le résultat de cette étape c'est d'avoir une application radio d'avoir une transcription du speech vers le texte qui fonctionne et qui s'enregistre automatiquement""")
            analyze_btn = gr.Button("Analyser")
            text_result = gr.Textbox(label="Tâches Extraites", lines=12)
            
            analyze_btn.click(fn=analyze_transcript, inputs=text_input, outputs=text_result)
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)