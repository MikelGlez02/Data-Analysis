from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import torch

class RecipeSummarizer:
    def __init__(self, model_name: str = 'google/pegasus-cnn_dailymail'): # Inicia el resumidor con modelo preentrenado
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # Cargamos tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device) # Cargamos modelo
        self.summarizer = pipeline('summarization', model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1 ) # Configurar pipeline de resumen
    
    def summarize_instructions(self, instructions: List[str], max_length: int = 150, min_length: int = 30) -> str: # Resúmen de las instrucciones de una receta
        if not instructions:
            return ""
        full_text = ' '.join(instructions) # Se combina las instrucciones en una sola cadena de texto
        summary = self.summarizer(full_text, max_length=max_length, min_length=min_length, do_sample=False,truncation=True)
        return summary[0]['summary_text']
    
    def batch_summarize(self, recipes: List[Dict], batch_size: int = 8) -> List[str]: # Resúmenes de un lote de recetas
        summaries = []
        for i in range(0, len(recipes), batch_size):
            batch = recipes[i:i+batch_size]
            batch_texts = [' '.join(recipe.get('directions', [])) for recipe in batch]
            batch_summaries = self.summarizer(batch_texts,max_length=150,min_length=30,do_sample=False,truncation=True,batch_size=batch_size) # Resúmenes del lote actual
            summaries.extend([s['summary_text'] for s in batch_summaries])
        return summaries

def load_summarizer(model_name: str) -> RecipeSummarizer: # Se carga un resumidor dependiendo de qué modelo se ha puesto en los argumentos
    available_models = {'BART': 'facebook/bart-large-cnn','T5': 't5-small','LED': 'allenai/led-large-16384','PEGASUS': 'google/pegasus-cnn_dailymail'}
    if model_name not in available_models:
        raise ValueError(f"Modelo de resumen no soportado: {model_name}. Opciones: {list(available_models.keys())}")
    return RecipeSummarizer(available_models[model_name])
