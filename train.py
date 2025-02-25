import os
import torch
from docx import Document
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# 🔹 1. Función para cargar y dividir el texto en fragmentos manejables
def cargar_y_dividir_documento(doc_path, tokenizer, max_length=512):
    doc = Document(doc_path)
    texto_completo = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
    
    # Tokenizar todo el texto
    tokens = tokenizer(texto_completo, truncation=False, return_tensors="pt")["input_ids"][0]
    
    # Dividir el texto en fragmentos de `max_length` tokens
    fragmentos = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    # Convertir a strings nuevamente
    fragmentos_texto = [tokenizer.decode(frag, skip_special_tokens=True) for frag in fragmentos]
    
    return fragmentos_texto

# 🔹 2. Configurar el modelo y el tokenizador
modelo_base = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(modelo_base)

# Asegurar que el tokenizador tenga un token de padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 🔹 3. Cargar y preparar datos
doc_path = "modelo/documento.docx"
fragmentos_texto = cargar_y_dividir_documento(doc_path, tokenizer)

# Crear dataset con los fragmentos como entradas y el mismo texto como salida
dataset = Dataset.from_dict({
    "prompt": fragmentos_texto,
    "response": fragmentos_texto
})

# 🔹 4. Tokenizar datos
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"], 
        text_pair=examples["response"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

dataset = dataset.map(tokenize_function, batched=True)

# 🔹 5. Cargar modelo y aplicar LoRA
model = AutoModelForCausalLM.from_pretrained(modelo_base, torch_dtype=torch.float32)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = PeftModel(model, peft_config)

# 🔹 6. Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="modelo/salida",
    per_device_train_batch_size=1,  # 🔹 Ajustado para CPU
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch"
)

# 🔹 7. Inicializar entrenador y entrenar
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

trainer.train()

# 🔹 8. Guardar modelo ajustado
output_dir = "modelo/afinados"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ ¡Entrenamiento completado y modelo guardado!")
