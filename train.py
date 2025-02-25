import os
import torch
from docx import Document
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# 🔹 1. Función para cargar y dividir el texto en fragmentos
def cargar_y_dividir_documento(doc_path, tokenizer, max_length=512):
    doc = Document(doc_path)
    texto_completo = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
    
    tokens = tokenizer(texto_completo, truncation=False, return_tensors="pt")["input_ids"][0]
    fragmentos = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    fragmentos_texto = [tokenizer.decode(frag, skip_special_tokens=True) for frag in fragmentos]
    
    return fragmentos_texto

# 🔹 2. Configurar el modelo y el tokenizador
modelo_base = "meta-llama/Llama-2-7b-hf"  # Llama 2 de Hugging Face
tokenizer = AutoTokenizer.from_pretrained(modelo_base)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 🔹 3. Cargar y preparar datos
doc_path = "modelo/documento.docx"
fragmentos_texto = cargar_y_dividir_documento(doc_path, tokenizer)

dataset = Dataset.from_dict({
    "text": fragmentos_texto
})

# 🔹 4. Tokenización
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize_function, batched=True)

# 🔹 5. Cargar modelo y aplicar LoRA
model = AutoModelForCausalLM.from_pretrained(modelo_base, torch_dtype=torch.float32)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = PeftModel(model, peft_config)

# 🔹 6. Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="modelo/afinados",
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch"
)

# 🔹 7. Entrenar modelo con LoRA
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

print("🚀 Iniciando entrenamiento con LoRA...")
trainer.train()

# 🔹 8. Guardar el modelo ajustado
output_dir = "modelo/afinados"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ ¡Entrenamiento completado y modelo guardado!")
