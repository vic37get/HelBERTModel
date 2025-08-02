# HelBERT 👨‍⚖️

⚖📝 **HelBERT** é um repositório completo para pré-treinamento, fine-tuning, avaliação e análise de modelos de linguagem baseados em BERT, com foco em textos de Editais de Licitações e domínio jurídico-administrativo brasileiro.

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-treinamento do HelBERT](#pré-treinamento-do-helbert)
- [Fine-tuning e Avaliação](#fine-tuning-e-avaliação)
- [Como Executar](#como-executar)
- [Resultados e Métricas](#resultados-e-métricas)
- [HelBERT no Hugging Face](#helbert-no-hugging-face)

---

## Sobre o Projeto

Este repositório contém todo o pipeline para:
- **Pré-treinamento** de modelos BERT no domínio de licitações públicas.
- **Fine-tuning** para tarefas específicas (classificação, NER, etc).
- **Avaliação** e análise de métricas dos modelos.
- **Scripts utilitários** para limpeza, preparação de dados e visualização de resultados.

O HelBERT foi treinado do zero utilizando grandes volumes de editais e documentos públicos, visando melhorar o desempenho em tarefas jurídicas e administrativas.

---

## Estrutura do Repositório

A pasta principal de código é [`Codigos/`](Codigos/), organizada da seguinte forma:

- **BaseDeDados/**: Scripts de limpeza, preparação e manipulação de datasets.
- **CalculoMetricasHelBERTs/**: Cálculo e análise de métricas dos modelos.
- **Distilled/**: Técnicas de destilação de modelos.
- **FineTuning/**: Scripts para fine-tuning e avaliação em diferentes tarefas (Classificação, NER, etc).
- **Graficos/**: Geração de gráficos e visualizações.
- **LSG/**: Métodos para long sequence modeling.
- **PreTreinamento/**: Scripts para pré-treinamento do HelBERT.
- **utils/**: Funções utilitárias para manipulação de arquivos, métricas, etc.

Além disso, o repositório contém exemplos de datasets, modelos treinados e resultados de experimentos.

---

## Pré-treinamento do HelBERT

O pré-treinamento do HelBERT é realizado a partir de grandes corpora de editais, utilizando scripts em [`Codigos/PreTreinamento/`](Codigos/PreTreinamento/). O processo inclui:
- Limpeza e normalização dos textos ([`cleaner_pretreinamento.py`](Codigos/BaseDeDados/cleaner_pretreinamento.py))
- Tokenização e preparação dos dados
- Treinamento do modelo BERT com Masked Language Modeling

---

## Fine-tuning e Avaliação

O fine-tuning é realizado para diferentes tarefas, como:
- **Classificação de textos** ([`FineTuning/Classificacao/`](Codigos/FineTuning/Classificacao/))
- **Reconhecimento de Entidades Nomeadas (NER)** ([`FineTuning/EntidadesNomeadas/`](Codigos/FineTuning/EntidadesNomeadas/))
- **Outras tarefas específicas** (Fertilidade, Objetos, etc)

Scripts de avaliação e cálculo de métricas estão disponíveis em [`CalculoMetricasHelBERTs/`](Codigos/CalculoMetricasHelBERTs/) e [`FineTuning/*/`](Codigos/FineTuning/).

---

## Como Executar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt

2. **Prepare os dados:**

Utilize os scripts em Codigos/BaseDeDados/ para limpeza e preparação dos datasets.

3. **Pré-treine o modelo:**

Execute os scripts em Codigos/PreTreinamento/ para treinar o HelBERT do zero.

4. **Fine-tuning e avaliação:**

Utilize os scripts em Codigos/FineTuning/ para treinar e avaliar o modelo em tarefas específicas.

5. **Visualize os resultados:**

Gere gráficos e relatórios com os notebooks em Codigos/Graficos/ e Codigos/CalculoMetricasHelBERTs/.

---

## Resultados e Métricas

Os resultados dos experimentos, métricas de avaliação e comparações com outros modelos estão disponíveis em:

Codigos/CalculoMetricasHelBERTs/metricas_modelos.json

Notebooks de análise em Codigos/CalculoMetricasHelBERTs/

---

## HelBERT no Hugging Face 🤗

O modelo **HelBERT-base** está disponível publicamente no [Hugging Face Hub](https://huggingface.co/vic35get/HelBERT-base):

[![Hugging Face](https://img.shields.io/badge/HuggingFace-HelBERT--base-yellow?logo=huggingface)](https://huggingface.co/vic35get/HelBERT-base)

---

### Como utilizar o HelBERT-base

Você pode importar e utilizar o modelo diretamente em seu código Python com a biblioteca `transformers`:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("vic35get/HelBERT-base")
model = AutoModel.from_pretrained("vic35get/HelBERT-base")

# Exemplo de uso
inputs = tokenizer("Exemplo de texto jurídico para o HelBERT.", return_tensors="pt")
outputs = model(**inputs)

---