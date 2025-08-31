# AULA DO ROGERIO - A.I MODELO DE APRENDIZAGEM DE MAQUINA
# DUPLA: GUILHERME HENRIQUE & GUILHERME NICCHIO. 
# Recomendador de Alimentos por Perguntas

Este projeto é um sistema simples de recomendação de alimentos baseado em perguntas do usuário. Ele utiliza **Python**, **pandas**, **scikit-learn** (SVM e TF-IDF) e **regex** para processar e responder perguntas sobre alimentos com base em calorias, categorias ou recomendações gerais.

---

## Funcionalidades

- **Recomendar alimentos por calorias:** Sugere alimentos abaixo de um limite calórico especificado.
- **Recomendar alimentos por categoria:** Sugere alimentos de categorias específicas (como frutas ou carnes).
- **Recomendação geral:** Sugere alimentos aleatórios, independente da categoria ou calorias.
- **Interação em tempo real:** Permite ao usuário digitar perguntas e receber respostas instantâneas.
  
---

## Requisitos

- Bibliotecas Python:

```bash
pip install pandas scikit-learn openpyxl
