import re
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# --- Carregar e tratar dados ---
df = pd.read_excel("alimentos.xlsx")

colunas_numericas = {
    "Energia(kcal)": "kcal",
    "Colesterol(mg)": "mg",
    "Proteína(g)": "g",
    "Carboidrato(g)": "g",
    "Vitamina C(mg)": "mg",
    "Lipídeos(g)": "g"
}
for coluna in colunas_numericas:
    df[coluna] = (
        pd.to_numeric(
            df[coluna].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        ).fillna(0)
    )

# --- Dados de treinamento ---
perguntas = [
    "Me indique lanches com menos de 150 calorias",
    "Sugira alimentos com poucas calorias",
    "Preciso de comidas leves até 100 cal",
    "3 refeições de até 120 calorias",
    "Ideias para lanche com baixa caloria",
    "Pode me sugerir alguma fruta?",
    "Tem alguma carne que você recomenda?",
    "Quero um alimento da categoria de cereais",
    "Me dá uma sugestão de legume saudável?",
    "O que você tem de bom na categoria de verduras?",
    "Me recomenda algum alimento?",
    "Sugira algo saudável pra comer",
    "O que você indica para hoje?",
    "Me dá uma ideia do que posso comer",
    "Tem alguma sugestão de comida?",
    "Quais alimentos têm pouco colesterol?",
    "Me dá sugestões com menos de 100 mg de colesterol",
    "Tem alguma comida com baixo colesterol?",
    "Preciso evitar colesterol, o que posso comer?",
    "Indica algo leve com até 80 mg de colesterol?",
    "Quais alimentos são ricos em proteína?",
    "Me recomenda algo com bastante proteína",
    "Preciso de alimentos com alta proteína",
    "Me indica algo com até 10g de proteína",
    "Tem alguma opção com boa quantidade de proteína?",
    "Quais alimentos são ricos em carboidratos?",
    "Me sugere algo com bastante carboidrato",
    "Preciso de algo com até 15g de carboidrato",
    "Tem algum alimento com poucos carboidratos?",
    "Recomenda um lanche com baixo carbo?",
    "Quais alimentos são ricos em vitamina C?",
    "Me sugere algo com bastante vitamina C",
    "Preciso de algo com até 30mg de vitamina C",
    "Tem alguma opção com boa quantidade de vitamina C?",
    "Indica algo leve com até 20mg de vitamina C?",
    "Quais alimentos têm muita gordura?",
    "Me indica algo com pouca gordura",
    "Preciso de opções com menos de 5g de gordura",
    "Sugere um alimento com baixo teor de gordura",
    "Tem alguma comida com gordura saudável?"
]
intencoes = (
    ["recomendar_por_calorias"] * 5 +
    ["recomendar_por_categoria"] * 5 +
    ["recomendar_geral"] * 5 +
    ["recomendar_por_colesterol"] * 5 +
    ["recomendar_por_proteina"] * 5 +
    ["recomendar_por_carboidrato"] * 5 +
    ["recomendar_por_vitamina_c"] * 5 +
    ["recomendar_por_gordura"] * 5
)

# Modelo
model = make_pipeline(TfidfVectorizer(), SVC(kernel="linear"))
model.fit(perguntas, intencoes)

# --- Funções auxiliares ---
def recomendar_por_valor(coluna, unidade, pergunta, limite_default_msg):
    match = re.search(r"(\d+)\s*" + unidade, pergunta.lower())
    if match:
        limite = int(match.group(1))
        candidatos = df[df[coluna] <= limite]
        if not candidatos.empty:
            return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos", coluna]].to_html(index=False)
        return f"Não encontrei alimentos abaixo desse valor de {coluna.lower()}."
    return limite_default_msg

def recomendar_por_categoria(pergunta):
    categorias_map = {
        "fruta": "Fruta",
        "carne": "Carne",
        "cereal": "Cereais",
        "legume": "Legume",
        "oleo": "Óleo",
        "bebida": "Bebida",
        "verdura": "Verdura",
        "pescados": "Pescado",
        "leite": "Leite",
        "ovo": "Ovo",
        "açucar": "Açúcar",
        "miscelanea": "Miscelânea",
        "outros": "Outros",
        "preparado": "Preparado",
        "nozes": "Nozes"
    }
    for chave, valor in categorias_map.items():
        if chave in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains(valor, case=False, na=False)]
            if not candidatos.empty:
                return candidatos.sample(1)[["Descrição dos alimentos", "Energia(kcal)"]].to_html(index=False)
    return "Não encontrei essa categoria, tente outra (ex: fruta, carne, cereal)."

def responder(pergunta):
    intencao = model.predict([pergunta])[0]

    if intencao == "recomendar_por_calorias":
        return recomendar_por_valor("Energia(kcal)", "cal", pergunta, "Você pode especificar o limite de calorias?")
    elif intencao == "recomendar_por_colesterol":
        return recomendar_por_valor("Colesterol(mg)", "mg", pergunta, "Você pode especificar o limite de colesterol?")
    elif intencao == "recomendar_por_proteina":
        return recomendar_por_valor("Proteína(g)", "g", pergunta, "Você pode especificar o limite de proteína?")
    elif intencao == "recomendar_por_carboidrato":
        return recomendar_por_valor("Carboidrato(g)", "g", pergunta, "Você pode especificar o limite de carboidrato?")
    elif intencao == "recomendar_por_vitamina_c":
        return recomendar_por_valor("Vitamina C(mg)", "mg", pergunta, "Você pode especificar o limite de vitamina C?")
    elif intencao == "recomendar_por_gordura":
        return recomendar_por_valor("Lipídeos(g)", "g", pergunta, "Você pode especificar o limite de gordura?")
    elif intencao == "recomendar_por_categoria":
        return recomendar_por_categoria(pergunta)
    elif intencao == "recomendar_geral":
        return df.sample(1)[["Descrição dos alimentos", "Energia(kcal)"]].to_html(index=False)
    return "Não sei, desculpe."

# --- Rotas ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pergunta = request.form.get("pergunta")
        resposta = responder(pergunta)
        return render_template("resposta.html", pergunta=pergunta, resposta=resposta)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
