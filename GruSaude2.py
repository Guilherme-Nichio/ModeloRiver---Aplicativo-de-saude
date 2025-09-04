import re # biblioteca para encontrar padrões ou identificar palavras / valores
import pandas as pd # biblioteca para ler o dataframe
from flask import Flask, render_template, request # flask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# define o aplicativo 
app = Flask(__name__)
# carrega os dados da tabela nutricional com o pandas
df = pd.read_excel("alimentos.xlsx")
#transforma as colunas de valor em numerico 
colunas_numericas = {
    "Energia(kcal)": "kcal", # define a unidade de cada coluna
    "Colesterol(mg)": "mg",
    "Proteína(g)": "g",
    "Carboidrato(g)": "g",
    "Vitamina C(mg)": "mg",
    "Lipídeos(g)": "g"
}
#para cada valor dentro de colunas_numericas
for coluna in colunas_numericas:
    df[coluna] = ( 
        pd.to_numeric(
            df[coluna].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        ).fillna(0) #transforma o valor em numerico, subsitui , por . e preenche os vazios com 0
    )
# perguntas para treinar o modelo
perguntas = [
    "Me indique lanches com menos de 150 calorias",
    "Sugira alimentos com poucas calorias",
    "Preciso de comidas leves até 100 cal",
    "3 refeições de até 120 calorias",
    "Ideias para lanche com baixa caloria",
    "Pode me sugerir alguma fruta?",
    "Tem alguma carne que você recomenda?",
    "Quero um alimento da categoria de cereal",
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
#intenções de cada pergunta
intencoes = (
    # as intenções estão dessa maneira para resumir o tamanho do codigo, como se estivesse escrito 5x a mesma intenção
    ["recomendar_por_calorias"] * 5 + 
    ["recomendar_por_categoria"] * 5 +
    ["recomendar_geral"] * 5 +
    ["recomendar_por_colesterol"] * 5 +
    ["recomendar_por_proteina"] * 5 +
    ["recomendar_por_carboidrato"] * 5 +
    ["recomendar_por_vitamina_c"] * 5 +
    ["recomendar_por_gordura"] * 5
)

# Criar o modelo de linguagem com TF-IDF e SVM
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, SVC(kernel="linear"))
model.fit(perguntas, intencoes)

# Se caso a intenção for recomentar por valor , tipo alimento por quantidade de calorias, ou alto, baixo em calorias ou proteinas.
def recomendar_por_valor(coluna, unidade, pergunta, limite_default_msg):
    pergunta_lower = pergunta.lower() # pega a pergunta do usuario e define para minusculo

    # Se caso na pergunta tiver o numero explicito de calorias ou outra caracteristica, ele procura nessa string pergunta_lower, que é a pergunta do usuario , e segue por qual tipo de unidade ele está.
    match = re.search(r"(\d+)\s*" + unidade, pergunta_lower) 
    if match: 
        limite = float(match.group(1))
        candidatos = df[df[coluna] <= limite]
        if not candidatos.empty:
            return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos", coluna]].to_html(index=False)
        return f"Não encontrei alimentos abaixo desse valor de {coluna.lower()}."

    # Se não tiver número tipo de calorias ou outra caracteristica, interpretar palavras qualitativas, como baixo , alto . para poder indicar os alimentos correspondentes.
    if any(p in pergunta_lower for p in ["baixo", "pouco", "reduzido"]):
        limite = df[coluna].quantile(0.3)  # 30% menores valores
        candidatos = df[df[coluna] <= limite]
        if not candidatos.empty:
            return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos", coluna]].to_html(index=False)
        return f"Não encontrei alimentos com baixo teor de {coluna.lower()}."

    if any(p in pergunta_lower for p in ["alto", "muito", "bastante", "rico"]):
        limite = df[coluna].quantile(0.7)  # 30% maiores valores
        candidatos = df[df[coluna] >= limite]
        if not candidatos.empty:
            return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos", coluna]].to_html(index=False)
        return f"Não encontrei alimentos ricos em {coluna.lower()}."

    # Se não encontrar nada
    return limite_default_msg

# se caso for uma pergunta de categoria, tipo me indique uma fruta, ele vai executar essa função;
def recomendar_por_categoria(pergunta):
    pergunta_lower = pergunta.lower()
    #aqui definimos as variações como fruta, frutas, carne ,carnes , para poder ter mais chance de acertar a forma que o usuario escrever na pergunta.
    categorias_map = {
        r"frut(as)?": "Fruta",
        r"carn(e|es)": "Carne",
        r"cereal(is)?": "Cereais",
        r"legum(e|es)": "Legume",
        r"verdur(as)?": "Verdura",
        r"pescad(o|os)": "Pescado",
        r"oleo(s)?": "Óleo",
        r"bebid(as)?": "Bebida",
        r"noz(es)?": "Nozes",
        r"ov(o|os)": "Ovo",
        r"leit(e|es)": "Leite",
        r"açuc(ar|ares)": "Açúcar",
        r"miscelane(a|as)": "Miscelânea",
        r"preparad(o|os|a|as)": "Preparado",
        r"outro(s)?": "Outros"
    }

    for cate, valor in categorias_map.items(): 
        # essa função do re.search procura o primeiro lugar do texto que o padrão aparece.
        if re.search(cate, pergunta_lower):
            candidatos = df[df["Categoria"].str.contains(valor, case=False, na=False)]
            if not candidatos.empty:
                # sample = sortear a linha aleatoria
                return candidatos.sample(3)[["Descrição dos alimentos", "Energia(kcal)"]].to_html(index=False) 
                #to_html siginifica que  converte o dataframe em uma tabela

    return "Não encontrei essa categoria, tente outra (ex: fruta, carne, cereal)."

# função principal para responder com base nas intenções;
def responder(pergunta):

    intencao = model.predict([pergunta])[0] # [0] para pegar somente o primeiro elemento da lista, ou seja se o usuario fazer mais de uma pergunta, retorna somente a primeira.

    #compara qual o tipo de intenção

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
        # essa é a função que 
        return recomendar_por_categoria(pergunta)
    elif intencao == "recomendar_geral":
        return df.sample(1)[["Descrição dos alimentos", "Energia(kcal)"]].to_html(index=False) # a função "sample" sorteia uma linha aletaoria 
    return "Não sei, desculpe."

#rota para o flask
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST": # se caso o formulario da pergunta for enviado 
        pergunta = request.form.get("pergunta")
        resposta = responder(pergunta) # faz a função responder com a pergunta do usuario
        return render_template("resposta.html", pergunta=pergunta,  resposta=resposta)# rendeniza no modelo do template de resposta
    return render_template("index.html")

# executa o servidor do flask para deixar a aplicação rodando
if __name__ == "__main__":
    app.run(debug=True)
