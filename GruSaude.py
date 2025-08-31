import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Aqui meu fio carregamos o documento excel e fazemos um pequeno tratameto para garantir que não de ruim.
df = pd.read_excel("alimentos.xlsx")
df["Energia(kcal)"] = pd.to_numeric(
    df["Energia(kcal)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Energia(kcal)"] = df["Energia(kcal)"].fillna(0)

# Respostas associadas às perguntas
perguntas = [
    "Quero 3 itens para tomar café da manhã de até 100 calorias",
    "Me recomende uma fruta",
    "Me indique um alimento saudável",
    "Sugira um lanche com menos de 200 calorias",
    "Quais carnes você recomenda?",
    "Preciso de algo doce para sobremesa"
]
# Intenções associadas a cada pergunta
intencoes = [
    "recomendar_por_calorias",
    "recomendar_por_categoria",
    "recomendar_geral",
    "recomendar_por_calorias",
    "recomendar_por_categoria",
    "recomendar_geral"
]

# Criar o modelo de linguagem com TF-IDF e SVM 
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, SVC(kernel="linear"))  # Usando SVM com kernel linear
model.fit(perguntas, intencoes) # Treinamento do modelo

# Função para responder perguntas
def responder(pergunta):
    intencao = model.predict([pergunta])[0]
# aqui fiz um exemplo se caso a intenção for saber calorias ( vai ser a mesma logica se caso quisermos saber outras coisas como vitamina c etc...)
    if intencao == "recomendar_por_calorias":
        match = re.search(r"(\d+)\s*cal", pergunta.lower())
        if match:
            limite = int(match.group(1))
            candidatos = df[df["Energia(kcal)"] <= limite]
            if not candidatos.empty:
                return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
            else:
                return "Não encontrei alimentos abaixo desse valor calórico."
        else:
            return "Você pode especificar o limite de calorias?"
#se caso a intencao da pergunta é categoria, significa que temos que montar um "filtro" para nossas respostas corresponderem corretamente ao que a peça falar.
    elif intencao == "recomendar_por_categoria": 
#Criei 2 categorias de exemplo para fruta e para carne. ( ver no excel as categorias, acho que são 15)
        if "fruta" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Fruta", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "carne" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Carne", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        else:
            return "Não encontrei essa categoria, tente outra (ex: fruta, carne, cereal)."
# aqui é tanto faz, nao importa a categoria dos alimentos. ( pensei em usar isso para tipo ( recomende algo para cafe da manhã, recomente isso para outra coisa )) pensar em como deixar isso mais filtrado, nem que seja uma coluna dizendo se é recomendado para que tipo de alimentação.
    elif intencao == "recomendar_geral":
        return df.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
    return "Não sei, desculpe." # se não saber a intenção , logo nao sabe de nada e retorna que nao sabe.

# Interação com o usuário
while True:
    pergunta_usuario = input("Pergunte algo (ou 'sair'): ")
    if pergunta_usuario.lower() == "sair":
        print("bye...")
        break
    print(responder(pergunta_usuario))
