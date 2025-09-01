import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Aqui carregamos o documento excel e fazemos um pequeno tratameto para garantir que não de ruim.
df = pd.read_excel("alimentos.xlsx")
#logica para converter calorias
df["Energia(kcal)"] = pd.to_numeric(
    df["Energia(kcal)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Energia(kcal)"] = df["Energia(kcal)"].fillna(0)

#logica para converter colesterol apartir daqui fiz uma por uma para converter de cada categoria q vamos usar
df["Colesterol(mg)"] = pd.to_numeric(
    df["Colesterol(mg)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Colesterol(mg)"] = df["Colesterol(mg)"].fillna(0)

#PROTEINA
df["Proteína(g)"] = pd.to_numeric(
    df["Proteína(g)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Proteína(g)"] = df["Proteína(g)"].fillna(0)

#CARBOIDRATO
df["Carboidrato(g)"] = pd.to_numeric(
    df["Carboidrato(g)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Carboidrato(g)"] = df["Carboidrato(g)"].fillna(0)

#VITAMINA C
df["Vitamina C(mg)"] = pd.to_numeric(
    df["Vitamina C(mg)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Vitamina C(mg)"] = df["Vitamina C(mg)"].fillna(0)

#GORDURA
df["Lipídeos(g)"] = pd.to_numeric(
    df["Lipídeos(g)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce" 
)
df["Lipídeos(g)"] = df["Lipídeos(g)"].fillna(0)

# Respostas associadas às perguntas
perguntas = [
    #perguntas calorias
    "Me indique lanches com menos de 150 calorias",
    "Sugira alimentos com poucas calorias",
    "Preciso de comidas leves até 100 cal",
    "3 refeições de até 120 calorias",
    "Ideias para lanche com baixa caloria",
    #perguntas categorias
    "Pode me sugerir alguma fruta?",
    "Tem alguma carne que você recomenda?",
    "Quero um alimento da categoria de cereais",
    "Me dá uma sugestão de legume saudável?",
    "O que você tem de bom na categoria de verduras?",
    #perguntas gerais
    "Me recomenda algum alimento?",
    "Sugira algo saudável pra comer",
    "O que você indica para hoje?",
    "Me dá uma ideia do que posso comer",
    "Tem alguma sugestão de comida?",
    #perguntas colesterol
    "Quais alimentos têm pouco colesterol?",
    "Me dá sugestões com menos de 100 mg de colesterol",
    "Tem alguma comida com baixo colesterol?",
    "Preciso evitar colesterol, o que posso comer?",
    "Indica algo leve com até 80 mg de colesterol?",
    #pergunta proteina
    "Quais alimentos são ricos em proteína?",
    "Me recomenda algo com bastante proteína",
    "Preciso de alimentos com alta proteína",
    "Me indica algo com até 10g de proteína",
    "Tem alguma opção com boa quantidade de proteína?",
    #perguntas carboidrato
    "Quais alimentos são ricos em carboidratos?",
    "Me sugere algo com bastante carboidrato",
    "Preciso de algo com até 15g de carboidrato",
    "Tem algum alimento com poucos carboidratos?",
    "Recomenda um lanche com baixo carbo?",
    #perguntas vitamina C
    "Quais alimentos são ricos em vitamina C?",
    "Me sugere algo com bastante vitamina C",
    "Preciso de algo com até 30mg de vitamina C",
    "Tem alguma opção com boa quantidade de vitamina C?",
    "Indica algo leve com até 20mg de vitamina C?",
    #perguntas godura
    "Quais alimentos têm muita gordura?",
    "Me indica algo com pouca gordura",
    "Preciso de opções com menos de 5g de gordura",
    "Sugere um alimento com baixo teor de gordura",
    "Tem alguma comida com gordura saudável?"

]
# Intenções associadas a cada pergunta
intencoes = [
    ["recomendar_por_calorias"] * 5,                       #inteções para cada tipo x5
    ["recomendar_por_categoria"] * 5,
    ["recomendar_geral"] * 5,
    ["recomendar_por_colesterol"] * 5,
    ["recomendar_por_proteina"] * 5,
    ["recomendar_por_carboidrato"] * 5,
    ["recomendar_por_vitamina_c"] * 5,
    ["recomendar_por_gordura"] * 5

]

#qui fazemos o "flatten" para que intencoes fique 1D, conforme o esperado pelo sklearn, achatei a lista
intencoes = [item for sublist in intencoes for item in sublist]

# Criar o modelo de linguagem com TF-IDF e SVM (para a maquina conseguir converter texto em vetores numericos para entender)
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, SVC(kernel="linear"))  # Usando SVM com kernel linear
model.fit(perguntas, intencoes) # Treinamento do modelo

# Função para responder perguntas
def responder(pergunta):
    intencao = model.predict([pergunta])[0]
    #categorias de recomendação no aprendizado 
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

    elif intencao == "recomendar_por_colesterol":
        match = re.search(r"(\d+)\s*mg", pergunta.lower())
        if match:
            limite = int(match.group(1))
            candidatos = df[df["Colesterol(mg)"] <= limite]
            if not candidatos.empty:
                return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos","Colesterol(mg)"]].to_string(index=False)
            else:
                return "Não encontrei alimentos abaixo desse valor de colesterol."
        else:
            return "Você pode especificar o limite de colesterol?"

    elif intencao == "recomendar_por_proteina":
        match = re.search(r"(\d+)\s*g", pergunta.lower())
        if match:
            limite = int(match.group(1))
            candidatos = df[df["Proteina(g)"] <= limite]
            if not candidatos.empty:
                return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos","Proteina(g)"]].to_string(index=False)
            else:
                return "Não encontrei alimentos abaixo desse valor de proteína."
        else:
            return "Você pode especificar o limite de proteína?"

    elif intencao == "recomendar_por_carboidrato":
        match = re.search(r"(\d+)\s*g", pergunta.lower())
        if match:
            limite = int(match.group(1))
            candidatos = df[df["Carboidrato(g)"] <= limite]
            if not candidatos.empty:
                return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos","Carboidrato(g)"]].to_string(index=False)
            else:
                return "Não encontrei alimentos abaixo desse valor de carboidrato."
        else:
            return "Você pode especificar o limite de carboidrato?"

    elif intencao == "recomendar_por_vitamina_c":
        match = re.search(r"(\d+)\s*mg", pergunta.lower())
        if match:
            limite = int(match.group(1))
            candidatos = df[df["Vitamina C(mg)"] <= limite]
            if not candidatos.empty:
                return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos","Vitamina C(mg)"]].to_string(index=False)
            else:
                return "Não encontrei alimentos abaixo desse valor de vitamina C."
        else:
            return "Você pode especificar o limite de vitamina C?"

    elif intencao == "recomendar_por_gordura":
        match = re.search(r"(\d+)\s*g", pergunta.lower())
        if match:
            limite = int(match.group(1))
            candidatos = df[df["Gordura(g)"] <= limite]
            if not candidatos.empty:
                return candidatos.sample(min(3, len(candidatos)))[["Descrição dos alimentos","Li"]].to_string(index=False)
            else:
                return "Não encontrei alimentos abaixo desse valor de gordura."
        else:
            return "Você pode especificar o limite de gordura?"

#se caso a intencao da pergunta é categoria, significa que temos que montar um "filtro" para nossas respostas corresponderem corretamente ao que a peça falar.
    elif intencao == "recomendar_por_categoria": 
#Criar um filtro para cada categoria da planilha
        if "fruta" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Fruta", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "carne" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Carne", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "cereal" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Cereais", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "legume" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Legume", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "oleo" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Óleo", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "bebida" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Bebida", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "verdura" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Verdura", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "pescados" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Pescado", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "leite" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Leite", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "ovo" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Ovo", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "açucar" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Açúcar", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "miscelanea" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Miscelânea", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "outros" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Outros", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "preparado" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Preparado", case=False, na=False)]
            return candidatos.sample(1)[["Descrição dos alimentos","Energia(kcal)"]].to_string(index=False)
        elif "nozes" in pergunta.lower():
            candidatos = df[df["Categoria"].str.contains("Nozes", case=False, na=False)]
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