''' Este arquivo foi criado com o objetivo de extrair as informações de um documento PDF /
para que o modelo possa armazenar os chunks e transmitir a resposta para o usuário e para um arquivo .JSON.
Este projeto utiliza as seguintes funções e métodos:
    
    PyPDFLoader ==> Carrega o documento, indicado quando se pretende trabalhar com PDF.
    
    load_and_split ==> Realiza o carregamento do arquivo, retornando uma lista de listas, onde cada lista representa um /
        assunto do PDF. Melhor para situações onde o arquivo é maior, pois só armazena o texto dos assuntos relevantes.
    
    RecursiveCharacterTextSplitter ==> Indicada para arquivos com assuntos mais complexos, possui maior ajustabilidade.
        Para divisão básica em fragmentos e acesso direto ao texto com possibilidade de sobreposição /
        de fragmentos(chunk_overlap) para melhor aprendizagem.
    
    split_documents ==> Separa o conteúdo do PDF em chunks.
    
    FAISS.from_documents ==> Armazenar os chunks dentro do Vector DB.

    ConversationBufferWindowMemory ==> Armazena apenas as interações definidas pelo K, ideal quando o histórico/
        completo não é necessário, ou a memória é limitada.
        
    run ==> Execute o modelo, vai me retornar um objeto chain. Posteriormente será aproveitado na qa_chain.
    
    RetrievalQA ==> Responde a pergunta utilizando o modelo, maneira mais personalizavel.'''

import json
import timeit
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.memory import FileChatMessageHistory,ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from class_func import create_chat,create_embeddings,define_pastas,verify_pdf,create_prompts
from langchain.cache import InMemoryCache
from langchain.chains import RetrievalQA


llm_chat = create_chat()
llm_embeddings = create_embeddings()
file = define_pastas(r'data')
prompt = create_prompts()
print(file)

try:
    # O [0] é para não retornar como lista, apenas o nome
    data = verify_pdf(file)[0]
    print(data)
except (FileNotFoundError, ValueError) as e:
    print(e)

# Ou passar o caminho completo do arquivo, fazer a leitura e retornar como lista
loader = PyPDFLoader(f'{file}/{data}')
print(loader)

# O split é para dividir o documento em varios objetos Documents, cada um contendo um trecho e criar os chunks
# Ideal para documentos grandes, ou que tratam do mesmo assunto(Ideal para treinar LLM)
pages_load = loader.load_and_split()
print(pages_load)

# Criação do objeto RecursiveCharacterTextSplitter para gerar o chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Tamanho do chunk/texto que vai estar ali dentro
    chunk_size=1000,
    # Tamanho da sobreposição, conseguindo aproveitar palavras/frases que foram divididas
    chunk_overlap=200,
    # Medir o comprimento
    length_function=len,   
)

# Realizar a divisão do Texto e gerar os chunks
chunks = text_splitter.split_documents(pages_load)

# Realizar o armazenamento do tempo de resposta
# llm_cache = InMemoryCache / FAISS não trabalha com cache?

# Criação do objeto FAISS, para armazenar os chunks
db = FAISS.from_documents(
    documents=chunks,
    embedding=llm_embeddings,
     #cache = llm_cache / FAISS não aceita cache?
     )

query = "O que é uma imagem?"

# Realizar a busca por similirdade, retorna a quantidade especificada pelo K
contexto = db.similarity_search(query,k=3,return_metadata=True)

# Criação do objeto ConversationBufferWindowMemory para armazenar o chat
memory = ConversationBufferWindowMemory(
    # Armazena as respostas em um arquivo .json
    chat_memory=FileChatMessageHistory(file_path="historic_json/messages_load_split.json"),
    # Armazena o histórico da interação com o modelo
    memory_key="chat_history",
    # Chave do input
    input_key="query",
    # Quantidade de interações que serão "armazenadas" para continuar o contexto da conversa.
    k=2,
    # Traz a pesquisa por similaridade do conteúdo que gerou a resposta
    contexto=contexto,
    return_messages=True,
)

# Criação do objeto LLMChain, cadeia de conexão para gerar a resposta do LLM
llm_chain=LLMChain(
    # Modelo que será utilizado
    llm=llm_chat,
    # Definição do prompt utilizado
    prompt=prompt,
    # Retornar logs
    verbose=True,
    # Configuração da memória que será utilizada
    memory=memory,
)

# Iniciar contagem do tempo de resposta
start_time = timeit.default_timer()

# Gerar a resposta do modelo, vai me retornar um objeto chain. Posteriormente será aproveitado na qa_chain
# Método simples que responde a pergunta utilizando o modelo, maneira mais rápida e fácil.
response = llm_chain.run(query=query,contexto=contexto,memory=memory)
print(response)

# Cria um objeto RetrievalQA, utilizará o LLM para entender a pergunta
# Classe que permite personalizar o processo de Pergunta e resposta, dando mais controle sobre a qualidade e a precisão das respostas.
qa_chain = RetrievalQA.from_chain_type(
    # Modelo que será utilizado
    llm=llm_chat,
    # Tipo de cadeia para responder perguntas factuais
    chain_type="stuff",
    # Gerar os logs de resposta
    verbose=True,
    # Retornar os dois principais resultados
    retriever=db.as_retriever(search_kwargs={"k": 2})
)

# Finaliza a contagem do tempo de resposta
elapsed_time = timeit.default_timer() - start_time

print(qa_chain(query))
print(f"Executado em {elapsed_time} segundos")

# Gerar um arquivo .json personalizado
try:
    with open("historic_json/Historico_load_split.json", "r", encoding="utf-8") as arquivo:
        historico = json.load(arquivo)
except (json.JSONDecodeError, FileNotFoundError):
    historico = []

nova_entrada = {
    "Query": query,
    "Answer": response
}

historico.append(nova_entrada)

with open("historic_json/Historico_load_split.json", "w",encoding="utf-8") as arquivo:
    json.dump(historico, arquivo, ensure_ascii=False, indent=4)