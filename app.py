# Install all libraries by running in the terminal: pip install -r requirements.txt
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from docx import Document
import PyPDF2
import os
import tempfile
import tiktoken


# fetch environmental variables
load_dotenv()

# helper function to process the input text file, remove empty lines and unneeded formatting marks
def process_input_file(input_file_path):
    '''
    process_input_text() helper function takes the input file in txt, docx or pdf format
    as an argument and removes empty lines and non-essential characters. The output is saved
    in a temporary directory.
    
    Parameters:
        input_file_path (str): path to the input text file
    
    Returns:
        processed temporary text file path saved in temp/
    '''
    # Create a temporary file in the same directory as the input file
    temp_dir = os.path.join(os.path.dirname(input_file_path), "temp")
    os.makedirs(temp_dir, exist_ok = True)

    temp_file = tempfile.NamedTemporaryFile(mode = 'w', delete = False, dir = temp_dir, encoding = 'UTF-8')

    try:
        file_extension = os.path.splitext(input_file_path)[1].lower()

        # Read the contents of the file based on its type
        if file_extension == '.txt':
            with open(input_file_path, 'r', encoding='UTF-8') as input_file:
                lines = input_file.readlines()
        elif file_extension == '.docx':
            doc = Document(input_file_path)
            lines = [p.text for p in doc.paragraphs]
        elif file_extension == '.pdf':
            with open(input_file_path, 'rb') as input_file:
                reader = PyPDF2.PdfFileReader(input_file)
                lines = []
                for page_num in range(reader.numPages):
                    page = reader.getPage(page_num)
                    lines.append(page.extract_text())
        else:
            raise ValueError("Unsupported file format: " + file_extension)

        # Remove empty lines and lines consisting only of '-' or '_'
        non_empty_lines = [line.strip() for line in lines if line.strip() and not all(char in {'-', '_'} for char in line.strip())]

        # Write processed text to the temporary file
        temp_file.write('\n'.join(non_empty_lines))
    finally:
        # Close the temporary file
        temp_file.close()

    # Get the path of the temporary file
    temp_file_path = temp_file.name

    return temp_file_path


# helper function to ask questions to LLM chain
def answer_question(q, chain):
    '''
    answer_question() is a helper function to ask a single question from
    a LLM chain
    
    Parameters:
        q (str): user's question
        crc (langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain):
            ConversationalRetrievalChain object from Langchain
    
    '''
    result = chain.invoke({'question': q})
    return result['answer']


def ask_and_get_answer(vector_store, q, k = 5):
     # initialize LLM
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model = os.getenv("OPENAI_DEPLOYMENT_NAME"), 
        temperature = 0)
    # Configure vector store to act as a retriever (finding similar items, returning top 5)
    retriever = vector_store.as_retriever(
        search_type = 'similarity', 
        search_kwargs={'k': k})
    # Create a memory buffer to track the conversation
    memory = ConversationBufferMemory(
        memory_key = 'chat_history', 
        return_messages = True)

    chain = RetrievalQA.from_chain_type(
        llm = llm, 
        chain_type = "stuff", 
        retriever = retriever)

    answer = chain.invoke(q)
    return answer['result']


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    '''
    load_documents() is a helper function to load txt file
    as langchain documents
    
    Parameters:
        file (str): path to file
    '''
    try:
        loader = TextLoader(file, encoding = 'UTF-8')
    except:
        print("TextLoader failed to load the text from load_documents function")
    
    data = loader.load()
    return data


# calculate embedding cost using tiktoken
def calculate_input_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000000 * 10


# splitting data in chunks
def chunk_data(data, chunk_size = 1024, chunk_overlap = 80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(
        model=os.getenv("TEXT_EMBEDDING_MODEL"), 
        dimensions=1536)  # 512 works as well
    # Create an in-memory Chroma vector store using the provided text chunks 
    # and the embedding model 
    vector_store = Chroma.from_documents(
        documents = chunks, 
        embedding = embeddings)
    return vector_store



# main app function
def main(input_text_file, remove_temp_file = True):
    '''
    main() function takes the text file with content and allows the user
    to ask questions through command line, then prints out the answer to each 
    question.
    
    Parameters:
        input_text_file (str): path to the input text file with content
        remove_temp_file (bool): if True temporary file with processed content
            is deleted from temp directory
    
    Returns:
        printed output with the GPT answer to each question
    '''
    # process input file
    processed_text_file_path = process_input_file(input_text_file)
    
    data = load_document(processed_text_file_path)
    if data is None:
        print(f"Failed to load document: {processed_text_file_path}")
    
    # split the text using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 80)
    chunks = text_splitter.split_documents(data)
    
    # Instantiate an embedding model from AzureOpenAI
    embeddings = OpenAIEmbeddings(
        model = os.getenv("TEXT_EMBEDDING_MODEL"), 
        dimensions = 1536)  

    # Create an in-memory Chroma vector store using the provided text chunks 
    # and the embedding model 
    vector_store = Chroma.from_documents(documents = chunks, embedding = embeddings)
    
    # initialize the Azure LLM
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  
        model = os.getenv("OPENAI_DEPLOYMENT_NAME"), 
        temperature=0)  

    # Configure vector store to act as a retriever (finding similar items, returning top 5)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    # Create a memory buffer to track the conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # build messages
    system_template = r'''
    You are answering questions only concerning the provided content of the input document.  
    If you are asked a question that is not related to the document you response will be:
    'The question is not relevant to the domain of interest'.
    ---------------
    Context: ```{context}```
    '''

    user_template = '''
    Answer questions only concerning the provided content of the input document.  
    If you are asked a question that is not related to the document you response will be:
    'The question is not relevant to the domain of interest'. 
    Here is the user's question: ```{question}```
    '''

    messages= [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
        ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)
    
    # Set up conversational retrieval chain
    crc = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        chain_type = 'stuff',
        combine_docs_chain_kwargs = {'prompt': qa_prompt },
        verbose = False)
    

    while True:
        new_user_question = input("Please ask your question about the document. If you want to stop type 'exit'.\n")
        if new_user_question.lower() == "exit":
            print("You ended the program.  Goodbye!")
            # removing temporary file with processed data if remove_temp_file==True
            if remove_temp_file:
                os.remove(processed_text_file_path)
            break
        else:
            response = answer_question(q = new_user_question, chain = crc)
            print(f"Answering question: {new_user_question}")
            # print the response
            print(f"Here is the response: \n")
            print(response, "\n")


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    st.image("imgage2.png")
    st.subheader("Document Knowledge Retrieval Chatbot 🤖")

    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=1024, on_change=clear_history)

        # k number (top results retrieved from text for LLM) input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_input_embedding_cost(chunks)
                st.write(f'Source document embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')


    # user's question text input widget
    question = st.text_input('Ask a question about the content of your file:')
    if question: # if the user entered a question and hit enter
        # build messages
        system_template = r'''
        You are answering questions only concerning the provided content of the input document.  
        If you are asked a question that is not related to the document you response will be:
        'The question is not relevant to the domain of interest'.
        ---------------
        Context: ```{context}```
        '''

        user_template = '''
        Answer questions only concerning the provided content of the input document.  
        If you are asked a question that is not related to the document you response will be:
        'The question is not relevant to the domain of interest'. 
        Here is the user's question: ```{question}```
        '''

        messages= [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
            ]

        qa_prompt = ChatPromptTemplate.from_messages(messages)
        
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            # Configure vector store to act as a retriever (finding similar items, returning top 5)
        
            st.write(f'k: {k}')
            #answer = ask_and_get_answer(vector_store, q, k)
            # initialize LLM
            llm = ChatOpenAI(
                api_key = os.getenv("OPENAI_API_KEY"),  
                model = os.getenv("OPENAI_DEPLOYMENT_NAME"), 
                temperature = 0)
            retriever = vector_store.as_retriever(
                search_type = 'similarity', 
                search_kwargs={'k': k})
            # Create a memory buffer to track the conversation
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            # Set up conversational retrieval chain
            crc = ConversationalRetrievalChain.from_llm(
                llm = llm,
                retriever = retriever,
                memory = memory,
                chain_type = 'stuff',
                combine_docs_chain_kwargs = {'prompt': qa_prompt },
                verbose = False)
            
            response = crc.invoke({'question': question})

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value = response)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./chat_with_documents.py

