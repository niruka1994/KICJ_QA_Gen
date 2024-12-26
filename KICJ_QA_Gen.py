import streamlit as st
import time
import os
import tiktoken
import pandas as pd
from io import BytesIO
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import KonlpyTextSplitter, RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT')


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


default_instruction = """
Exclude questions and answers related to tables, ratios, and numbers.
Exclude questions and answers related to the location of the page and the table of contents.
Restrict the question(s) to the context information provided only.
Based on the example below, you need to generate questions and answers in Korean.
"""

default_example = """
#EXAMPLE:
QUESTION: 변호사와 의뢰인 간 비밀유지권의 주요 개념은 무엇입니까?
ANSWER: 변호사와 의뢰인 간 비밀유지권은 변호사와 의뢰인 사이에 주고받은 의사소통에 관한 사실이나 자료가 법정에 제출 또는 공개되는 것을 거부할 수 있는 권리입니다.

QUESTION: 우리나라에서 비밀유지권 도입이 필요한 이유는 무엇입니까?
ANSWER: 비밀유지권 도입은 의뢰인의 권리보호와 글로벌 스탠다드에 맞춘 법치주의와 적법절차의 내실화를 위해 필요합니다.

QUESTION: 변호사법 제26조에 따르면 변호사의 비밀유지 의무는 어떻게 규정되어 있습니까?
ANSWER: 변호사법 제26조는 변호사가 직무상 알게 된 비밀을 유지해야 한다고 명시하고 있습니다.
"""

st.set_page_config(
    page_title="KICJ Q/A Gen",
    page_icon=":slightly_smiling_face:")

st.markdown("""
    <style>
        .sub-header {
            font-size: 18px;
            font-weight: bold;
            color: #696363;
            text-align: center;
        }
        .warning-text {
            font-size: 18px;
            color: red;
            font-weight: bold;
            text-align: center;
        }
        .center-text {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
        }
        .custom-divider-rainbow {
        margin: 2em 0;
        height: 2px;
        background: linear-gradient(to right, 
            #FF0000, #FFA500, #FFFF00, 
            #008000, #0000FF, #4B0082, #EE82EE);
        }
        .custom-divider-gray {
        margin: 2em 0;
        height: 2px;
        background: #e8e7e6;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="center-text">🙂 KICJ Q/A Gen</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider-rainbow"></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">이 프로그램은 GPT를 통하여 PDF 파일을 Q/A 데이터로 변환해주는 사이트 입니다.</div>', unsafe_allow_html=True)
st.markdown('<div class="warning-text">AI는 허위정보나 오류를 포함할 수 있으니, 항상 추가 검증을 진행하시기 바랍니다.</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider-rainbow"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Your PDF", type="pdf")

col1, col2 = st.columns([1, 1])
with col1:
    chunking_strategy = st.radio("Chunk Strategy (Optional)", ("Recursive", "Konlpy"),
                                 help="페이지를 일련의 묶음(청크)으로 분할하여 Q/A를 생성하게 됩니다. *Recursive = 문단, Konlpy = 형태소")

with col2:
    model_choice = st.radio("AI Model (Optional)", ("GPT-3.5-Turbo", "GPT-4o"))

if model_choice == 'GPT-3.5-Turbo':
    model_choice = 'gpt-3.5-turbo'
elif model_choice == 'GPT-4o':
    model_choice = 'gpt-4o'

input_chunk_size = st.slider("Chunk Size (Optional)", 500, 10000, 3000,
                             help="청크의 사이즈를 지정합니다.")
input_chunk_overlap = st.slider("Chunk Overlap Size (Optional)", 50, 1000, 300,
                                help="청크간 겹쳐지는 사이즈를 지정합니다.")
num_questions = st.slider("Q/A Per (Optional)", 1, 10, 3,
                          help="청크별 생성되는 Q/A 수를 지정합니다. *청크 내용이 부족한경우, 설정한 수에 도달하지 못할 수도 있습니다.")
custom_instruction = st.text_area("Instruction (Optional)", value=default_instruction,
                                  height=150, help="AI에게 지시할 제약조건을 서술할 수 있습니다.")
custom_example = st.text_area("Sample Q/A for AI (Optional)", value=default_example, height=150,
                              help="AI가 Q/A 생성에 참고할 예제를 서술할 수 있습니다. *예제의 내용보단, 문장의 형태(틀)를 참고합니다.")

if chunking_strategy == 'Recursive':
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap,
                                                   length_function=tiktoken_len)
elif chunking_strategy == 'Konlpy':
    text_splitter = KonlpyTextSplitter(chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap,
                                       length_function=tiktoken_len)

st.markdown('<div class="custom-divider-gray"></div>', unsafe_allow_html=True)

process_button = st.button("Process")

prompt_template = f"""
Context information is below. You are only aware of this context and nothing else.
---------------------

{{context}}

---------------------
Given this context, generate only questions based on the below query.
You are an Teacher/Professor in Legal.
Your task is to provide exactly **{{num_questions}}** question(s) for an upcoming quiz/examination.
You are not to provide more or less than this number of questions.
The question(s) should be diverse in nature across the document.
The purpose of question(s) is to test the understanding of the students on the context information provided.
{custom_instruction}
{custom_example}
"""

prompt = PromptTemplate.from_template(prompt_template)

if uploaded_file is not None and process_button:
    with st.spinner("Processing..."):
        output_dir = os.path.join(os.getcwd(), "PDF")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_folder_path = os.path.join(output_dir, timestamp)
        os.makedirs(new_folder_path, exist_ok=True)

        file_name = uploaded_file.name
        tmp_file_path = os.path.join(new_folder_path, file_name)

        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.read())

        pdf = PyPDFLoader(tmp_file_path)
        documents = pdf.load()

        combined_text = "".join(doc.page_content for doc in documents)
        split_documents = text_splitter.split_text(combined_text)

        llm = ChatOpenAI(
            model=model_choice,
            streaming=True,
            temperature=0
        )
        chain = (
                prompt |
                llm
        )

        placeholder = st.empty()

        qa_pairs = []

        for idx, document in enumerate(split_documents):
            if document:
                try:
                    response = chain.invoke(
                        {"context": document, "num_questions": num_questions}
                    )
                    st.markdown("**************************")
                    st.markdown(f"**Chunk {idx + 1}:**", help=document)
                    st.markdown(f"{response.content}")
                    chunk_text = document
                    qa_generated = response.content
                    vectorizer = CountVectorizer().fit_transform(
                        [chunk_text, qa_generated])
                    vectors = vectorizer.toarray()
                    cosine_sim_author = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
                    st.markdown(f"***청크와 생성된 Q/A의 코사인 유사도 : {cosine_sim_author}***")
                    qa_pairs.append(response)

                    with placeholder.container():
                        st.markdown(
                            f"""
                            <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>
                                <h4>Completed Chunk {idx + 1}:</h4>
                                <p>{response.content}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    time.sleep(1)
                except KeyError as e:
                    st.error(f"Error processing question {idx + 1}: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # 데이터 저장할 리스트 초기화
        chunk_data = []

        # 청크별 데이터를 추출하여 리스트에 저장
        for idx, response in enumerate(qa_pairs):
            chunk_context = split_documents[idx]
            qa_pairs_split = response.content.split('QUESTION:')

            for qa_pair in qa_pairs_split[1:]:
                question_answer = qa_pair.split('ANSWER:')
                if len(question_answer) == 2:
                    question = question_answer[0].strip()
                    answer = question_answer[1].strip()
                    chunk_data.append({
                        'Chunk 번호': idx + 1,
                        'Chunk Context': chunk_context,
                        'QUESTION': question,
                        'ANSWER': answer
                    })

        # pandas DataFrame 생성
        df = pd.DataFrame(chunk_data)

        # 엑셀 파일을 메모리에 저장하기 위한 BytesIO 객체 생성
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')

        # 엑셀 파일 다운로드 버튼 제공
        st.download_button(
            label="엑셀파일 저장(*저장시 본 페이지 초기화)",
            data=excel_buffer,
            file_name=f"Q_A_Generated_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
