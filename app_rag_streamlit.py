# C:\streamlit\app_rag_streamlit.py
import os, re, time
import streamlit as st
import tiktoken
from loguru import logger

# --- LangChain 0.2.x 계열 ---
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
try:
    # 폴백: 추출률 높은 로더(설치 필요: pymupdf)
    from langchain_community.document_loaders import PyMuPDFLoader
except Exception:
    PyMuPDFLoader = None

try:
    from langchain_community.document_loaders import Docx2txtLoader
except Exception:
    Docx2txtLoader = None
try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
except Exception:
    UnstructuredPowerPointLoader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

import google.generativeai as genai

# =========================
# Utils
# =========================
def tiktoken_len(text: str) -> int:
    tok = tiktoken.get_encoding("cl100k_base")
    return len(tok.encode(text))

def _chars(docs):
    return sum(len(d.page_content or "") for d in docs)

def _preview(text, n=300):
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")

def load_pdf_any(path):
    """PDF 텍스트 로딩: PyPDF → (비어있으면) PyMuPDF 폴백"""
    docs = []
    try:
        docs = PyPDFLoader(path).load_and_split()
    except Exception as e:
        logger.warning(f"PyPDFLoader 실패: {e}")

    if _chars(docs) < 50 and PyMuPDFLoader is not None:
        try:
            docs = PyMuPDFLoader(path).load_and_split()
            logger.info("PyMuPDFLoader 폴백 사용")
        except Exception as e:
            logger.warning(f"PyMuPDFLoader 실패: {e}")
    return docs

def load_docs(files):
    """업로드 파일 저장 후 페이지 단위로 로드(+PDF 폴백)"""
    docs = []
    for f in files:
        name = f.name
        with open(name, "wb") as o:
            o.write(f.getvalue())
        lower = name.lower()
        cur = []
        if lower.endswith(".pdf"):
            cur = load_pdf_any(name)
        elif lower.endswith(".docx") and Docx2txtLoader:
            cur = Docx2txtLoader(name).load_and_split()
        elif lower.endswith(".pptx") and UnstructuredPowerPointLoader:
            cur = UnstructuredPowerPointLoader(name).load_and_split()
        else:
            logger.warning(f"Unsupported or missing dependency for: {name}")
            continue

        docs.extend(cur)
        # 사이드바에 per-file 통계 출력
        st.sidebar.write(f"📄 **{name}** → pages: {len(cur)}, chars: {_chars(cur)}")
        if lower.endswith(".pdf") and _chars(cur) < 50:
            st.sidebar.warning("이 PDF는 텍스트가 거의 없습니다. 스캔본(OCR 필요) 가능성이 큽니다.")
    return docs

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=120, length_function=tiktoken_len
    )
    return splitter.split_documents(documents)

def build_vector(chunks):
    emb = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, emb)

# =========================
# 모델 탐색/선택 (+429 폴백)
# =========================
def discover_models():
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    names = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or getattr(m, "generation_methods", [])
        if "generateContent" in methods:
            names.append(m.name.split("/")[-1])
    return names

def pick_model_dynamic():
    avail = discover_models()

    forced_list = [m.strip() for m in os.getenv("GEMINI_MODEL_LIST", "").split(",") if m.strip()]
    if forced_list:
        forced_list = [m for m in forced_list if m in avail]
        if forced_list:
            return forced_list

    # Gemma/Pali/-exp 제외
    filtered = [m for m in avail if not (m.startswith("gemma") or m.startswith("pali") or "-exp" in m)]

    # 1.5/프로 우선
    ordered = []
    pref_15 = [r"gemini-1\.5-flash-.*", r"gemini-1\.5-pro-.*", r"gemini-1\.5-flash", r"gemini-1\.5-pro", r"gemini-pro"]
    for pat in pref_15:
        for m in filtered:
            if re.fullmatch(pat, m) and m not in ordered:
                ordered.append(m)

    # 없으면 2.x(쿼터 주의)
    pref_2x = [r"gemini-2\.0-flash-.*", r"gemini-2\.0-pro-.*", r"gemini-2\.5-flash-.*",
               r"gemini-2\.0-flash", r"gemini-2\.0-pro", r"gemini-2\.5-flash", r"gemini-2\.5-pro"]
    if not ordered:
        for pat in pref_2x:
            for m in filtered:
                if re.fullmatch(pat, m) and m not in ordered:
                    ordered.append(m)

    if not ordered:
        st.sidebar.error(f"발견된 모델(참고): {avail}")
        raise RuntimeError("이 KEY로 사용할 수 있는 Gemini 텍스트 모델을 찾지 못했습니다.")
    return ordered

def make_llm_with_fallback():
    tried = []
    candidates = pick_model_dynamic()
    for model_id in candidates:
        try:
            llm = ChatGoogleGenerativeAI(model=model_id, temperature=0, max_output_tokens=512)
            _ = llm.invoke("ping").content  # 최소 호출로 가용성 확인
            return llm, model_id, candidates
        except ResourceExhausted:
            tried.append((model_id, "quota_exhausted")); continue
        except Exception as e:
            tried.append((model_id, f"{type(e).__name__}")); continue
    raise RuntimeError(f"사용 가능한 모델이 없습니다. 시도 내역: {tried}")

def get_conversation_chain(vs):
    llm, picked, cand = make_llm_with_fallback()
    # 입력 토큰 절약 + 검색 정확도 보완(최소 2개 가져와서 MMR)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 6})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=False,
    )
    return chain, picked, cand

# =========================
# 스로틀 + 백오프
# =========================
def throttle(min_interval=2.5):
    st.session_state.setdefault("last_ts", 0.0)
    now = time.time()
    if now - st.session_state["last_ts"] < min_interval:
        st.warning(f"요청 간격을 {min_interval}초 이상으로 해주세요.")
        st.stop()
    st.session_state["last_ts"] = now

def backoff_call(fn, tries=3, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except ResourceExhausted:
            wait = base ** i
            st.warning(f"429(리밋). {wait:.1f}s 대기 후 재시도…")
            time.sleep(wait)
    return fn()

# =========================
# App
# =========================
def main():
    st.set_page_config(page_title="Streamlit_RAG", page_icon="📚")

    # secrets → env
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError:
        pass

    st.title("_Private Data :red[Q/A Chat]_ 📚")

    st.session_state.setdefault("messages", [{"role": "assistant", "content": "안녕하세요! 문서 업로드 후 Process를 눌러주세요."}])
    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("model_id", None)
    st.session_state.setdefault("candidates", [])
    st.session_state.setdefault("ready", False)

    with st.sidebar:
        files = st.file_uploader("Upload files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        api_key = st.text_input("Google API Key (옵션: secrets.toml 사용 시 비워두기)", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        if st.button("Process"):
            st.session_state["ready"] = False
            st.session_state["conversation"] = None
            st.session_state["vectorstore"] = None
            st.session_state["model_id"] = None
            st.session_state["candidates"] = []

            if not os.environ.get("GOOGLE_API_KEY"):
                st.error("Google API Key가 필요합니다 (secrets.toml 또는 입력)."); st.stop()
            if not files:
                st.error("최소 1개 파일을 업로드하세요."); st.stop()

            raw_docs = load_docs(files)
            total_chars = _chars(raw_docs)
            st.sidebar.write(f"🔎 total pages: {len(raw_docs)}, total chars: {total_chars}")
            if total_chars < 50:
                st.sidebar.error("문서에서 텍스트를 거의 추출하지 못했습니다. 스캔본일 수 있습니다(OCR 필요).")

            if not raw_docs:
                st.error("읽을 수 있는 문서가 없습니다."); st.stop()

            chunks = split_docs(raw_docs)
            st.sidebar.write(f"chunks: {len(chunks)}")
            if chunks:
                st.sidebar.write("preview:", _preview(chunks[0].page_content))

            vs = build_vector(chunks)
            try:
                chain, model_id, cand = get_conversation_chain(vs)
            except Exception as e:
                st.error("LLM 초기화 실패"); st.exception(e); st.stop()

            st.session_state["vectorstore"] = vs
            st.session_state["conversation"] = chain
            st.session_state["model_id"] = model_id
            st.session_state["candidates"] = cand
            st.session_state["ready"] = True
            st.success(f"준비 완료! 모델: **{model_id}**")

    with st.sidebar.expander("🔧 Diagnostics"):
        st.write("GOOGLE_API_KEY:", "✅" if os.environ.get("GOOGLE_API_KEY") else "❌")
        st.write("Ready:", st.session_state["ready"])
        st.write("Selected model:", st.session_state["model_id"])
        if st.session_state["candidates"]:
            st.write("Candidates:", st.session_state["candidates"])

    # 메시지 렌더
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 질의
    if q := st.chat_input("질문을 입력하세요."):
        if not st.session_state["ready"]:
            st.warning("먼저 파일 업로드 후 Process를 눌러주세요."); st.stop()

        throttle(2.5)
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            try:
                result = backoff_call(lambda: st.session_state["conversation"]({"question": q}))
            except Exception as e:
                st.error("호출 오류"); st.exception(e); st.stop()

            answer = result.get("answer", "")
            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

            src = result.get("source_documents") or []
            if src:
                with st.expander("참고 문서"):
                    for i, d in enumerate(src[:3], 1):
                        src_path = d.metadata.get("source", f"doc_{i}")
                        st.markdown(f"- **{src_path}**")
                        st.caption(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

if __name__ == "__main__":
    main()
