# 📘 LlamaIndex

> **기반 실습**: `1__Llama_index.ipynb`  
> **핵심 데이터**: Paul Graham Essay (`paul_graham_essay.txt`)  
> **사용 모델**: OpenAI `gpt-3.5-turbo` (Embedding: `text-embedding-ada-002`)

---

## 🗺️ 전체 파이프라인 개요

```
[Raw Files]
    ↓  SimpleDirectoryReader
[Documents]          ← 파일 단위 객체
    ↓  VectorStoreIndex.from_documents()
    ├─ SentenceSplitter  →  [Nodes]       ← 청크 단위 객체
    └─ Embedding Model   →  [Vectors]     ← 벡터 저장소
[VectorStoreIndex]
    ↓  as_query_engine() / as_retriever()
[Query Engine]
    ├─ Retriever      →  관련 Nodes 검색 (벡터 유사도)
    └─ Synthesizer    →  LLM으로 최종 답변 생성
```

---

## 1. 환경 설정 및 라이브러리

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # OpenAI API 키 등록

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    PromptTemplate
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.node_parser import SentenceSplitter

from openai import OpenAI
import openai
```

> **설치 명령어**
> ```bash
> pip install llama_index==0.12.2 openai==1.55.3 packaging>=24.0
> ```

---

## 2. 데이터 로딩 — `SimpleDirectoryReader`

### 기본 사용법

```python
# "data/" 폴더 내 모든 파일을 읽어 Document 리스트로 변환
documents = SimpleDirectoryReader("data").load_data()
```

### 반환값 구조

```
documents = [
    Document(paul_graham_essay.txt),   ← documents[0]
    Document(another_file.pdf),        ← documents[1]  (파일이 더 있다면)
    ...
]
```

- `documents`는 Python **리스트(list)**
- **파일 1개 = Document 1개**
- `documents[0]` → 첫 번째 파일의 전체 내용

### Document 속성

```python
doc = documents[0]

doc.text        # 파일의 전체 텍스트 내용 (str)
doc.doc_id      # 자동 생성된 고유 ID (UUID)
doc.metadata    # {'file_name': '...', 'file_path': '...', ...}
```

### 지원 파일 형식

| 확장자 | 설명 |
|--------|------|
| `.txt` | 텍스트 파일 |
| `.pdf` | PDF 문서 |
| `.docx` | Microsoft Word |
| `.csv` | CSV 데이터 |
| `.md` | 마크다운 |
| `.ipynb` | Jupyter Notebook |
| `.hwp` | 한글 문서 |
| `.jpeg/.png` | 이미지 |
| `.mp3/.mp4` | 오디오/비디오 |

### 고급 옵션

```python
# 특정 파일만 지정 로드
documents = SimpleDirectoryReader(
    input_files=["data/essay.txt", "data/report.pdf"]
).load_data()

# 특정 확장자만 로드
documents = SimpleDirectoryReader(
    "data",
    required_exts=[".txt", ".pdf"]
).load_data()

# 특정 파일 제외
documents = SimpleDirectoryReader(
    "data",
    exclude=["data/temp.txt"]
).load_data()
```

---

## 3. 인덱스 생성 — `VectorStoreIndex`

### 기본 사용법

```python
index = VectorStoreIndex.from_documents(documents)
```

### 내부 동작 흐름

```
documents
    │
    ▼
SentenceSplitter (기본 chunk_size=1024, chunk_overlap=200)
    │
    ├──→ Node 1 (텍스트 청크)
    ├──→ Node 2
    └──→ Node N
         │
         ▼ (각 Node마다 자동 실행)
    OpenAI Embedding (text-embedding-ada-002)
         │
         ▼
    1536차원 벡터
         │
         ▼
    VectorStore (메모리 딕셔너리에 저장)
```

### 외부 Vector DB 없이 로컬 메모리 사용

```python
# 외부 DB 미연결 시 → 로컬 메모리의 딕셔너리 구조로 저장
index = VectorStoreIndex.from_documents(documents)
```

### Node 구조 확인

```python
# 인덱스 내 노드 ID 목록 확인
node_id_dict = index.index_struct.nodes_dict

for key, value in node_id_dict.items():
    print(key, value)  # key == value == UUID 형태
    node_example_id = value
    break

print("총 노드 수:", len(node_id_dict))
# 기본 설정(chunk_size=1024)에서 paul_graham_essay.txt → 22개 노드
```

### 임베딩 벡터 확인

```python
import numpy as np

# 특정 노드의 임베딩 벡터 추출
embedding_vector = index.vector_store.data.embedding_dict[node_example_id]

arr = np.array(embedding_vector)
print(len(arr))  # 1536 (text-embedding-ada-002의 차원 수)
```

---

## 4. Node 분할 — `SentenceSplitter`

### 개념

```
Document (전체 파일)
    │
    ▼  SentenceSplitter
    ├── Node 0  [0 ~ 1024 token]
    ├── Node 1  [824 ~ 1848 token]  ← overlap: 200 token 겹침
    ├── Node 2  [1648 ~ 2672 token]
    └── ...
```

- `chunk_size`: 각 노드의 **최대 토큰 수**
- `chunk_overlap`: 인접 노드 간 **겹치는 토큰 수** (문맥 연속성 유지)
- 분할 기준은 **문자(character) 수가 아닌 토큰(token) 수**

### 직접 Node 생성

```python
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = parser.get_nodes_from_documents(documents)

print(len(nodes))          # 22 (기본 설정 기준)
print(nodes[0].text)       # 첫 번째 청크 텍스트
print(nodes[1].text)       # 두 번째 청크 텍스트 (overlap 부분 포함)
print(len(nodes[0].text))  # 문자 수 (토큰 수와 다름!)
```

### Index 생성 시 chunk 파라미터 커스터마이징

```python
text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=50)

index = VectorStoreIndex.from_documents(
    documents=documents,
    transformations=[text_splitter]
)
# chunk_size=200 → 노드 수 증가 (153개)
# chunk_size=1024 → 노드 수 적음 (22개)
```

### 토크나이저로 실제 토큰 수 확인

```python
import tiktoken

# SentenceSplitter의 기본 토크나이저 (gpt-3.5-turbo용)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

encoded_text = tokenizer.encode(nodes[0].text)
print(f"토큰 수: {len(encoded_text)}")   # ≈ chunk_size와 유사 (메타데이터 포함 계산)

decoded_text = tokenizer.decode(encoded_text)
print(f"복원 텍스트:\n{decoded_text}")
```

> **⚠️ 주의**: 실제 토큰 수가 `chunk_size`와 정확히 일치하지 않을 수 있음.
> LlamaIndex는 노드 생성 시 **메타데이터**도 포함해 토큰을 계산하기 때문.

---

## 5. Query Engine — `index.as_query_engine()`

### 기본 사용법

```python
# 인덱스에서 Query Engine 직접 생성
query_engine = index.as_query_engine()

response = query_engine.query("What is the first programs the author tried writing?")
print(response)
# → "The first programs the author tried writing were on the IBM 1401..."
```

### 내부 구조

```
query_engine.query("질문")
    │
    ├─ [1] Retriever: 질문을 임베딩 → 벡터 유사도로 관련 Node 검색 (기본 2개)
    │
    └─ [2] Response Synthesizer: 검색된 Node 텍스트 + 질문 → LLM (gpt-3.5-turbo) → 답변
```

### Query Engine의 한계: 외부 데이터 범위 벗어난 질문

```python
# 데이터와 무관한 질문 → LLM의 사전학습(파라미터) 지식으로 답변 가능성 있음
response = query_engine.query("How many countries participated in the production of the space station?")
print(response)
# → 외부 데이터에 없으므로 부정확하거나 hallucination 가능
```

### OpenAI LLM 직접 호출 (비교용)

```python
def generate_answer(question):
    messages = [{"role": "user", "content": question}]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=messages,
    )
    return response.choices[0].message.content

# RAG 없이 순수 LLM만 사용
print(generate_answer("How many countries participated in the production of the space station?"))
# → "15" (LLM의 사전학습 지식)
```

---

## 6. Retriever — `index.as_retriever()`

### 기본 사용법

```python
retriever = index.as_retriever()

# 쿼리와 유사한 Node를 벡터 검색으로 가져옴
ret_passages = retriever.retrieve("Who is the author?")

for i, passage in enumerate(ret_passages):
    print(f"### Passage {i}\n", passage.text)
```

### 검색 결과 분석

```python
# 검색된 모든 텍스트를 하나로 합치기
passages_text = ""
for passage in ret_passages:
    passages_text += passage.text

# 특정 키워드 포함 여부 확인
if "paul graham" in passages_text.lower():
    print("'Paul Graham'이 검색 결과에 포함됨")
else:
    print("'Paul Graham'이 검색 결과에 없음")
```

### Retriever로 RAG 프롬프트 직접 구성

```python
ret_context = "".join([r.text for r in ret_passages])

question = f"""Context information is below.
---------------------
{ret_context}
---------------------
Given the context information and not prior knowledge,
answer the query and the reasons of answer.
Query: Who is the author?
Answer:
"""

print(generate_answer(question))
```

> **⚠️ 인사이트**: Retriever가 "Paul Graham"이라는 단어가 포함된 Node를 찾지 못해도,
> LLM이 간접적인 문맥(첫 인칭 서술, 생애 회고 등)으로 저자를 추론할 수 있음.
> → **파라미터 지식(LLM 사전학습)**과 **검색된 문맥** 모두 활용됨.

---

## 7. 데이터베이스 관리 — Insert / Update / Delete

### 7-1. Insert — 새 데이터 추가

```python
# 새 Document 객체 생성
new_data = "Natural diamonds were formed in the upper mantle of Earth at temperatures of 900–1,400 °C and at pressures of 5–6 GPa."
docu = Document(text=new_data, id_="new_doc_id")  # id_ 직접 지정 가능

# 인덱스에 삽입
index.insert(docu)

# 새 Query Engine 생성 (인덱스 갱신 반영)
new_query_engine = index.as_query_engine()

question = "What are the temperature and air pressure conditions under which natural diamonds are produced?"
res = new_query_engine.query(question)
print(res)
# → "900–1,400°C and 5–6 GPa" (삽입된 데이터 기반 응답)
```

### 7-2. Update — 기존 데이터 수정

#### 방법 1: `update_ref_doc` (단일 Document 업데이트)

```python
# Document 내용 변경
docu.set_content(value="... temperatures of 2,000–6,000 °C and at pressures of 8–9 GPa.")

# 동일 id_의 데이터를 업데이트 (기존 docstore도 삭제)
index.update_ref_doc(
    docu,
    update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
)

query_engine_v2 = index.as_query_engine()
print(query_engine_v2.query(question))
# → 업데이트된 수치로 응답
```

#### 방법 2: `refresh_ref_docs` (여러 Document 일괄 갱신)

```python
# Document 내용 재변경
docu.set_content(value="... temperatures of 6,000–8,000 °C and at pressures of 12–15 GPa.")

# 리스트로 전달 → 동일 id면 업데이트, 없으면 삽입
output = index.refresh_ref_docs([docu])
print(output)  # [True] → True: 실제로 갱신됨, False: 변경 없음

query_engine_v3 = index.as_query_engine()
print(query_engine_v3.query(question))
```

| 메서드 | 입력 | 출력 | 특징 |
|--------|------|------|------|
| `update_ref_doc` | Document 1개 | None | 명시적 단일 업데이트 |
| `refresh_ref_docs` | Document 리스트 | bool 리스트 | Upsert (없으면 삽입, 있으면 업데이트) |

### 7-3. Delete — 데이터 삭제

```python
# doc_id 확인
doc_id = docu.doc_id
print(doc_id)  # "new_doc_id"

# 인덱스 및 docstore에서 삭제
index.delete_ref_doc(doc_id, delete_from_docstore=True)

# 삭제 후 Query Engine 재생성
query_engine_v4 = index.as_query_engine()
print(query_engine_v4.query(question))
# → 삭제된 데이터를 더 이상 참조하지 않음
```

#### `delete_from_docstore` 옵션

| 값 | 동작 |
|----|------|
| `True` | 인덱스 + docstore 모두에서 완전 삭제 |
| `False` | 인덱스에서만 제거, docstore에는 남아있음 (재삽입 가능) |

### 인덱스 내 Document 목록 확인

```python
print(index.ref_doc_info.keys())
# → dict_keys(['원본_doc_id', 'new_doc_id', ...])
```

---

## 8. Custom Query Engine

### 8-1. StandardQueryEngine (기본 구조 재현)

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import get_response_synthesizer

class StandardQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        # Step 1: 관련 Node 검색
        nodes = self.retriever.retrieve(query_str)
        # Step 2: Node + 질문 → LLM → 답변
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj

# 사용법
retriever = index.as_retriever()
synthesizer = get_response_synthesizer(response_mode="compact")

query_engine = StandardQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer
)

response = query_engine.query("What did the author do growing up?")
print(str(response))
```

### 8-2. OurCustomQueryEngine (PromptTemplate 적용)

```python
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

# 기본 QA 프롬프트
simple_qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# 짧은 요약 프롬프트
short_sum_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "give the summary of context, less than 20 words.\n"
    "Query: {query_str}\n"
    "Summary: "
)

class OurCustomQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate = simple_qa_prompt  # 기본값

    def custom_query(self, query_str: str):
        # Step 1: 관련 Node 검색
        nodes = self.retriever.retrieve(query_str)

        # Step 2: 검색된 Node 텍스트를 하나로 합치기
        context_str = "\n\n".join([n.node.get_content() for n in nodes])

        # Step 3: 커스텀 프롬프트로 LLM 호출
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return str(response)

# 사용법
llm = OpenAI(model="gpt-3.5-turbo")

query_engine_custom = OurCustomQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=simple_qa_prompt,   # 또는 short_sum_prompt로 변경 가능
)

response = query_engine_custom.query("What did the author do growing up?")
print(response)
```

### StandardQueryEngine vs OurCustomQueryEngine 비교

| 항목 | StandardQueryEngine | OurCustomQueryEngine |
|------|--------------------|--------------------|
| LLM 호출 | Synthesizer 내부 자동 처리 | 명시적 `llm.complete()` 호출 |
| 프롬프트 | LlamaIndex 내부 기본 프롬프트 | 사용자 정의 `PromptTemplate` |
| 유연성 | 낮음 | 높음 (프롬프트 교체 가능) |
| 사용 목적 | 기본 Q&A | 요약, 특수 지시 등 태스크 커스터마이징 |

---

## 9. PromptTemplate 활용

```python
from llama_index.core import PromptTemplate

# {변수명} 형태로 플레이스홀더 정의
my_prompt = PromptTemplate(
    "Context:\n{context_str}\n\n"
    "Task: {task_instruction}\n"
    "Query: {query_str}\n"
    "Response: "
)

# format()으로 변수 채우기
filled_prompt = my_prompt.format(
    context_str="..retrieved text..",
    task_instruction="Translate to Korean",
    query_str="What did the author do?"
)

print(filled_prompt)
```

---

## 10. 핵심 개념 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    LlamaIndex 핵심 객체                      │
├──────────────┬──────────────────────────────────────────────┤
│ Document     │ 파일 1개 = 객체 1개 / .text, .doc_id, .metadata│
│ Node         │ Document를 청크 단위로 분할한 조각             │
│ Embedding    │ Node를 수치 벡터로 변환 (1536차원)             │
│ VectorStore  │ 임베딩 벡터를 저장하는 저장소 (로컬 or 외부 DB) │
│ Index        │ VectorStore + 검색 알고리즘의 조합             │
│ Retriever    │ 쿼리와 유사한 Node를 벡터 검색                 │
│ Synthesizer  │ 검색 결과 + 쿼리 → LLM → 최종 답변 생성       │
│ QueryEngine  │ Retriever + Synthesizer의 통합 인터페이스      │
└──────────────┴──────────────────────────────────────────────┘
```

### RAG의 한계와 대응

| 상황 | 문제 | 대응 방법 |
|------|------|----------|
| 데이터 외 질문 | LLM 파라미터 지식 사용 → hallucination | 프롬프트에 "context만 사용" 강제 |
| 간접 추론 필요 | Retriever가 관련 청크를 못 찾을 수 있음 | chunk_size 조정 또는 reranker 추가 |
| 답변 근거 불투명 | 왜 그 답이 나왔는지 알 수 없음 | 근거 생성 포함 프롬프트 사용 |
