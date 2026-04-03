# 📚 GraphRAG MCP 실습 정리

> **KDD Cup CRAG (Comprehensive RAG Benchmark)** 기반의 RAG 시스템 구축 실습  
> 총 3개의 노트북: 데이터 전처리 → Task 1 (Web 기반 RAG) → Task 2 (Knowledge Graph + Web 통합 RAG)

---

## 목차

1. [전체 시스템 아키텍처](#전체-시스템-아키텍처)
2. [Notebook 1: 데이터 전처리 (Data Preprocessing)](#notebook-1-데이터-전처리)
3. [Notebook 2: Task 1 - Web 기반 검색 요약 RAG](#notebook-2-task-1---web-기반-검색-요약-rag)
4. [Notebook 3: Task 2 - Knowledge Graph + Web 통합 RAG](#notebook-3-task-2---knowledge-graph--web-통합-rag)
5. [전체 파이프라인 흐름 요약](#전체-파이프라인-흐름-요약)

---

## 전체 시스템 아키텍처

```
[사용자 질문]
     │
     ▼
┌─────────────────────────────────────────────────┐
│              GraphRAG 시스템                     │
│                                                 │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │   Retriever  │    │   KG Query Engine    │   │
│  │  (Web 검색)  │    │  (Knowledge Graph)   │   │
│  └──────┬───────┘    └──────────┬───────────┘   │
│         │                      │               │
│         └──────────┬───────────┘               │
│                    ▼                           │
│             ┌────────────┐                     │
│             │   Reader   │                     │
│             │   (LLM)    │                     │
│             └─────┬──────┘                     │
└───────────────────┼─────────────────────────────┘
                    ▼
              [최종 답변 생성]
```

**Task 1**: Retriever + Reader (Web 검색 결과만 활용)  
**Task 2**: Retriever + KG Query Engine + Reader (Web + Knowledge Graph 통합)

---

## Notebook 1: 데이터 전처리

### 1.1 환경 설정 및 라이브러리 설치

```python
import os
os.environ.pop("PYTHONPATH", None)

# 핵심 라이브러리 설치
# - openai: GPT 모델 API 호출
# - blingfire: 문장 단위 토크나이저 (HTML 파싱 후 청크 분리에 사용)
# - lxml: BeautifulSoup의 HTML 파서 백엔드
! pip install openai==1.55.3 blingfire matplotlib lxml --quiet

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "sk-..."
import openai
```

> ⚠️ `PYTHONPATH`를 초기화하는 이유: Colab/Jupyter 환경에서 패키지 충돌 방지

---

### 1.2 CRAG 데이터셋 로드

CRAG 데이터셋은 **JSONL 형식 + bz2 압축**으로 제공됩니다. 각 라인이 하나의 JSON 객체입니다.

```python
import json
import bz2

def pretty_json_print(data):
    """JSON 데이터를 가독성 좋게 출력하는 유틸리티 함수"""
    json_string = json.dumps(data, indent=4)
    lines = json_string.splitlines()
    formatted_lines = "\n\n".join(lines)
    print(formatted_lines)

file_path = 'crag_task_1_dev_v4_release.jsonl.bz2'

# bz2 압축 해제 후 줄 단위로 JSON 파싱
with bz2.open(file_path, 'rt') as file:
    for line in file:
        try:
            data = json.loads(line.strip())
            pretty_json_print(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        break  # 첫 번째 샘플만 확인
```

**CRAG 데이터셋 스키마:**

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `interaction_id` | string | 고유 식별자 |
| `query_time` | string | 질문/검색 발생 시각 |
| `domain` | string | finance, music, movie, sports, open |
| `question_type` | string | simple, comparison, aggregation, multi-hop 등 8가지 |
| `static_or_dynamic` | string | static, slow-changing, fast-changing, real-time |
| `query` | string | RAG가 답해야 할 질문 |
| `answer` | string | 정답 |
| `search_results` | list | 최대 5개 HTML 페이지 (Task 1 기준) |

---

### 1.3 데이터셋을 리스트로 변환

```python
dataset = []

with bz2.open(file_path, 'rt') as file:
    for line in file:
        try:
            data = json.loads(line.strip())
            dataset.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 소요시간: 1~5분
```

---

### 1.4 도메인별 데이터 분포 분석

```python
import matplotlib.pyplot as plt
from collections import Counter

# 도메인별 고유 샘플 추출
unique_domains = {}
for item in dataset:
    if 'domain' in item:
        domain_value = item['domain']
        if domain_value not in unique_domains:
            unique_domains[domain_value] = item

# 도메인별 분포 시각화
domain_counts = Counter(item['domain'] for item in dataset if 'domain' in item)

plt.figure(figsize=(10, 6))
plt.bar(domain_counts.keys(), domain_counts.values())
plt.xlabel('Domain')
plt.ylabel('Count')
plt.title('Number of Data Items per Domain')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**LLM 단독 답변 생성 (RAG 필요성 확인):**

```python
def generate_answer(question):
    """GPT-3.5에게 직접 질문 (검색 없이) → RAG 필요성 입증"""
    messages = [{"role": "user", "content": question}]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content

# 예시: sports 도메인 질문
wanted_domain = 'sports'
gpt_answer = generate_answer(unique_domains[wanted_domain]['query'])
```

> 💡 LLM이 단독으로 틀리는 경우(특히 real-time, fast-changing 질문)를 확인 → RAG의 필요성 증명

---

### 1.5 검색 결과(search_results) 구조 분석

```python
# search_results 스키마:
# - page_name    : 웹페이지 제목
# - page_url     : URL
# - page_snippet : 핵심 내용 요약 (짧음, 누락 가능)
# - page_result  : 전체 HTML 원문 (길고 노이즈 많음)
# - page_last_modified : 최종 수정 시각

example_data = dataset[2617]

for page in example_data['search_results']:
    print(f"Length of title: {len(page['page_name'])}")
    print(f"Length of snippet: {len(page['page_snippet'])}")
    print(f"Length of result: {len(page['page_result'])}")  # 수만 자에 달함
    print("------------------------------")
```

---

### 1.6 HTML 파싱 및 청크 분리 (핵심 전처리)

HTML 원문에서 텍스트만 추출하고 문장 단위로 분리합니다.

```python
from bs4 import BeautifulSoup
from blingfire import text_to_sentences_and_offsets

all_chunks = []

for html_text in example_data['search_results']:
    # 1단계: BeautifulSoup으로 HTML 태그 제거, 순수 텍스트 추출
    soup = BeautifulSoup(html_text["page_result"], features="lxml")
    text = soup.get_text(" ", strip=True)

    if not text:
        all_chunks.append("")
    else:
        # 2단계: blingfire로 문장 경계 감지 → (시작, 끝) offset 목록 반환
        _, offsets = text_to_sentences_and_offsets(text)

        for start, end in offsets:
            # 3단계: 각 문장을 최대 4000자로 자르고 청크 리스트에 추가
            chunk = text[start:end][:4000]
            all_chunks.append(chunk)

# 청크 평균 길이 계산
total_length = sum(len(c) for c in all_chunks)
print("Average length of chunk: ", total_length / len(all_chunks))
```

> ✅ **왜 청크로 분리하나?**
> - `page_result`는 수만 자의 HTML → LLM 컨텍스트 초과
> - 청크 분리 후 임베딩 → 관련 청크만 선택적으로 LLM에 전달
> - 문장 경계에서 분리 → 의미 단절 최소화

---

## Notebook 2: Task 1 - Web 기반 검색 요약 RAG

### 전체 구조

```
[query] + [search_results]
     │
     ▼
 parse_htmls()       ← HTML → 텍스트
     │
     ▼
 extract_chunks()    ← 텍스트 → 문장 단위 청크
     │
     ▼
 embed_text()        ← 청크 + 쿼리 임베딩 생성
     │
     ▼
 cosine_similarity() ← 유사도 계산 → Top-K 청크 선택
     │
     ▼
 Reader.generate()   ← LLM이 Top-K 청크 기반으로 답변 생성
```

---

### 2.1 전역 파라미터 설정

```python
# 생성 답변에 참조할 최대 문장 수
NUM_CONTEXT_SENTENCES = 20

# 각 청크의 최대 문자 수 (문장 단위로 자름)
MAX_CONTEXT_SENTENCE_LENGTH = 1000

# LLM에 전달할 전체 참조 텍스트의 최대 길이
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# 임베딩 모델 배치 크기 (GPU 메모리에 따라 조절)
SENTENCETRANSFORMER_BATCH_SIZE = 128
```

---

### 2.2 청크 추출기 (Chunk Extractor)

```python
from bs4 import BeautifulSoup
from blingfire import text_to_sentences_and_offsets

def parse_htmls(search_results):
    """
    HTML 검색 결과에서 순수 텍스트를 추출한다.
    
    Args:
        search_results: CRAG 데이터셋의 search_results 필드 (list of dict)
    Returns:
        all_documents: 각 웹페이지에서 추출한 텍스트 리스트
    """
    all_documents = []
    for html_text in search_results:
        soup = BeautifulSoup(html_text["page_result"], features="lxml")
        text = soup.get_text(" ", strip=True)
        all_documents.append(text)
    return all_documents


def extract_chunks(all_documents):
    """
    텍스트 문서들을 문장 단위 청크로 분리한다.
    
    핵심: blingfire의 text_to_sentences_and_offsets()는
    문장 경계를 감지하여 (start, end) 오프셋 목록을 반환함.
    → 단어/문장 중간에서 잘리지 않음 (정보 손실 최소화)
    
    Args:
        all_documents: parse_htmls()의 출력
    Returns:
        all_chunks: 개별 문장 청크 리스트
    """
    all_chunks = []
    for document in all_documents:
        if not document:
            all_chunks.append("")
        else:
            _, offsets = text_to_sentences_and_offsets(document)
            for start, end in offsets:
                chunk = document[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
                all_chunks.append(chunk)
    return all_chunks
```

> 💡 전체 문서 대비 청크 길이가 약 **100배** 감소 → 임베딩 연산 효율화

---

### 2.3 BaseRetriever (직접 구현 버전)

LlamaIndex 없이 OpenAI 임베딩 + 코사인 유사도로 Retriever를 직접 구현합니다.

```python
import numpy as np
import openai
import os

class BaseRetriever:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed_text(self, texts):
        """
        OpenAI text-embedding-3-small 모델로 임베딩 생성.
        
        - 단일 문자열도 리스트로 변환하여 배치 처리
        - 반환: (N, embedding_dim) shape의 numpy 배열
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [np.array(item.embedding) for item in response.data]
        return np.array(embeddings)

    def retrieve(self, query, search_results, topk):
        """
        질문과 가장 유사한 Top-K 청크를 반환한다.
        
        Step 1: HTML → 텍스트 → 청크 추출
        Step 2: 모든 청크 + 쿼리를 임베딩 벡터로 변환
        Step 3: 코사인 유사도 계산 → Top-K 인덱스 선택
        """
        # Step 1
        all_documents = parse_htmls(search_results)
        all_chunks = extract_chunks(all_documents)

        # Step 2
        all_embeddings = self.embed_text(all_chunks)       # (N, D)
        query_embedding = self.embed_text(query)[0]        # (D,)

        # Step 3: 코사인 유사도 = 내적 / (각 노름의 곱)
        cosine_scores = np.dot(all_embeddings, query_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = (-cosine_scores).argsort()[:topk]  # 내림차순 정렬
        top_k_chunks = np.array(all_chunks)[top_k_indices]

        return top_k_chunks
```

**사용 예시:**
```python
retriever = BaseRetriever()
topk = 5

with bz2.open(dataset_path, "rt") as file:
    for line in file:
        item = json.loads(line)
        retrieved_results = retriever.retrieve(item['query'], item['search_results'], topk)
        break

for rank, result in enumerate(retrieved_results):
    print(f"rank {rank+1}: {result}")
```

---

### 2.4 LlamaIndexRetriever (프레임워크 활용 버전)

LlamaIndex를 사용하면 인덱싱, 청킹, 검색을 간결하게 구현할 수 있습니다.

```python
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# 전역 임베딩 모델 설정
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class LlamaIndexRetriever:
    def __init__(self):
        # chunk_size=512 토큰 단위로 분리, 청크 간 중복 없음
        self.parser = SentenceSplitter(chunk_size=512, chunk_overlap=0)

    def retrieve(self, query, search_results, topk):
        """
        LlamaIndex 파이프라인:
        1. HTML → 텍스트 변환 후 Document 객체 생성
        2. SentenceSplitter로 노드 분리
        3. VectorStoreIndex에 임베딩 벡터 저장 (인메모리)
        4. similarity_top_k로 Top-K 유사 청크 검색
        """
        documents = []
        for document in parse_htmls(search_results):
            if not document:
                documents.append(Document(text=""))
            else:
                documents.append(Document(text=document))

        # 자동으로 청크 분리 + 임베딩 생성 + 벡터 인덱스 구축
        base_index = VectorStoreIndex.from_documents(
            documents=documents,
            transformations=[self.parser]
        )

        # 인덱스에서 Retriever 생성
        base_retriever = base_index.as_retriever(similarity_top_k=topk)
        retrieved_nodes = base_retriever.retrieve(query)

        # 노드에서 텍스트 내용만 추출
        retrieved_results = [
            node.node.get_content().strip() for node in retrieved_nodes
        ]
        return retrieved_results
```

> 💡 **BaseRetriever vs LlamaIndexRetriever 비교**
>
> | 항목 | BaseRetriever | LlamaIndexRetriever |
> |------|--------------|---------------------|
> | 코드 복잡도 | 높음 (직접 구현) | 낮음 (추상화) |
> | 커스터마이즈 | 자유로움 | 제한적 |
> | 학습 목적 | 내부 동작 이해 | 빠른 프로토타이핑 |

---

### 2.5 Reader (LLM 기반 답변 생성기)

```python
from openai import OpenAI

oai_client = OpenAI()

class Reader:
    def __init__(self):
        # 시스템 프롬프트: CRAG 평가 기준에 맞게 설계
        # - 간결한 답변 (평가 시 첫 75 토큰만 사용)
        # - 정보 없을 시 "I don't know" 응답 유도
        self.system_prompt = """
        You are provided with a question and various references.
        Your task is to answer the question succinctly, using the fewest words possible.
        If the references do not contain the necessary information to answer the question, respond with 'I don't know'.
        There is no need to explain the reasoning behind your answers.
        """

    def generate_response(self, query: str, top_k_chunks: list) -> str:
        """
        Top-K 청크를 참조로 LLM이 답변을 생성한다.
        temperature=0 → 결정론적 출력 (재현 가능)
        """
        llm_input = self.prompt_generator(query, top_k_chunks)
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=llm_input
        ).choices[0].message.content
        return completion

    def prompt_generator(self, query, top_k_chunks):
        """
        프롬프트 구성:
        [System] 역할 지시
        [User]   참조 텍스트 + 질문
        """
        references = ""
        if len(top_k_chunks) > 0:
            references += "# References \n"
            for chunk in top_k_chunks:
                references += f"- {chunk.strip()}\n"

        # 참조 텍스트 길이 제한 (LLM 컨텍스트 윈도우 보호)
        references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

        user_message = f"{references}\n------\n\n"
        user_message += f"Using only the references listed above, answer the following question: \n"
        user_message += f"Question: {query}\n"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_message},
        ]
```

---

### 2.6 RAG 시스템 통합 (Task 1 완성)

```python
class RAG:
    def __init__(self):
        self.retriever = LlamaIndexRetriever()
        self.reader = Reader()

    def inference(self, query, search_results, topk):
        """
        완전한 RAG 파이프라인:
        1. Retriever: 검색 결과에서 관련 청크 추출
        2. Reader: 청크를 참조로 LLM이 답변 생성
        """
        # Step 1: Retrieve
        retrieved_results = self.retriever.retrieve(query, search_results, topk)

        # Step 2: Read (Generate)
        answer = self.reader.generate_response(query, retrieved_results)

        return answer, retrieved_results
```

**평가 실행:**
```python
rag = RAG()
topk = 5
repeat = 0

with bz2.open(dataset_path, "rt") as file:
    for line in file:
        if repeat > 9:
            break
        item = json.loads(line)
        answer = rag.inference(item['query'], item['search_results'], topk)[0]
        print(f"query: {item['query']}")
        print(f"predicted: {answer}")
        print(f"ground truth: {item['answer']}")
        print()
        repeat += 1
```

---

### 2.7 오류 케이스 분석

Task 1 RAG가 실패하는 대표적인 케이스:

```python
# ❌ 실패 케이스 1: 잘못된 지식 (정적 질문이지만 LLM 오류)
# Q: "In 2004, which animated film won the best animated feature oscar?"
# 정답: "Finding Nemo"  /  GPT 오답: "The Incredibles"

# ❌ 실패 케이스 2: 구조화된 금융 데이터 (표/KG 필요)
# Q: "What is the ex-dividend date of Microsoft in the 1st qtr of 2024?"
# Q: "I'm looking for the P/E ratio of DKS?"
# Q: "What's AUPH's earnings per share?"

# 이런 금융 수치 데이터는 HTML 텍스트에 구조가 없어서
# 단순 텍스트 검색으로는 정확한 값을 찾기 어려움
# → Task 2에서 Knowledge Graph API로 해결
rag = RAG()
topk = 5
repeat = 0

with bz2.open(dataset_path, "rt") as file:
    for line in file:
        if repeat >= 65:
            break
        if repeat not in [14, 53, 64]:  # 금융 도메인 실패 샘플
            repeat += 1
            continue
        item = json.loads(line)
        answer, retrieved_results = rag.inference(item['query'], item['search_results'], topk)
        print(f"query: {item['query']}")
        print(f"predicted: {answer}")
        print(f"ground truth: {item['answer']}")
        repeat += 1
```

---

## Notebook 3: Task 2 - Knowledge Graph + Web 통합 RAG

### 3.1 Mock KG API 클래스

KDD Cup에서 제공하는 Knowledge Graph는 REST API 서버로 접근합니다.

```python
import requests
import json
import os

class CRAG(object):
    """
    Mock Knowledge Graph API 클라이언트.
    
    서버 주소: 외부 KG API 서버 (운영진이 제공)
    프로토콜: HTTP POST, JSON 요청/응답
    
    금융 도메인 지원 API 목록:
    - get_company_name     : 회사명으로 정식 명칭 조회
    - get_ticker_by_name   : 회사명으로 주식 티커 조회
    - get_price_history    : 주가 이력 조회
    - get_detailed_price_history : 세부 주가 이력
    - get_dividends_history: 배당금 이력
    - get_market_capitalization : 시가총액
    - get_eps              : 주당순이익 (EPS)
    - get_pe_ratio         : 주가수익비율 (P/E)
    - get_info             : 종합 정보
    """
    def __init__(self, server=None):
        self.server = os.environ.get('CRAG_SERVER', "http://10.2.0.165:8000")

    def _post(self, endpoint, query):
        """공통 POST 요청 헬퍼"""
        url = self.server + endpoint
        headers = {'accept': "application/json"}
        data = {'query': query}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)

    def finance_get_company_name(self, query: str):
        return self._post('/finance/get_company_name', query)

    def finance_get_ticker_by_name(self, query: str):
        return self._post('/finance/get_ticker_by_name', query)

    def finance_get_price_history(self, ticker_name: str):
        return self._post('/finance/get_price_history', ticker_name)

    def finance_get_dividends_history(self, ticker_name: str):
        return self._post('/finance/get_dividends_history', ticker_name)

    def finance_get_market_capitalization(self, ticker_name: str):
        return self._post('/finance/get_market_capitalization', ticker_name)

    def finance_get_eps(self, ticker_name: str):
        return self._post('/finance/get_eps', ticker_name)

    def finance_get_pe_ratio(self, ticker_name: str):
        return self._post('/finance/get_pe_ratio', ticker_name)

    def finance_get_info(self, ticker_name: str):
        return self._post('/finance/get_info', ticker_name)


# API 사용 예시
api = CRAG()
company = api.finance_get_company_name("microsoft")
ticker = api.finance_get_ticker_by_name(company["result"][0])
eps = api.finance_get_eps(ticker['result'])
```

---

### 3.2 Query Generator (LLM 기반 엔티티 추출)

자연어 질문에서 API 호출에 필요한 구조화된 파라미터를 추출합니다.

```python
# 엔티티 추출 프롬프트 템플릿
entity_extract_template = """
You are given a Query and Query Time. Do the following:

1) Determine the domain the query is about. The domain should be one of the following: 
   "finance", "sports", "music", "movie", "encyclopedia". 
   If none applies, use "other". Use "domain" as the key in the result json.

2) Extract structured information from the query. 
   For `finance` queries, extract:
   - `market_identifier`: 종목명 또는 티커 심볼
   - `metric`: price / dividend / P/E ratio / EPS / marketCap / other 중 하나
   - `datetime`: 질문의 시간 범위 (명시 없으면 Query Time 사용)

Return the results in a FLAT json.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

def prompt_generator(query):
    """질문을 엔티티 추출 프롬프트로 포맷팅"""
    user_message = f"Query: {query}\n"
    return [
        {"role": "system", "content": entity_extract_template},
        {"role": "user",   "content": user_message},
    ]
```

```python
from openai import OpenAI
from json import JSONDecoder

oai_client = OpenAI()

def extract_json_objects(text, decoder=JSONDecoder()):
    """
    LLM 출력에서 JSON 오브젝트를 파싱하는 안전한 함수.
    LLM이 가끔 마크다운 코드블록 등 부가 텍스트를 함께 출력할 때 대응.
    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results

def generate_query(query):
    """
    자연어 질문 → 구조화된 쿼리 딕셔너리 변환.
    
    반환:
        completion  : {"domain": "finance", "market_identifier": "MSFT", "metric": "EPS", ...}
        is_finance  : bool (finance 도메인 여부)
    """
    llm_input = prompt_generator(query)
    completion = oai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=llm_input
    ).choices[0].message.content

    try:
        completion = json.loads(completion)
    except:
        # JSON 파싱 실패 시 텍스트에서 JSON 오브젝트 추출
        completion = extract_json_objects(completion)

    is_finance = (
        "domain" in completion and completion["domain"] == "finance"
    )
    return completion, is_finance
```

**예시 출력:**
```python
# 입력: "What is the P/E ratio of DKS?"
# 출력:
# {
#   "domain": "finance",
#   "market_identifier": "DKS",
#   "metric": "P/E ratio",
#   "datetime": "2024-XX-XX"
# }
```

---

### 3.3 날짜 유틸리티 함수

KG API 결과에서 날짜 기반 필터링에 사용됩니다.

```python
import datetime
from datetime import timedelta
from dateutil import parser
import pytz
import re

def normalize_key(key):
    """키 정규화: 특수문자 제거 + 소문자 변환"""
    return re.sub(r'[^a-zA-Z0-9]', '', key).lower()

def get_metric_from_response(response, metric):
    """API 응답 딕셔너리에서 정규화된 키로 값 탐색"""
    normalized_metric = normalize_key(metric)
    if response is not None:
        for key, value in response.items():
            if normalize_key(key) == normalized_metric:
                return value
    return None

def convert_to_standard_format(date_string):
    """
    다양한 날짜 포맷 → EST 기준 표준 포맷으로 통일
    예: "February 14, 2024" → "2024-02-14 00:00:00 EST"
    """
    try:
        dt = parser.parse(date_string)
        est = pytz.timezone('US/Eastern')
        if dt.tzinfo is None:
            dt = est.localize(dt)
        else:
            dt = dt.astimezone(est)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    except (ValueError, OverflowError) as e:
        return date_string

def add_one_day(date_string):
    """날짜 +1일 (API 날짜 경계 처리)"""
    try:
        dt = parser.parse(date_string)
        est = pytz.timezone('US/Eastern')
        dt = est.localize(dt) if dt.tzinfo is None else dt.astimezone(est)
        dt_plus_one = dt + timedelta(days=1)
        return dt_plus_one.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception as e:
        return f"Invalid date string: {e}"

def subtract_one_day(date_string):
    """날짜 -1일 (API 날짜 경계 처리)"""
    try:
        dt = parser.parse(date_string)
        est = pytz.timezone('US/Eastern')
        dt = est.localize(dt) if dt.tzinfo is None else dt.astimezone(est)
        dt_minus_one = dt - timedelta(days=1)
        return dt_minus_one.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception as e:
        return f"Invalid date string: {e}"
```

---

### 3.4 Query Executor (의사결정 트리 기반 API 실행)

```
[Decision Tree]

질문에서 market_identifier 추출
         │
         ▼
  finance_get_company_name()
         │
         ▼
  finance_get_ticker_by_name()
         │
         ▼
  metric 판단
  ├── price    → get_price_history()
  ├── dividend → get_dividends_history()
  ├── P/E ratio→ get_pe_ratio()
  ├── EPS      → get_eps()
  ├── marketCap→ get_market_capitalization()
  └── other    → get_info() + metric 필터링
         │
         ▼
  datetime 필터링 (±1일 허용)
         │
         ▼
  kg_results 반환
```

```python
import copy

def get_finance_kg_results(generated_query):
    """
    의사결정 트리에 따라 KG API를 순차 호출하여 결과를 반환한다.
    
    날짜 허용 오차: ±1일 (API 데이터가 정확히 해당 날짜에 없을 수 있음)
    """
    # 날짜 파싱 (범위 지정 가능: "2024-01-01 - 2024-03-31")
    formatted_time_list = []
    if 'datetime' in generated_query:
        datetime_list = generated_query['datetime'].split(' - ')
        for dt in datetime_list:
            formatted_time_list.append(convert_to_standard_format(dt.strip()))

    kg_results = []

    if "market_identifier" in generated_query and generated_query["market_identifier"]:
        company_names = (
            generated_query["market_identifier"].split(",")
            if isinstance(generated_query["market_identifier"], str)
            else generated_query["market_identifier"]
        )

        for company_name in company_names:
            try:
                # 1단계: 회사명 → 정식명칭 → 티커 조회
                res = api.finance_get_company_name(company_name)["result"]
                ticker_name = company_name.upper() if res == [] else \
                              api.finance_get_ticker_by_name(res[0])["result"]

                # 2단계: metric에 따라 적절한 API 호출
                metric = generated_query['metric'].lower().strip()
                if metric == 'price':
                    response = api.finance_get_price_history(ticker_name)['result']
                elif metric == 'dividend':
                    response = api.finance_get_dividends_history(ticker_name)['result']
                elif metric == 'p/e ratio':
                    response = api.finance_get_pe_ratio(ticker_name)['result']
                elif metric == 'eps':
                    response = api.finance_get_eps(ticker_name)["result"]
                elif metric == 'marketcap':
                    response = api.finance_get_market_capitalization(ticker_name)['result']
                else:
                    response = api.finance_get_info(ticker_name)['result']
                    metric_value = get_metric_from_response(response, generated_query['metric'])
                    if metric_value is not None:
                        response = metric_value

                # 3단계: datetime 기준 필터링 (±1일 허용)
                for formatted_time in formatted_time_list:
                    if formatted_time in response:
                        filtered_response = copy.deepcopy(response[formatted_time])
                    elif add_one_day(formatted_time) in response:
                        filtered_response = copy.deepcopy(response[add_one_day(formatted_time)])
                    elif subtract_one_day(formatted_time) in response:
                        filtered_response = copy.deepcopy(response[subtract_one_day(formatted_time)])
                    else:
                        filtered_response = copy.deepcopy(response)

                    kg_results.append({
                        company_name + " " + generated_query["metric"]: filtered_response,
                        'time': formatted_time
                    })

            except Exception as e:
                kg_results.append({"error": str(e)})

    return kg_results
```

---

### 3.5 KGQueryEngine (통합 KG 쿼리 엔진)

```python
class KGQueryEngine:
    """
    질문을 받아 KG에서 관련 정보를 추출하는 통합 엔진.
    
    내부적으로:
    1. generate_query(): LLM으로 구조화된 쿼리 생성
    2. get_finance_kg_results(): API 호출 및 결과 필터링
    
    비금융 도메인은 KG 미사용 (빈 문자열 반환)
    """
    def query(self, query):
        generated_query, is_finance = self.generate_query(query)
        if is_finance:
            kg_results = self.get_finance_kg_results(generated_query)
        else:
            kg_results = ""
        return kg_results, is_finance

    def generate_query(self, query):
        llm_input = prompt_generator(query)
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=llm_input
        ).choices[0].message.content

        try:
            completion = json.loads(completion)
        except:
            completion = extract_json_objects(completion)

        is_finance = (
            "domain" in completion and completion["domain"] == "finance"
        )
        return completion, is_finance

    def get_finance_kg_results(self, generated_query):
        # (위의 get_finance_kg_results 함수와 동일한 로직을 메서드로 포함)
        return get_finance_kg_results(generated_query)
```

---

### 3.6 RAGWithKG (LLM + Knowledge Graph)

```python
class RAGWithKG:
    """
    KG만 사용하는 RAG (Web 검색 결과 미사용).
    금융 도메인 질문에 특화.
    """
    def __init__(self):
        self.kg_query_engine = KGQueryEngine()
        self.reader = Reader()

    def inference(self, query):
        # Step 1: KG에서 관련 정보 조회
        kg_results, is_finance = self.kg_query_engine.query(query)

        # Step 2: KG 결과를 참조로 LLM이 답변 생성
        answer = self.reader.generate_response(query, [kg_results])

        return answer, kg_results

# 검증: Task 1에서 실패했던 금융 질문들
rag = RAGWithKG()
repeat = 0
with bz2.open(dataset_path, "rt") as file:
    for line in file:
        item = json.loads(line)
        if repeat >= 65:
            break
        if repeat not in [14, 53, 64]:
            repeat += 1
            continue
        answer, kg_results = rag.inference(item['query'])
        print(f"query: {item['query']}")
        print(f"predicted: {answer}")
        print(f"ground truth: {item['answer']}")
        print(f"kg results: {kg_results}")
        repeat += 1
```

---

### 3.7 RAGWithSRKG - 최종 통합 시스템 (Web + KG)

```python
class RAGWithSRKG:
    """
    최종 통합 RAG 시스템:
    - 금융 도메인: KG API 결과 우선 사용
    - 비금융 도메인: Web 검색 결과 사용
    
    구성 요소:
    1. LlamaIndexRetriever: Web 검색 결과 → Top-K 청크
    2. KGQueryEngine: Knowledge Graph → 구조화된 금융 데이터
    3. Reader: 통합 컨텍스트 기반 LLM 답변 생성
    """
    def __init__(self):
        self.retriever = LlamaIndexRetriever()
        self.kg_query_engine = KGQueryEngine()
        self.reader = Reader()

    def inference(self, query, search_results, topk):
        # Step 1: Web 검색 결과에서 Top-K 청크 추출
        retrieved_results = self.retriever.retrieve(query, search_results, topk)

        # Step 2: KG에서 관련 정보 조회
        kg_results, is_finance = self.kg_query_engine.query(query)

        # Step 3: 도메인에 따라 참조 소스 선택
        if is_finance:
            # 금융 질문: 정확한 수치가 있는 KG 결과 우선
            combined_results = [kg_results]
        else:
            # 비금융 질문: Web 검색 청크 사용
            combined_results = retrieved_results

        # Step 4: LLM으로 최종 답변 생성
        answer = self.reader.generate_response(query, combined_results)

        return answer, combined_results
```

**최종 통합 시스템 테스트:**
```python
rag = RAGWithSRKG()
topk = 5

# 4가지 대표 케이스 검증
# [5]  : "In 2004, which animated film won the best animated feature oscar?" → Finding Nemo
# [14] : "What is the ex-dividend date of Microsoft in 1st qtr of 2024?"   → Feb 14, 2024
# [53] : "I'm looking for the P/E ratio of DKS?"                           → 13.75
# [64] : "What's AUPH's earnings per share?"                               → 0.4

repeat = 0
with bz2.open(dataset_path, "rt") as file:
    for line in file:
        if repeat >= 65:
            break
        if repeat not in [5, 14, 53, 64]:
            repeat += 1
            continue
        item = json.loads(line)
        answer, retrieved_results = rag.inference(item['query'], item['search_results'], topk)
        print(f"query: {item['query']}")
        print(f"predicted: {answer}")
        print(f"ground truth: {item['answer']}")
        repeat += 1
```

---

## 전체 파이프라인 흐름 요약

```
[입력 질문]
     │
     ├──────────────────────────────────────┐
     │                                      │
     ▼                                      ▼
[LlamaIndexRetriever]               [KGQueryEngine]
HTML 파싱 → 청크 분리               LLM으로 엔티티 추출
임베딩 → 코사인 유사도              도메인 판별 (finance?)
Top-K 청크 선택                     Mock API 호출
     │                              KG 결과 필터링
     │                                      │
     └──────────────┬───────────────────────┘
                    │
              도메인 분기
         ┌──── finance? ────┐
         │ Yes              │ No
         ▼                  ▼
    [KG 결과]         [Web 청크]
         │                  │
         └────────┬──────────┘
                  ▼
            [Reader - LLM]
      system_prompt + 참조 + 질문
      GPT-3.5-turbo (temperature=0)
                  │
                  ▼
            [최종 답변]
```

### 진화 과정

| 시스템 | 구성 | 한계 |
|--------|------|------|
| **LLM Only** | GPT-3.5 단독 | 최신 정보 부재, 환각 |
| **Task 1 RAG** | Web 검색 + LLM | 금융 수치 데이터 취약 |
| **RAGWithKG** | KG API + LLM | Web 정보 미활용 |
| **RAGWithSRKG** | Web + KG + LLM | 도메인별 최적 소스 선택 |

### 핵심 설계 원칙

1. **청크 단위 검색**: 전체 문서 대신 문장 단위 청크 → 검색 정밀도 향상, 연산 효율화
2. **도메인 라우팅**: 금융 질문 → KG / 나머지 → Web으로 자동 분기
3. **간결한 프롬프트**: CRAG 평가가 첫 75 토큰만 채점 → 답변 간결성 필수
4. **"I don't know" 유도**: 잘못된 정보보다 모름 응답이 점수에 유리
5. **날짜 허용 오차**: KG 데이터 부재 시 ±1일 범위로 재시도

---

*Generated from KDD Cup CRAG 실습 자료 (1__Data_preprocessing.ipynb, 2__Task_1.ipynb, 3__Task_2.ipynb)*
