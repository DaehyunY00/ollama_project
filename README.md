# 🎯 국방 M&S RAG 시스템

> 국방 모델링 및 시뮬레이션 문서 기반 지능형 질의응답 시스템

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [API 문서](#api-문서)
- [예제](#예제)
- [문제해결](#문제해결)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

## 📖 개요

국방 M&S RAG 시스템은 국방 모델링 및 시뮬레이션(M&S) 분야의 전문 문서들을 학습하여, 사용자의 질문에 대해 정확하고 전문적인 답변을 제공하는 AI 시스템입니다.

### 🎯 주요 목표

- **전문성**: 국방 M&S 분야의 깊이 있는 지식 제공
- **정확성**: 문서 기반의 신뢰할 수 있는 답변 생성
- **효율성**: 빠른 검색과 응답 시간
- **확장성**: 새로운 문서와 도메인 추가 용이

## ✨ 주요 기능

### 🔍 지능형 문서 검색
- **다중 형식 지원**: PDF, DOCX, TXT, MD 파일 처리
- **의미 기반 검색**: 벡터 임베딩을 통한 맥락적 검색
- **도메인 특화**: 지상전, 해상전, 공중전, 사이버전 등 영역별 검색

### 🤖 AI 기반 질의응답
- **Ollama 통합**: 로컬 LLM 모델 활용
- **컨텍스트 인식**: 관련 문서 내용을 바탕으로 한 답변
- **전문 용어 처리**: 국방 M&S 특화 용어 이해

### 📊 데이터 파일 생성
- **XML**: 시뮬레이션 설정 파일
- **JSON**: 모델 파라미터 파일
- **CSV**: 시나리오 데이터 파일

### 🔧 데이터 검증
- **형식 검증**: 생성된 파일의 구조적 무결성 확인
- **내용 검증**: 국방 M&S 도메인 규칙 적용
- **품질 보고서**: 상세한 검증 결과 리포트

## 🖥️ 시스템 요구사항

### 하드웨어
- **CPU**: Intel i5 이상 또는 AMD Ryzen 5 이상
- **메모리**: 16GB RAM 이상 (권장: 32GB)
- **저장공간**: 10GB 이상 여유 공간
- **GPU**: CUDA 지원 GPU (선택사항, 성능 향상)

### 소프트웨어
- **운영체제**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.11 이상
- **Ollama**: 최신 버전

### 의존성
- LangChain Framework
- ChromaDB (벡터 데이터베이스)
- Streamlit (웹 인터페이스)
- 기타 Python 패키지 (requirements.txt 참조)

## 🚀 설치 및 설정

### 1. 저장소 클론

```bash
git clone https://github.com/your-org/defense-ms-rag.git
cd defense-ms-rag
```

### 2. Python 가상환경 설정

```bash
# Python 3.11로 ollama 가상환경 생성
python3.11 -m venv ollama

# 가상환경 활성화
# Windows
ollama\Scripts\activate
# macOS/Linux
source ollama/bin/activate
```

### 3. 의존성 설치

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 패키지 설치
pip install -r requirements.txt
```

### 4. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (https://ollama.ai/download)
# 모델 다운로드
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### 5. 환경 변수 설정

```bash
# .env 파일 생성 (이미 제공됨)
cp .env.example .env

# 필요시 설정값 수정
nano .env
```

### 6. 초기 설정 실행

```bash
# 설정 스크립트 실행
# Linux/macOS
bash scripts/setup.sh

# Windows
scripts\setup.bat
```

## 📘 사용법

### 커맨드라인 인터페이스

#### 기본 사용

```bash
# 시스템 상태 확인
python src/main.py --status

# 문서 로드
python src/main.py --load-docs ./docs/pdfs

# 단일 질문
python src/main.py --question "HLA 표준이란 무엇인가요?"

# 대화형 모드
python src/main.py
```

#### 고급 사용

```bash
# 지식 베이스 재설정
python src/main.py --reset

# 특정 설정 파일 사용
python src/main.py --config ./config/custom_config.yaml

# 도메인별 질문
python src/main.py --question "전투 시뮬레이션 모델링" --domain "지상전"
```

### 프로그래밍 인터페이스

```python
from src.main import DefenseRAGSystem

# 시스템 초기화
rag_system = DefenseRAGSystem()

# 문서 로드
rag_system.load_documents_from_directory("./docs/pdfs")

# 질문하기
result = rag_system.ask_question("M&S 검증 방법론에 대해 설명해주세요")
print(result['answer'])

# 도메인별 검색
documents = rag_system.search_documents("시뮬레이션", domain="해상전")
```

### 웹 인터페이스

```bash
# Streamlit 앱 실행
streamlit run app.py

# 브라우저에서 http://localhost:8501 접속
```

## 📁 프로젝트 구조

```
ollama_project/
├── 📁 ollama/                      # Python 3.11 가상환경
├── 📄 README.md                   # 프로젝트 가이드
├── 📄 requirements.txt            # Python 패키지 목록
├── 📄 .env                       # 환경 변수
│
├── 📁 config/                     # 설정 파일들
│   ├── 📄 config.yaml            # 기본 설정
│   └── 📄 ollama_config.json     # Ollama 설정
│
├── 📁 src/                       # 소스 코드
│   ├── 📄 main.py                # 메인 실행 파일
│   ├── 📁 rag/                   # RAG 시스템
│   │   ├── 📄 document_loader.py # 문서 로더
│   │   ├── 📄 vector_store.py    # 벡터 스토어
│   │   └── 📄 retriever.py       # 검색기
│   ├── 📁 models/                # 모델 관련
│   │   └── 📄 ollama_client.py   # Ollama 클라이언트
│   └── 📁 utils/                 # 유틸리티
│       ├── 📄 file_generator.py  # 파일 생성기
│       └── 📄 data_validator.py  # 데이터 검증
│
├── 📁 docs/                      # 문서 폴더
│   ├── 📄 project_guide.md       # 프로젝트 상세 가이드
│   └── 📁 pdfs/                  # RAG용 PDF 문서들
│
├── 📁 data/                      # 데이터 폴더
│   ├── 📁 samples/               # 샘플 데이터
│   ├── 📁 templates/             # 템플릿 파일들
│   └── 📁 output/                # 생성된 파일들
│
├── 📁 tests/                     # 테스트 코드
│   ├── 📄 test_rag.py           # RAG 시스템 테스트
│   └── 📄 test_file_generator.py # 파일 생성기 테스트
│
└── 📁 scripts/                   # 실행 스크립트들
    ├── 📄 setup.sh              # 초기 설정 (Linux/macOS)
    └── 📄 setup.bat             # 초기 설정 (Windows)
```

## 🔌 API 문서

### DefenseRAGSystem 클래스

#### 주요 메서드

##### `load_documents_from_directory(directory_path: str) -> bool`
디렉토리의 모든 지원 문서를 로드합니다.

**매개변수:**
- `directory_path`: 문서가 있는 디렉토리 경로

**반환값:**
- `bool`: 성공 여부

##### `ask_question(question: str, domain: str = None) -> Dict[str, Any]`
질문에 대한 답변을 생성합니다.

**매개변수:**
- `question`: 사용자 질문
- `domain`: 특정 도메인 (선택사항)

**반환값:**
- `Dict`: 답변과 메타데이터

##### `search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]`
문서를 검색합니다.

**매개변수:**
- `query`: 검색 쿼리
- `k`: 반환할 문서 수

**반환값:**
- `List[Dict]`: 검색된 문서 목록

### DefenseFileGenerator 클래스

#### 주요 메서드

##### `generate_simulation_config_xml(scenario_name: str) -> str`
시뮬레이션 설정 XML 파일을 생성합니다.

##### `generate_model_parameters_json(model_name: str) -> str`
모델 파라미터 JSON 파일을 생성합니다.

##### `generate_scenario_data_csv(scenario_name: str) -> str`
시나리오 데이터 CSV 파일을 생성합니다.

## 📝 예제

### 기본 질의응답

```python
# 시스템 초기화
rag_system = DefenseRAGSystem()

# 질문 예시
questions = [
    "HLA 표준의 주요 구성요소는 무엇인가요?",
    "전투 효과도 분석 방법론을 설명해주세요",
    "시뮬레이션 검증과 확인의 차이점은?",
    "분산 시뮬레이션의 장단점은 무엇인가요?"
]

for question in questions:
    result = rag_system.ask_question(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print("-" * 50)
```

### 도메인별 검색

```python
# 지상전 관련 문서 검색
ground_docs = rag_system.search_documents("전차 시뮬레이션", domain="지상전")

# 해상전 관련 질문
naval_answer = rag_system.ask_question(
    "잠수함 작전 시뮬레이션", 
    domain="해상전"
)
```

### 데이터 파일 생성

```python
from src.utils.file_generator import DefenseFileGenerator

generator = DefenseFileGenerator()

# 완전한 시나리오 생성
files = generator.generate_complete_scenario(
    scenario_name="한반도_방어_시나리오",
    num_units=200,
    duration=7200
)

print("생성된 파일들:")
for file_type, file_path in files.items():
    print(f"- {file_type}: {file_path}")
```

### 데이터 검증

```python
from src.utils.data_validator import DefenseDataValidator

validator = DefenseDataValidator()

# 디렉토리 내 모든 파일 검증
results = validator.validate_directory("./data/output")

# 검증 보고서 생성
report = validator.generate_validation_report(
    results, 
    "./data/validation_report.txt"
)
```

## 🔧 설정

### config.yaml 주요 설정

```yaml
# Ollama 설정
ollama:
  host: "http://localhost:11434"
  model: "llama3:8b"
  embedding_model: "nomic-embed-text"
  temperature: 0.7

# 벡터 데이터베이스
vector_db:
  persist_directory: "./data/chroma_db"
  collection_name: "defense_ms_docs"

# 문서 처리
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: ["pdf", "txt", "md", "docx"]

# RAG 설정
rag:
  retrieval:
    top_k: 5
    score_threshold: 0.7
  generation:
    max_context_length: 4000
```

### 환경 변수 (.env)

```bash
# Ollama 설정
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3:8b
EMBEDDING_MODEL=nomic-embed-text

# 데이터베이스
CHROMA_DB_PATH=./data/chroma_db
COLLECTION_NAME=defense_ms_docs

# 로그 설정
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_system.log
```

## 🚨 문제해결

### 일반적인 문제

#### 1. Ollama 연결 실패

**증상:** `Connection refused` 오류
**해결:** 
```bash
# Ollama 서비스 상태 확인
ollama list

# Ollama 재시작
ollama serve
```

#### 2. 메모리 부족

**증상:** `Out of memory` 오류
**해결:**
- `config.yaml`에서 `chunk_size` 감소
- 더 작은 모델 사용 (`llama3:8b` → `llama3.2:3b`)
- 시스템 RAM 확장

#### 3. 문서 로드 실패

**증상:** PDF 텍스트 추출 오류
**해결:**
```bash
# OCR 라이브러리 설치 (필요시)
pip install pytesseract
apt-get install tesseract-ocr  # Ubuntu
```

#### 4. 벡터 데이터베이스 오류

**증상:** ChromaDB 초기화 실패
**해결:**
```bash
# 데이터베이스 재설정
python src/main.py --reset

# 권한 문제 해결
chmod -R 755 ./data/chroma_db
```

### 성능 최적화

#### GPU 사용 (NVIDIA)

```bash
# CUDA 지원 확인
nvidia-smi

# GPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 메모리 최적화

```yaml
# config.yaml 설정
document_processing:
  chunk_size: 500        # 기본: 1000
  chunk_overlap: 100     # 기본: 200

rag:
  retrieval:
    top_k: 3            # 기본: 5
```

## 🧪 테스트

### 테스트 실행

```bash
# 전체 테스트
pytest tests/

# 특정 테스트
pytest tests/test_rag.py -v

# 커버리지 포함
pytest --cov=src tests/
```

### 수동 테스트

```bash
# 기본 기능 테스트
python tests/test_rag.py

# 시스템 상태 체크
python src/main.py --status
```

## 📊 모니터링

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f logs/rag_system.log

# 오류 로그만 확인
grep ERROR logs/rag_system.log
```

### 성능 메트릭

```python
# 시스템 통계 조회
status = rag_system.get_system_status()
print(f"문서 수: {status['vector_store']['document_count']}")
print(f"응답 시간: {status['health']['response_time']:.2f}초")
```

## 🤝 기여하기

### 개발 환경 설정

```bash
# 개발용 패키지 설치
pip install -r requirements-dev.txt

# pre-commit 훅 설정
pre-commit install
```

### 코드 스타일

- **Python**: PEP 8 준수
- **문서화**: Google 스타일 독스트링
- **테스트**: pytest 사용
- **타입 힌트**: 모든 함수에 타입 어노테이션

### Pull Request 가이드라인

1. 기능 브랜치 생성: `git checkout -b feature/new-feature`
2. 코드 작성 및 테스트
3. 문서 업데이트
4. PR 생성 및 리뷰 요청

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원

### 문서
- [사용자 가이드](docs/user_guide.md)
- [개발자 가이드](docs/developer_guide.md)
- [API 레퍼런스](docs/api_reference.md)

### 커뮤니티
- [GitHub Issues](https://github.com/your-org/defense-ms-rag/issues)
- [Discord 채널](https://discord.gg/your-channel)
- [이메일 지원](mailto:support@your-org.com)

### 버전 기록

#### v1.0.0 (2025-08-20)
- 초기 릴리스
- 기본 RAG 기능 구현
- 국방 M&S 문서 지원
- CLI 및 웹 인터페이스

#### v1.1.0 (예정)
- 음성 인터페이스 추가
- 다국어 지원
- 고급 분석 기능

---

**🎯 국방 M&S RAG 시스템** - 전문적이고 신뢰할 수 있는 AI 기반 질의응답 솔루션

*이 문서는 지속적으로 업데이트됩니다. 최신 정보는 GitHub 저장소를 확인해주세요.*