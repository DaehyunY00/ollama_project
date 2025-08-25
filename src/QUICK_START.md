# 🚀 국방 M&S RAG 시스템 빠른 시작 가이드

## 📋 준비사항

### 1. 시스템 요구사항
- **Python 3.11** 이상
- **RAM**: 16GB 이상 권장
- **디스크**: 10GB 이상 여유 공간

### 2. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (https://ollama.ai/download)
# Windows/macOS: 공식 사이트에서 설치 프로그램 다운로드
# Linux: 
curl -fsSL https://ollama.ai/install.sh | sh

# 필요한 모델 다운로드
ollama pull llama3:8b
ollama pull nomic-embed-text

# 설치 확인
ollama list
```

## 🏗️ 프로젝트 설정

### 1. 가상환경 생성 및 활성화

```bash
# 프로젝트 디렉토리로 이동
cd ollama_project

# Python 3.11로 ollama 가상환경 생성
python3.11 -m venv ollama

# 가상환경 활성화
# Windows
ollama\Scripts\activate
# macOS/Linux
source ollama/bin/activate
```

### 2. 패키지 설치

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt
```

## 🎯 RAG 시스템 테스트

### 방법 1: 데모 스크립트 실행 (추천)

```bash
# 데모 스크립트 실행
python scripts/rag_demo.py

# 메뉴에서 선택:
# 1. 문서 로딩 테스트
# 2. RAG 시스템 전체 데모  ← 이것을 선택하세요
# 3. 대화형 테스트
# 4. 종료
```

### 방법 2: 메인 시스템 사용

#### Step 1: 시스템 상태 확인
```bash
python src/main.py --status
```

#### Step 2: 문서 로드

**PDF 문서가 있는 경우:**
```bash
# docs/pdfs 디렉토리에 PDF 파일들을 넣고
python src/main.py --load-docs ./docs/pdfs
```

**웹페이지 로드:**
```bash
# 미리 준비된 URL 목록 사용
python src/main.py --load-urls ./data/ms_urls.txt
```

**혼합 소스 로드:**
```bash
# 로컬 문서 + 웹페이지 함께 로드
python src/main.py --load-mixed ./docs/pdfs ./data/ms_urls.txt
```

#### Step 3: 대화형 질의응답
```bash
python src/main.py
```

### 방법 3: 단일 질문 테스트

```bash
python src/main.py --question "HLA 표준이란 무엇인가요?"
```

## 📚 샘플 질문들

시스템을 테스트할 때 다음 질문들을 사용해보세요:

### 기본 M&S 개념
- "M&S란 무엇인가요?"
- "HLA 표준의 주요 구성요소는 무엇인가요?"
- "시뮬레이션 검증과 확인의 차이점은?"
- "분산 시뮬레이션의 장단점은?"

### 도메인별 질문
- "K2 전차의 시뮬레이션 모델링 시 고려사항은?" (지상전)
- "해상전 시뮬레이션에서 환경 요소는 어떻게 반영되나요?" (해상전)
- "공중전 교전 시뮬레이션의 핵심 요소들을 설명해주세요" (공중전)

### 기술적 질문
- "전투 효과도 분석 방법론에 대해 설명해주세요"
- "VV&A 프로세스의 단계별 절차는?"
- "RTI의 역할과 기능은 무엇인가요?"

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. "Connection refused" 오류
```bash
# Ollama 서비스 확인 및 재시작
ollama serve
```

#### 2. 모델을 찾을 수 없음
```bash
# 모델 재다운로드
ollama pull llama3:8b
ollama pull nomic-embed-text
```

#### 3. 메모리 부족
- `config/config.yaml`에서 `chunk_size`를 1000에서 500으로 줄이기
- 더 작은 모델 사용 (llama3.2:3b)

#### 4. 웹페이지 로드 실패
- 인터넷 연결 확인
- 방화벽 설정 확인
- URL 파일의 경로 확인

## 📊 시스템 모니터링

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/rag_system.log

# 오류만 확인
grep ERROR logs/rag_system.log
```

### 벡터 데이터베이스 상태
```bash
# 대화형 모드에서 '/status' 명령어 사용
python src/main.py
> /status
```

## 🎨 사용자 정의

### 1. 새로운 웹페이지 URL 추가
`data/ms_urls.txt` 파일을 편집하여 새로운 URL 추가:

```text
# 새로운 국방 M&S 관련 사이트
https://your-new-site.com
https://another-ms-site.org
```

### 2. 프롬프트 수정
`config/config.yaml`에서 시스템 프롬프트 커스터마이징

### 3. 청크 크기 조정
메모리나 성능에 따라 `document_processing.chunk_size` 조정

## 📈 성능 최적화

### GPU 사용 (NVIDIA)
```bash
# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 메모리 최적화
```yaml
# config.yaml에서
document_processing:
  chunk_size: 500        # 기본: 1000
  chunk_overlap: 100     # 기본: 200

rag:
  retrieval:
    top_k: 3            # 기본: 5
```

## 🆘 추가 도움말

### 커뮤니티 및 지원
- GitHub Issues: 버그 리포트 및 기능 요청
- README.md: 상세한 사용법 및 API 문서
- 이메일 지원: 기술적 문의

### 다음 단계
1. 실제 국방 M&S 문서로 테스트
2. 도메인별 특화 질문 시도
3. 웹 인터페이스 사용 (예정)
4. API 통합 (예정)

---

**🎯 이제 국방 M&S RAG 시스템을 사용할 준비가 완료되었습니다!**

문제가 발생하면 `scripts/rag_demo.py`를 먼저 실행해보세요.