"""
RAG 시스템 테스트
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

# 상위 디렉토리의 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from rag.document_loader import DefenseDocumentLoader
    from rag.vector_store import DefenseVectorStore, OllamaEmbeddings
    from rag.retriever import DefenseRAGRetriever
    from models.ollama_client import DefenseOllamaClient
except ImportError as e:
    pytest.skip(f"모듈 임포트 실패: {e}", allow_module_level=True)


class TestDefenseDocumentLoader:
    """문서 로더 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """테스트용 설정 파일"""
        config = {
            'document_processing': {
                'chunk_size': 500,
                'chunk_overlap': 100,
                'supported_formats': ['txt', 'pdf', 'md'],
                'text_splitting': {
                    'separators': ["\n\n", "\n", " ", ""]
                }
            }
        }
        
        config_path = Path(temp_dir) / "test_config.yaml"
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    @pytest.fixture
    def sample_text_file(self, temp_dir):
        """샘플 텍스트 파일"""
        content = """국방 모델링 및 시뮬레이션 개요

국방 M&S는 군사 작전을 모델링하고 시뮬레이션하는 기술입니다.
주요 구성 요소는 다음과 같습니다:

1. 전투 모델링
2. 지휘통제 시뮬레이션
3. 효과도 분석
4. 검증 및 확인

HLA(High Level Architecture)는 분산 시뮬레이션 표준입니다.
DIS(Distributed Interactive Simulation)도 중요한 표준 중 하나입니다.

시뮬레이션 종류:
- 구성적 시뮬레이션
- 가상 시뮬레이션
- 실시간 시뮬레이션"""
        
        file_path = Path(temp_dir) / "sample_ms_doc.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)
    
    def test_document_loader_initialization(self, sample_config):
        """문서 로더 초기화 테스트"""
        loader = DefenseDocumentLoader(sample_config)
        
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100
        assert 'txt' in loader.supported_formats
        assert loader.text_splitter is not None
    
    def test_load_text_file(self, sample_config, sample_text_file):
        """텍스트 파일 로드 테스트"""
        loader = DefenseDocumentLoader(sample_config)
        text = loader.load_text_file(sample_text_file)
        
        assert "국방 모델링 및 시뮬레이션" in text
        assert "HLA" in text
        assert "DIS" in text
        assert len(text) > 100
    
    def test_load_single_file(self, sample_config, sample_text_file):
        """단일 파일 로드 테스트"""
        loader = DefenseDocumentLoader(sample_config)
        text = loader.load_single_file(sample_text_file)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "국방" in text
    
    def test_load_directory(self, sample_config, temp_dir):
        """디렉토리 로드 테스트"""
        # 여러 샘플 파일 생성
        samples = [
            ("지상전.txt", "지상전 시뮬레이션은 육군의 주요 작전 영역입니다. 전차와 보병이 중심입니다."),
            ("해상전.txt", "해상전 시뮬레이션은 해군 작전을 다룹니다. 함정과 잠수함이 주요 요소입니다."),
            ("항공전.md", "# 항공전 시뮬레이션\n공군 작전 시뮬레이션입니다. 전투기와 공중전이 핵심입니다.")
        ]
        
        for filename, content in samples:
            file_path = Path(temp_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        loader = DefenseDocumentLoader(sample_config)
        documents = loader.load_directory(temp_dir)
        
        assert len(documents) > 0
        assert all(hasattr(doc, 'page_content') for doc in documents)
        assert all(hasattr(doc, 'metadata') for doc in documents)
        
        # 도메인 식별 확인
        domains = [doc.metadata.get('domain', '') for doc in documents]
        assert '지상전' in domains or '해상전' in domains or '공중전' in domains
    
    def test_identify_domain(self, sample_config):
        """도메인 식별 테스트"""
        loader = DefenseDocumentLoader(sample_config)
        
        assert loader._identify_domain("전차 시뮬레이션", "tank_sim.txt") == "지상전"
        assert loader._identify_domain("함정 작전", "naval_ops.txt") == "해상전"
        assert loader._identify_domain("전투기 시뮬레이션", "fighter_sim.txt") == "공중전"
        assert loader._identify_domain("일반 문서", "general.txt") == "일반"
    
    def test_preprocess_text(self, sample_config):
        """텍스트 전처리 테스트"""
        loader = DefenseDocumentLoader(sample_config)
        
        input_text = "M&S는    중요한   기술입니다.   HLA를  사용합니다."
        processed = loader.preprocess_text(input_text)
        
        assert "M&S(모델링 및 시뮬레이션)" in processed
        assert "HLA(High Level Architecture)" in processed
        assert "  " not in processed  # 중복 공백 제거 확인


class TestOllamaEmbeddings:
    """Ollama 임베딩 테스트"""
    
    @pytest.fixture
    def mock_ollama_client(self, monkeypatch):
        """Ollama 클라이언트 모킹"""
        class MockOllamaClient:
            def __init__(self, host):
                self.host = host
            
            def embeddings(self, model, prompt):
                # 가짜 임베딩 반환 (실제로는 768차원)
                return {'embedding': [0.1] * 768}
        
        import ollama
        monkeypatch.setattr(ollama, 'Client', MockOllamaClient)
        
        return MockOllamaClient
    
    def test_ollama_embeddings_initialization(self, mock_ollama_client):
        """Ollama 임베딩 초기화 테스트"""
        embeddings = OllamaEmbeddings()
        
        assert embeddings.model == "nomic-embed-text"
        assert embeddings.host == "http://localhost:11434"
    
    def test_embed_documents(self, mock_ollama_client):
        """문서 임베딩 테스트"""
        embeddings = OllamaEmbeddings()
        texts = ["문서 1", "문서 2", "문서 3"]
        
        embeddings_result = embeddings.embed_documents(texts)
        
        assert len(embeddings_result) == 3
        assert len(embeddings_result[0]) == 768
        assert all(isinstance(emb, list) for emb in embeddings_result)
    
    def test_embed_query(self, mock_ollama_client):
        """쿼리 임베딩 테스트"""
        embeddings = OllamaEmbeddings()
        query = "시뮬레이션이란 무엇인가?"
        
        embedding_result = embeddings.embed_query(query)
        
        assert len(embedding_result) == 768
        assert isinstance(embedding_result, list)


class TestDefenseRAGRetriever:
    """RAG 검색기 테스트"""
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """테스트용 설정 파일"""
        config = {
            'rag': {
                'retrieval': {
                    'top_k': 3,
                    'search_type': 'similarity',
                    'score_threshold': 0.7
                },
                'generation': {
                    'max_context_length': 1000
                }
            },
            'vector_db': {
                'persist_directory': str(Path(temp_dir) / 'chroma_db'),
                'collection_name': 'test_docs'
            },
            'ollama': {
                'host': 'http://localhost:11434',
                'embedding_model': 'nomic-embed-text'
            }
        }
        
        config_path = Path(temp_dir) / "test_config.yaml"
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_retriever_initialization(self, sample_config):
        """검색기 초기화 테스트"""
        # 이 테스트는 실제 Ollama 서버가 필요하므로 스킵할 수 있음
        pytest.skip("실제 Ollama 서버 필요")
    
    def test_preprocess_query(self, sample_config):
        """쿼리 전처리 테스트"""
        pytest.skip("실제 Ollama 서버 필요")
    
    def test_identify_query_domain(self, sample_config):
        """쿼리 도메인 식별 테스트"""
        pytest.skip("실제 Ollama 서버 필요")
    
    def test_enhance_query(self, sample_config):
        """쿼리 강화 테스트"""
        pytest.skip("실제 Ollama 서버 필요")


class TestDefenseOllamaClient:
    """Ollama 클라이언트 테스트"""
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """테스트용 설정"""
        config = {
            'ollama': {
                'host': 'http://localhost:11434',
                'model': 'llama3:8b',
                'temperature': 0.7,
                'max_tokens': 1000,
                'timeout': 30
            },
            'rag': {
                'generation': {
                    'system_prompt_template': '당신은 국방 M&S 전문가입니다.',
                    'user_prompt_template': '질문: {question}\n컨텍스트: {context}\n답변:'
                }
            }
        }
        
        config_path = Path(temp_dir) / "test_config.yaml"
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_ollama_client_initialization(self, sample_config):
        """Ollama 클라이언트 초기화 테스트"""
        # 실제 Ollama 서버가 필요하므로 모킹하거나 스킵
        pytest.skip("실제 Ollama 서버 필요")
    
    def test_generate_response(self, sample_config):
        """응답 생성 테스트"""
        pytest.skip("실제 Ollama 서버 필요")
    
    def test_get_domain_specific_prompt(self, sample_config):
        """도메인별 프롬프트 테스트"""
        pytest.skip("실제 Ollama 서버 필요")


class TestIntegration:
    """통합 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integration_config(self, temp_dir):
        """통합 테스트용 설정"""
        config = {
            'document_processing': {
                'chunk_size': 500,
                'chunk_overlap': 100,
                'supported_formats': ['txt', 'md'],
                'text_splitting': {
                    'separators': ["\n\n", "\n", " ", ""]
                }
            },
            'vector_db': {
                'persist_directory': str(Path(temp_dir) / 'chroma_db'),
                'collection_name': 'integration_test'
            },
            'ollama': {
                'host': 'http://localhost:11434',
                'model': 'llama3:8b',
                'embedding_model': 'nomic-embed-text',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'rag': {
                'retrieval': {
                    'top_k': 3,
                    'search_type': 'similarity',
                    'score_threshold': 0.7
                },
                'generation': {
                    'max_context_length': 1000,
                    'system_prompt_template': '당신은 국방 M&S 전문가입니다.',
                    'user_prompt_template': '질문: {question}\n컨텍스트: {context}\n답변:'
                }
            }
        }
        
        config_path = Path(temp_dir) / "integration_config.yaml"
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    def test_end_to_end_workflow(self, integration_config, temp_dir):
        """전체 워크플로우 테스트"""
        pytest.skip("실제 Ollama 서버와 완전한 환경 필요")
        
        # 이 테스트는 다음 단계를 포함해야 함:
        # 1. 문서 로드
        # 2. 벡터 스토어에 저장
        # 3. 질문에 대한 검색
        # 4. 답변 생성
        # 5. 결과 검증


# 테스트 실행을 위한 유틸리티 함수들
def test_with_real_ollama():
    """실제 Ollama 서버를 사용한 테스트"""
    try:
        import ollama
        client = ollama.Client()
        # 간단한 연결 테스트
        client.list()
        return True
    except:
        return False


def run_minimal_test():
    """최소한의 기능 테스트"""
    print("🧪 기본 기능 테스트 시작...")
    
    # 1. 설정 파일 테스트
    try:
        import yaml
        config = {
            'test': 'value',
            'nested': {'key': 'value'}
        }
        yaml.dump(config, open('/tmp/test_config.yaml', 'w'))
        loaded = yaml.safe_load(open('/tmp/test_config.yaml', 'r'))
        assert loaded['test'] == 'value'
        print("✅ YAML 설정 파일 처리 성공")
    except Exception as e:
        print(f"❌ YAML 처리 실패: {e}")
    
    # 2. JSON 처리 테스트
    try:
        test_data = {'key': 'value', 'number': 123}
        json_str = json.dumps(test_data, ensure_ascii=False)
        loaded_data = json.loads(json_str)
        assert loaded_data['key'] == 'value'
        print("✅ JSON 처리 성공")
    except Exception as e:
        print(f"❌ JSON 처리 실패: {e}")
    
    # 3. 파일 시스템 테스트
    try:
        test_dir = Path('/tmp/defense_rag_test')
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / 'test.txt'
        test_file.write_text('테스트 내용', encoding='utf-8')
        
        content = test_file.read_text(encoding='utf-8')
        assert '테스트' in content
        
        shutil.rmtree(test_dir)
        print("✅ 파일 시스템 처리 성공")
    except Exception as e:
        print(f"❌ 파일 시스템 처리 실패: {e}")
    
    print("🧪 기본 기능 테스트 완료")


if __name__ == "__main__":
    run_minimal_test()