#!/usr/bin/env python3
"""
국방 M&S RAG 시스템 데모 스크립트
웹페이지와 PDF 문서를 로드하여 RAG 시스템을 테스트합니다.
"""

import sys
import os
from pathlib import Path
import yaml
import time

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from rag.document_loader import DefenseDocumentLoader
    from rag.vector_store import DefenseVectorStore
    from rag.retriever import DefenseRAGRetriever
    from models.ollama_client import DefenseOllamaClient
    from main import DefenseRAGSystem
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("src 디렉토리의 모든 필요한 모듈이 있는지 확인하세요.")
    sys.exit(1)


def create_demo_config():
    """데모용 설정 파일 생성"""
    config = {
        'project': {
            'name': 'Defense M&S RAG Demo',
            'version': '1.0.0'
        },
        'ollama': {
            'host': 'http://localhost:11434',
            'model': 'llama3:8b',
            'embedding_model': 'nomic-embed-text',
            'temperature': 0.7,
            'max_tokens': 2048,
            'timeout': 60
        },
        'vector_db': {
            'type': 'chromadb',
            'persist_directory': './data/demo_chroma_db',
            'collection_name': 'demo_defense_docs',
            'embedding_dimension': 768,
            'similarity_threshold': 0.7
        },
        'document_processing': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'supported_formats': ['pdf', 'txt', 'md', 'docx', 'webpage'],
            'text_splitting': {
                'method': 'recursive',
                'separators': ["\n\n", "\n", " ", ""]
            }
        },
        'rag': {
            'retrieval': {
                'top_k': 5,
                'search_type': 'similarity',
                'score_threshold': 0.7
            },
            'generation': {
                'max_context_length': 4000
            }
        }
    }
    
    # 데모 설정 파일 저장
    demo_config_path = project_root / 'config' / 'demo_config.yaml'
    demo_config_path.parent.mkdir(exist_ok=True)
    
    with open(demo_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return str(demo_config_path)


def create_sample_urls_file():
    """국방 M&S 관련 웹페이지 URL 목록 생성"""
    urls = [
        # 국방부 및 관련 기관
        "https://www.mnd.go.kr",
        "https://www.add.re.kr",
        # 국방 M&S 관련 학술/연구 자료
        "https://www.kimst.re.kr",  # 한국군사과학기술학회
        # 추가 URL들을 여기에 넣을 수 있음
    ]
    
    urls_file = project_root / 'data' / 'sample_urls.txt'
    urls_file.parent.mkdir(exist_ok=True)
    
    with open(urls_file, 'w', encoding='utf-8') as f:
        f.write("# 국방 M&S 관련 웹페이지 URL 목록\n")
        f.write("# '#'으로 시작하는 줄은 주석으로 처리됩니다\n\n")
        for url in urls:
            f.write(f"{url}\n")
    
    return str(urls_file)


def create_sample_questions():
    """국방 M&S 관련 샘플 질문들"""
    return [
        {
            'question': 'HLA(High Level Architecture) 표준이란 무엇인가요?',
            'domain': '일반',
            'description': 'M&S 기본 표준에 대한 질문'
        },
        {
            'question': '전투 효과도 분석 방법론에 대해 설명해주세요',
            'domain': '일반',
            'description': '전투 분석 방법론'
        },
        {
            'question': 'K2 전차의 시뮬레이션 모델링 시 주요 고려사항은?',
            'domain': '지상전',
            'description': '지상전 무기체계 모델링'
        },
        {
            'question': '해상전 시뮬레이션에서 환경 요소는 어떻게 반영되나요?',
            'domain': '해상전',
            'description': '해상 환경 모델링'
        },
        {
            'question': '공중전 교전 시뮬레이션의 핵심 요소들을 설명해주세요',
            'domain': '공중전',
            'description': '공중전 시뮬레이션'
        },
        {
            'question': '시뮬레이션 검증(Verification)과 확인(Validation)의 차이점은?',
            'domain': '일반',
            'description': 'VV&A 기본 개념'
        },
        {
            'question': '분산 시뮬레이션에서 상호운용성 확보 방안은?',
            'domain': '합동작전',
            'description': '시스템 통합 및 상호운용성'
        }
    ]


def check_ollama_status():
    """Ollama 서버 상태 확인"""
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        
        print("✅ Ollama 서버 연결 성공")
        print(f"   사용 가능한 모델: {len(models['models'])}개")
        
        # llama3:8b 모델 확인
        model_names = [model['name'] for model in models['models']]
        if 'llama3:8b' in model_names:
            print("✅ llama3:8b 모델 확인")
        else:
            print("⚠️  llama3:8b 모델이 없습니다. 다음 명령어로 다운로드하세요:")
            print("   ollama pull llama3:8b")
        
        # 임베딩 모델 확인
        if 'nomic-embed-text' in model_names:
            print("✅ nomic-embed-text 모델 확인")
        else:
            print("⚠️  nomic-embed-text 모델이 없습니다. 다음 명령어로 다운로드하세요:")
            print("   ollama pull nomic-embed-text")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama 서버 연결 실패: {e}")
        print("Ollama가 설치되어 있고 서비스가 실행 중인지 확인하세요.")
        return False


def demo_document_loading():
    """문서 로딩 데모"""
    print("\n" + "="*60)
    print("📚 문서 로딩 데모")
    print("="*60)
    
    config_path = create_demo_config()
    loader = DefenseDocumentLoader(config_path)
    
    # 1. 로컬 파일 로딩 테스트
    docs_dir = project_root / 'docs' / 'pdfs'
    if docs_dir.exists() and list(docs_dir.glob('*.pdf')):
        print(f"📄 로컬 PDF 파일 로딩 테스트: {docs_dir}")
        try:
            documents = loader.load_directory(str(docs_dir))
            print(f"✅ 성공: {len(documents)}개 문서 청크 로드")
        except Exception as e:
            print(f"❌ 실패: {e}")
    else:
        print("📄 로컬 PDF 파일이 없어 건너뛰기")
    
    # 2. 웹페이지 로딩 테스트 (소규모)
    print("\n🌐 웹페이지 로딩 테스트")
    try:
        # 간단한 테스트 페이지
        test_url = "https://www.mnd.go.kr"
        text = loader.load_webpage(test_url)
        print(f"✅ 웹페이지 로드 성공: {len(text)}자")
        print(f"   미리보기: {text[:100]}...")
    except Exception as e:
        print(f"❌ 웹페이지 로드 실패: {e}")
    
    return config_path


def demo_rag_system():
    """RAG 시스템 전체 데모"""
    print("\n" + "="*60)
    print("🧠 RAG 시스템 데모")
    print("="*60)
    
    config_path = create_demo_config()
    
    try:
        # RAG 시스템 초기화
        print("🚀 RAG 시스템 초기화 중...")
        rag_system = DefenseRAGSystem(config_path)
        print("✅ RAG 시스템 초기화 완료")
        
        # 샘플 문서 로드 (있는 경우에만)
        docs_dir = project_root / 'docs'
        sample_docs = []
        
        if docs_dir.exists():
            for ext in ['*.pdf', '*.txt', '*.md']:
                sample_docs.extend(docs_dir.rglob(ext))
        
        if sample_docs:
            print(f"📚 샘플 문서 로드 중... ({len(sample_docs)}개 파일)")
            success = rag_system.load_documents_from_directory(str(docs_dir))
            if success:
                print("✅ 문서 로드 완료")
            else:
                print("⚠️  문서 로드 중 일부 오류 발생")
        else:
            print("⚠️  샘플 문서가 없어서 기본 지식으로만 테스트합니다")
        
        # 샘플 질문 테스트
        questions = create_sample_questions()
        
        print(f"\n🤔 샘플 질문 테스트 ({len(questions)}개)")
        for i, q_data in enumerate(questions[:3], 1):  # 처음 3개만 테스트
            print(f"\n--- 질문 {i} ---")
            print(f"영역: {q_data['domain']}")
            print(f"질문: {q_data['question']}")
            print(f"설명: {q_data['description']}")
            
            try:
                start_time = time.time()
                result = rag_system.ask_question(
                    q_data['question'], 
                    domain=q_data['domain'] if q_data['domain'] != '일반' else None
                )
                end_time = time.time()
                
                print(f"\n💡 답변 (응답시간: {end_time - start_time:.2f}초):")
                print(result['answer'])
                
                if 'context_info' in result:
                    info = result['context_info']
                    print(f"\n📊 참고 정보:")
                    print(f"   - 참조 문서: {info['source_count']}개")
                    print(f"   - 관련 영역: {', '.join(info['domains'])}")
                    print(f"   - 신뢰도: {info['confidence']:.2f}")
                
            except Exception as e:
                print(f"❌ 질문 처리 실패: {e}")
            
            print("-" * 50)
        
        # 시스템 상태 출력
        print(f"\n📊 시스템 상태:")
        status = rag_system.get_system_status()
        if 'error' not in status:
            print(f"   - 벡터 DB 문서 수: {status['vector_store'].get('document_count', 0)}개")
            print(f"   - LLM 모델: {status['llm_model'].get('model_name', 'N/A')}")
            print(f"   - 상태: {status['health'].get('status', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG 시스템 데모 실패: {e}")
        return False


def interactive_demo():
    """대화형 데모"""
    print("\n" + "="*60)
    print("💬 대화형 RAG 테스트")
    print("="*60)
    print("직접 질문을 입력해보세요. 'quit'을 입력하면 종료됩니다.")
    
    config_path = create_demo_config()
    
    try:
        rag_system = DefenseRAGSystem(config_path)
        
        while True:
            user_question = input("\n🤔 질문을 입력하세요: ").strip()
            
            if user_question.lower() in ['quit', 'exit', '종료']:
                print("데모를 종료합니다.")
                break
            
            if not user_question:
                continue
            
            try:
                result = rag_system.ask_question(user_question)
                print(f"\n💡 답변:")
                print(result['answer'])
                
                if 'generation_info' in result:
                    gen_info = result['generation_info']
                    print(f"\n⏱️  응답 시간: {gen_info['total_time']:.2f}초")
                
            except Exception as e:
                print(f"❌ 답변 생성 실패: {e}")
    
    except Exception as e:
        print(f"❌ 대화형 데모 초기화 실패: {e}")


def main():
    """메인 함수"""
    print("🎯 국방 M&S RAG 시스템 데모")
    print("=" * 60)
    
    # 1. 환경 확인
    print("🔍 환경 확인 중...")
    if not check_ollama_status():
        print("\n❌ Ollama 서버 연결 실패로 데모를 중단합니다.")
        return
    
    # 2. 메뉴 선택
    while True:
        print("\n📋 데모 메뉴:")
        print("1. 문서 로딩 테스트")
        print("2. RAG 시스템 전체 데모")
        print("3. 대화형 테스트")
        print("4. 종료")
        
        choice = input("\n선택하세요 (1-4): ").strip()
        
        if choice == '1':
            demo_document_loading()
        elif choice == '2':
            demo_rag_system()
        elif choice == '3':
            interactive_demo()
        elif choice == '4':
            print("데모를 종료합니다.")
            break
        else:
            print("올바른 번호를 선택해주세요.")
    
    print("\n🎉 데모가 완료되었습니다!")


if __name__ == "__main__":
    main()