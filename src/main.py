"""
국방 M&S RAG 시스템 메인 실행 파일
문서 로드, 임베딩 생성, 질의응답 시스템을 통합한 메인 애플리케이션
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 모듈 임포트
try:
    from rag.document_loader import DefenseDocumentLoader
    from rag.vector_store import DefenseVectorStore
    from rag.retriever import DefenseRAGRetriever
    from models.ollama_client import DefenseOllamaClient
except ImportError as e:
    logger.error(f"모듈 임포트 실패: {e}")
    sys.exit(1)


class DefenseRAGSystem:
    """국방 M&S RAG 시스템 메인 클래스"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        RAG 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        
        # 컴포넌트 초기화
        logger.info("RAG 시스템 초기화 시작...")
        
        try:
            self.document_loader = DefenseDocumentLoader(config_path)
            self.vector_store = DefenseVectorStore(config_path)
            self.retriever = DefenseRAGRetriever(config_path)
            self.llm_client = DefenseOllamaClient(config_path)
            
            logger.info("RAG 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}")
            raise
    
    def load_documents_from_directory(self, directory_path: str) -> bool:
        """디렉토리에서 문서들을 로드하고 벡터 스토어에 추가"""
        try:
            logger.info(f"문서 로드 시작: {directory_path}")
            
            # 디렉토리 존재 확인
            if not Path(directory_path).exists():
                logger.error(f"디렉토리를 찾을 수 없습니다: {directory_path}")
                return False
            
            # 문서 로드
            documents = self.document_loader.load_directory(directory_path)
            
            if not documents:
                logger.warning("로드된 문서가 없습니다.")
                return False
            
            # 벡터 스토어에 추가
            logger.info("문서를 벡터 스토어에 추가 중...")
            doc_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"문서 로드 완료: {len(doc_ids)}개 문서 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"문서 로드 실패: {e}")
            return False
    
    def load_documents_from_mixed_sources(
        self, 
        directory_path: Optional[str] = None,
        urls_file: Optional[str] = None
    ) -> bool:
        """파일과 웹페이지를 혼합하여 로드"""
        try:
            logger.info("혼합 소스 문서 로드 시작")
            
            # 문서 로더 사용하여 혼합 소스 로드
            documents = self.document_loader.load_mixed_sources(directory_path, urls_file)
            
            if not documents:
                logger.warning("로드된 문서가 없습니다.")
                return False
            
            # 벡터 스토어에 추가
            logger.info("문서를 벡터 스토어에 추가 중...")
            doc_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"혼합 소스 로드 완료: {len(doc_ids)}개 문서 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"혼합 소스 로드 실패: {e}")
            return False
    
    def load_urls_from_file(self, urls_file: str) -> bool:
        """URL 파일에서 웹페이지들을 로드"""
        try:
            logger.info(f"URL 파일에서 웹페이지 로드: {urls_file}")
            
            # 문서 로더 사용하여 웹페이지 로드
            documents = self.document_loader.load_urls_from_file(urls_file)
            
            if not documents:
                logger.warning("로드된 웹페이지가 없습니다.")
                return False
            
            # 벡터 스토어에 추가
            doc_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"웹페이지 로드 완료: {len(doc_ids)}개 문서 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"웹페이지 로드 실패: {e}")
            return False
        """단일 문서 로드"""
        try:
            logger.info(f"단일 문서 로드: {file_path}")
            
            # 파일 존재 확인
            if not Path(file_path).exists():
                logger.error(f"파일을 찾을 수 없습니다: {file_path}")
                return False
            
            # 문서 로드
            text = self.document_loader.load_single_file(file_path)
            
            # Document 객체 생성
            from langchain.schema import Document
            file_path_obj = Path(file_path)
            
            metadata = {
                'source': str(file_path_obj),
                'filename': file_path_obj.name,
                'file_type': file_path_obj.suffix.lower().lstrip('.'),
                'file_size': file_path_obj.stat().st_size
            }
            
            # 청크로 분할
            chunks = self.document_loader.text_splitter.split_text(text)
            documents = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            # 벡터 스토어에 추가
            doc_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"단일 문서 로드 완료: {len(doc_ids)}개 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"단일 문서 로드 실패: {e}")
            return False
    
    def ask_question(
        self, 
        question: str, 
        domain: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        try:
            logger.info(f"질문 처리 시작: {question}")
            start_time = time.time()
            
            # 관련 컨텍스트 검색
            context_result = self.retriever.get_relevant_context(
                question, 
                domain_filter=domain
            )
            
            # 답변 생성
            response = self.llm_client.generate_defense_response(
                query=question,
                context=context_result['context'],
                domain=domain or "일반"
            )
            
            total_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'question': question,
                'answer': response['answer'],
                'domain': response.get('domain', '일반'),
                'context_info': {
                    'source_count': context_result['source_count'],
                    'domains': context_result['domains'],
                    'confidence': context_result['confidence']
                },
                'generation_info': {
                    'model': response.get('model', ''),
                    'generation_time': response.get('generation_time', 0),
                    'total_time': total_time
                }
            }
            
            if include_sources:
                result['sources'] = context_result['sources']
                result['context'] = context_result['context']
            
            logger.info(f"질문 처리 완료 - 총 소요 시간: {total_time:.2f}초")
            return result
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {
                'question': question,
                'answer': f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 벡터 스토어 정보
            collection_info = self.vector_store.get_collection_info()
            
            # LLM 모델 정보
            model_info = self.llm_client.get_model_info()
            
            # 헬스 체크
            health_status = self.llm_client.health_check()
            
            # 검색 시스템 통계
            retriever_stats = self.retriever.get_statistics()
            
            return {
                'vector_store': collection_info,
                'llm_model': model_info,
                'health': health_status,
                'retriever': retriever_stats,
                'system': {
                    'config_path': self.config_path,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def reset_knowledge_base(self) -> bool:
        """지식 베이스 재설정"""
        try:
            logger.info("지식 베이스 재설정 시작")
            success = self.vector_store.reset_collection()
            
            if success:
                logger.info("지식 베이스 재설정 완료")
            else:
                logger.error("지식 베이스 재설정 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"지식 베이스 재설정 실패: {e}")
            return False
    
    def export_knowledge_base_info(self, output_path: str) -> bool:
        """지식 베이스 정보 내보내기"""
        try:
            return self.vector_store.export_collection_info(output_path)
        except Exception as e:
            logger.error(f"지식 베이스 정보 내보내기 실패: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """문서 검색 (답변 생성 없이)"""
        try:
            documents = self.retriever.retrieve_documents(
                query=query,
                k=k,
                domain_filter=domain
            )
            
            results = []
            for doc in documents:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('filename', ''),
                    'domain': doc.metadata.get('domain', ''),
                    'chunk_id': doc.metadata.get('chunk_id', 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []


def interactive_mode(rag_system: DefenseRAGSystem):
    """대화형 모드"""
    print("\n" + "="*60)
    print("🎯 국방 M&S RAG 시스템 대화형 모드")
    print("="*60)
    print("명령어:")
    print("  - 질문을 입력하세요")
    print("  - '/status' : 시스템 상태 확인")
    print("  - '/search <쿼리>' : 문서 검색만 실행")
    print("  - '/load <파일경로>' : 단일 문서 로드")
    print("  - '/load-dir <디렉토리경로>' : 디렉토리 문서 로드")
    print("  - '/load-urls <URL파일경로>' : 웹페이지 로드")
    print("  - '/load-mixed <디렉토리> <URL파일>' : 혼합 소스 로드")
    print("  - '/reset' : 지식 베이스 재설정")
    print("  - '/quit' : 종료")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤖 질문을 입력하세요: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("시스템을 종료합니다.")
                break
            
            elif user_input.startswith('/status'):
                print("\n📊 시스템 상태:")
                status = rag_system.get_system_status()
                
                if 'error' not in status:
                    print(f"  - 문서 수: {status['vector_store'].get('document_count', 0)}개")
                    print(f"  - 모델: {status['llm_model'].get('model_name', 'N/A')}")
                    print(f"  - 상태: {status['health'].get('status', 'N/A')}")
                else:
                    print(f"  오류: {status['error']}")
            
            elif user_input.startswith('/search'):
                query = user_input[7:].strip()
                if query:
                    print(f"\n🔍 문서 검색: '{query}'")
                    results = rag_system.search_documents(query)
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            print(f"\n[결과 {i}]")
                            print(f"파일: {result['source']}")
                            print(f"내용: {result['content'][:200]}...")
                    else:
                        print("검색 결과가 없습니다.")
                else:
                    print("검색어를 입력해주세요. 예: /search 시뮬레이션")
            
            elif user_input.startswith('/load-dir'):
                dir_path = user_input[9:].strip()
                if dir_path:
                    print(f"\n📁 디렉토리 문서 로드: {dir_path}")
                    success = rag_system.load_documents_from_directory(dir_path)
                    if success:
                        print("✅ 디렉토리 문서 로드 완료")
                    else:
                        print("❌ 디렉토리 문서 로드 실패")
                else:
                    print("디렉토리 경로를 입력해주세요. 예: /load-dir ./docs/pdfs")
            
            elif user_input.startswith('/load-urls'):
                urls_file = user_input[10:].strip()
                if urls_file:
                    print(f"\n🌐 웹페이지 로드: {urls_file}")
                    success = rag_system.load_urls_from_file(urls_file)
                    if success:
                        print("✅ 웹페이지 로드 완료")
                    else:
                        print("❌ 웹페이지 로드 실패")
                else:
                    print("URL 파일 경로를 입력해주세요. 예: /load-urls ./data/ms_urls.txt")
            
            elif user_input.startswith('/load-mixed'):
                parts = user_input[11:].strip().split()
                if len(parts) >= 2:
                    dir_path, urls_file = parts[0], parts[1]
                    print(f"\n🔗 혼합 소스 로드: {dir_path} + {urls_file}")
                    success = rag_system.load_documents_from_mixed_sources(dir_path, urls_file)
                    if success:
                        print("✅ 혼합 소스 로드 완료")
                    else:
                        print("❌ 혼합 소스 로드 실패")
                else:
                    print("디렉토리와 URL 파일을 입력해주세요. 예: /load-mixed ./docs ./data/ms_urls.txt")
            
            elif user_input.startswith('/reset'):
                confirm = input("⚠️  지식 베이스를 재설정하시겠습니까? (y/N): ")
                if confirm.lower() == 'y':
                    success = rag_system.reset_knowledge_base()
                    if success:
                        print("✅ 지식 베이스 재설정 완료")
                    else:
                        print("❌ 지식 베이스 재설정 실패")
            
            else:
                # 일반 질문 처리
                print(f"\n🤔 질문: {user_input}")
                print("💭 답변 생성 중...")
                
                result = rag_system.ask_question(user_input)
                
                print(f"\n💡 답변:")
                print(result['answer'])
                
                if 'context_info' in result:
                    info = result['context_info']
                    print(f"\n📚 참고 정보:")
                    print(f"  - 참조 문서: {info['source_count']}개")
                    print(f"  - 관련 영역: {', '.join(info['domains'])}")
                    print(f"  - 신뢰도: {info['confidence']:.2f}")
                
                if 'generation_info' in result:
                    gen_info = result['generation_info']
                    print(f"  - 응답 시간: {gen_info['total_time']:.2f}초")
        
        except KeyboardInterrupt:
            print("\n\n시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="국방 M&S RAG 시스템")
    parser.add_argument(
        '--config', 
        default='./config/config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--load-docs', 
        help='문서 디렉토리 경로 (초기 로드용)'
    )
    parser.add_argument(
        '--load-urls',
        help='웹페이지 URL 목록 파일 경로'
    )
    parser.add_argument(
        '--load-mixed',
        nargs=2,
        metavar=('DOCS_DIR', 'URLS_FILE'),
        help='문서 디렉토리와 URL 파일을 함께 로드'
    )
    parser.add_argument(
        '--question', 
        help='단일 질문 (대화형 모드 대신 단일 질문 처리)'
    )
    parser.add_argument(
        '--reset', 
        action='store_true',
        help='지식 베이스 재설정'
    )
    parser.add_argument(
        '--status', 
        action='store_true',
        help='시스템 상태 출력'
    )
    
    args = parser.parse_args()
    
    try:
        # RAG 시스템 초기화
        print("🚀 국방 M&S RAG 시스템 시작...")
        rag_system = DefenseRAGSystem(args.config)
        
        # 지식 베이스 재설정
        if args.reset:
            print("🔄 지식 베이스 재설정...")
            success = rag_system.reset_knowledge_base()
            if success:
                print("✅ 재설정 완료")
            else:
                print("❌ 재설정 실패")
                return
        
        # 문서 로드
        if args.load_docs:
            print(f"📚 문서 로드: {args.load_docs}")
            success = rag_system.load_documents_from_directory(args.load_docs)
            if success:
                print("✅ 문서 로드 완료")
            else:
                print("❌ 문서 로드 실패")
                return
        
        # 웹페이지 로드
        if args.load_urls:
            print(f"🌐 웹페이지 로드: {args.load_urls}")
            success = rag_system.load_urls_from_file(args.load_urls)
            if success:
                print("✅ 웹페이지 로드 완료")
            else:
                print("❌ 웹페이지 로드 실패")
                return
        
        # 혼합 소스 로드
        if args.load_mixed:
            docs_dir, urls_file = args.load_mixed
            print(f"🔗 혼합 소스 로드: {docs_dir} + {urls_file}")
            success = rag_system.load_documents_from_mixed_sources(docs_dir, urls_file)
            if success:
                print("✅ 혼합 소스 로드 완료")
            else:
                print("❌ 혼합 소스 로드 실패")
                return
        
        # 시스템 상태 출력
        if args.status:
            print("📊 시스템 상태:")
            status = rag_system.get_system_status()
            
            if 'error' not in status:
                print(f"  - 벡터 DB: {status['vector_store'].get('document_count', 0)}개 문서")
                print(f"  - LLM 모델: {status['llm_model'].get('model_name', 'N/A')}")
                print(f"  - 상태: {status['health'].get('status', 'N/A')}")
            else:
                print(f"  ❌ 오류: {status['error']}")
            return
        
        # 단일 질문 처리
        if args.question:
            print(f"🤔 질문: {args.question}")
            result = rag_system.ask_question(args.question)
            print(f"\n💡 답변:\n{result['answer']}")
            return
        
        # 대화형 모드
        interactive_mode(rag_system)
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        print(f"❌ 시스템 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()