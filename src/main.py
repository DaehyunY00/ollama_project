"""
국방 M&S RAG 시스템 메인 실행 파일
문서 로드, 임베딩 생성, 질의응답 시스템을 통합한 메인 애플리케이션
하이브리드 검색 기능 지원
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
    """국방 M&S RAG 시스템 메인 클래스 (하이브리드 검색 지원)"""
    
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
    
    def load_documents_from_directory(self, directory_path: str, rebuild_bm25: bool = True) -> bool:
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
            
            # BM25 인덱스 재구축 (새 문서 추가 시)
            if rebuild_bm25 and hasattr(self.retriever, 'rebuild_bm25_index'):
                logger.info("BM25 인덱스 재구축 중...")
                self.retriever.rebuild_bm25_index()
            
            logger.info(f"문서 로드 완료: {len(doc_ids)}개 문서 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"문서 로드 실패: {e}")
            return False
    
    def load_documents_from_mixed_sources(
        self, 
        directory_path: Optional[str] = None,
        urls_file: Optional[str] = None,
        rebuild_bm25: bool = True
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
            
            # BM25 인덱스 재구축
            if rebuild_bm25 and hasattr(self.retriever, 'rebuild_bm25_index'):
                logger.info("BM25 인덱스 재구축 중...")
                self.retriever.rebuild_bm25_index()
            
            logger.info(f"혼합 소스 로드 완료: {len(doc_ids)}개 문서 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"혼합 소스 로드 실패: {e}")
            return False
    
    def load_urls_from_file(self, urls_file: str, rebuild_bm25: bool = True) -> bool:
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
            
            # BM25 인덱스 재구축
            if rebuild_bm25 and hasattr(self.retriever, 'rebuild_bm25_index'):
                logger.info("BM25 인덱스 재구축 중...")
                self.retriever.rebuild_bm25_index()
            
            logger.info(f"웹페이지 로드 완료: {len(doc_ids)}개 문서 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"웹페이지 로드 실패: {e}")
            return False
    
    def load_single_file(self, file_path: str, rebuild_bm25: bool = True) -> bool:
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
            
            # BM25 인덱스 재구축
            if rebuild_bm25 and hasattr(self.retriever, 'rebuild_bm25_index'):
                logger.info("BM25 인덱스 재구축 중...")
                self.retriever.rebuild_bm25_index()
            
            logger.info(f"단일 문서 로드 완료: {len(doc_ids)}개 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"단일 문서 로드 실패: {e}")
            return False
    
    def ask_question(
        self, 
        question: str, 
        domain: Optional[str] = None,
        include_sources: bool = True,
        use_hybrid: Optional[bool] = None,  # 새로 추가
        k: Optional[int] = None             # 새로 추가
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            domain: 특정 도메인 (선택사항)
            include_sources: 소스 정보 포함 여부
            use_hybrid: 하이브리드 검색 사용 여부 (None=자동, True=강제사용, False=비사용)
            k: 검색할 문서 수 (선택사항)
        
        Returns:
            Dict: 답변과 메타데이터
        """
        try:
            start_time = time.time()
            
            logger.info(f"질문 처리 시작: {question}")
            logger.info(f"도메인 필터: {domain}, 하이브리드: {use_hybrid}, 문서수: {k}")
            
            # 1. 관련 문서 검색 (하이브리드 옵션 포함)
            context_data = self.retriever.get_relevant_context(
                question, 
                k=k,
                domain_filter=domain,
                use_hybrid=use_hybrid  # 하이브리드 검색 옵션 전달
            )
            
            # 2. 컨텍스트 확인
            if context_data['source_count'] == 0:
                return {
                    'question': question,
                    'answer': "죄송합니다. 관련된 문서를 찾을 수 없어 답변을 드릴 수 없습니다.",
                    'sources': [],
                    'domains': [],
                    'confidence': 0.0,
                    'search_method': context_data.get('search_method', 'unknown'),
                    'response_time': time.time() - start_time,
                    'context_info': {
                        'source_count': 0,
                        'domains': [],
                        'confidence': 0.0
                    }
                }
            
            # 3. LLM을 사용하여 답변 생성
            response = self.llm_client.generate_defense_response(
                query=question,
                context=context_data['context'],
                domain=domain or "일반"
            )
            
            total_time = time.time() - start_time
            
            # 4. 응답 구성
            result = {
                'question': question,
                'answer': response.get('answer', '답변 생성에 실패했습니다.'),
                'domain': response.get('domain', '일반'),
                'context_info': {
                    'source_count': context_data['source_count'],
                    'domains': context_data['domains'],
                    'confidence': context_data['confidence']
                },
                'search_info': {
                    'search_method': context_data.get('search_method', 'unknown'),
                    'hybrid_used': context_data.get('search_method') == '하이브리드',
                    'documents_found': context_data['source_count']
                },
                'generation_info': {
                    'model': response.get('model', ''),
                    'generation_time': response.get('generation_time', 0),
                    'total_time': total_time
                }
            }
            
            if include_sources:
                result['sources'] = context_data['sources']
                result['context'] = context_data['context']
            
            logger.info(f"답변 생성 완료 ({result['search_info']['search_method']} 검색, {total_time:.2f}초)")
            return result
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {
                'question': question,
                'answer': f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                'sources': [],
                'domains': [],
                'confidence': 0.0,
                'search_method': 'error',
                'response_time': time.time() - start_time if 'start_time' in locals() else 0,
                'error': str(e)
            }
    
    def compare_search_methods(self, question: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """벡터 vs 하이브리드 검색 성능 비교"""
        try:
            logger.info(f"검색 방법 비교 시작: {question}")
            
            # 벡터 검색만 사용
            start_time = time.time()
            vector_result = self.ask_question(question, domain, use_hybrid=False, include_sources=False)
            vector_time = time.time() - start_time
            
            # 하이브리드 검색 사용
            start_time = time.time()
            hybrid_result = self.ask_question(question, domain, use_hybrid=True, include_sources=False)
            hybrid_time = time.time() - start_time
            
            return {
                'question': question,
                'domain': domain,
                'vector_search': {
                    'response_time': vector_time,
                    'source_count': vector_result['context_info']['source_count'],
                    'confidence': vector_result['context_info']['confidence'],
                    'domains_found': vector_result['context_info']['domains']
                },
                'hybrid_search': {
                    'response_time': hybrid_time,
                    'source_count': hybrid_result['context_info']['source_count'],
                    'confidence': hybrid_result['context_info']['confidence'],
                    'domains_found': hybrid_result['context_info']['domains']
                },
                'performance_diff': {
                    'time_delta': hybrid_time - vector_time,
                    'time_ratio': (hybrid_time / vector_time) if vector_time > 0 else 1.0,
                    'source_diff': hybrid_result['context_info']['source_count'] - vector_result['context_info']['source_count'],
                    'confidence_diff': hybrid_result['context_info']['confidence'] - vector_result['context_info']['confidence']
                }
            }
            
        except Exception as e:
            logger.error(f"검색 방법 비교 실패: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회 (하이브리드 정보 포함)"""
        try:
            # 벡터 스토어 정보
            collection_info = self.vector_store.get_collection_info()
            
            # LLM 모델 정보
            model_info = self.llm_client.get_model_info()
            
            # 헬스 체크
            health_status = self.llm_client.health_check()
            
            # 검색 시스템 통계 (하이브리드 정보 포함)
            retriever_stats = self.retriever.get_statistics()
            
            # 하이브리드 검색 상태 확인
            validation = self.retriever.validate_setup()
            
            return {
                'vector_store': collection_info,
                'llm_model': model_info,
                'health': health_status,
                'retriever': retriever_stats,
                'hybrid_status': {
                    'bm25_enabled': retriever_stats.get('use_bm25', False),
                    'bm25_index_size': retriever_stats.get('bm25_index_size', 0),
                    'hybrid_alpha': retriever_stats.get('hybrid_alpha', 0.7),
                    'search_ready': validation.get('hybrid_search_ready', False)
                },
                'system': {
                    'config_path': self.config_path,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def reset_knowledge_base(self, rebuild_bm25: bool = True) -> bool:
        """지식 베이스 재설정"""
        try:
            logger.info("지식 베이스 재설정 시작")
            success = self.vector_store.reset_collection()
            
            if success and rebuild_bm25:
                # BM25 인덱스도 초기화
                if hasattr(self.retriever, '_initialize_bm25'):
                    self.retriever._initialize_bm25()
                    logger.info("BM25 인덱스도 초기화됨")
            
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
    
    def search_documents(
        self, 
        query: str, 
        k: int = 5, 
        domain: Optional[str] = None,
        use_hybrid: Optional[bool] = None  # 새로 추가
    ) -> List[Dict[str, Any]]:
        """
        문서 검색 (답변 생성 없이)
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            domain: 도메인 필터
            use_hybrid: 하이브리드 검색 사용 여부
        """
        try:
            # 하이브리드 옵션을 포함한 문서 검색
            documents = self.retriever.retrieve_documents(
                query=query,
                k=k,
                domain_filter=domain,
                use_hybrid=use_hybrid  # 하이브리드 옵션 전달
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
            
            # 검색 방법 정보 추가
            search_method = 'hybrid' if use_hybrid else ('auto' if use_hybrid is None else 'vector')
            logger.info(f"문서 검색 완료: {len(results)}개 문서 ({search_method} 방식)")
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def get_search_explanation(self, query: str) -> Dict[str, Any]:
        """검색 과정 설명 (디버깅용)"""
        try:
            return self.retriever.get_search_explanation(query)
        except Exception as e:
            logger.error(f"검색 설명 생성 실패: {e}")
            return {'error': str(e)}


def interactive_mode(rag_system: DefenseRAGSystem):
    """대화형 모드 (하이브리드 명령어 추가)"""
    print("\n" + "="*60)
    print("🎯 국방 M&S RAG 시스템 대화형 모드")
    print("="*60)
    print("명령어:")
    print("  - 질문을 입력하세요")
    print("  - '/status' : 시스템 상태 확인")
    print("  - '/search <쿼리>' : 문서 검색만 실행")
    print("  - '/hybrid <쿼리>' : 하이브리드 검색 강제 사용")  # 새로 추가
    print("  - '/vector <쿼리>' : 벡터 검색만 사용")         # 새로 추가
    print("  - '/compare <쿼리>' : 검색 방법 성능 비교")      # 새로 추가
    print("  - '/explain <쿼리>' : 검색 과정 설명")          # 새로 추가
    print("  - '/load <파일경로>' : 단일 문서 로드")
    print("  - '/load-dir <디렉토리경로>' : 디렉토리 문서 로드")
    print("  - '/load-urls <URL파일경로>' : 웹페이지 로드")
    print("  - '/load-mixed <디렉토리> <URL파일>' : 혼합 소스 로드")
    print("  - '/reset' : 지식 베이스 재설정")
    print("  - '/quit' : 종료")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤖 입력하세요: ").strip()
            
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
                    print(f"  - LLM 모델: {status['llm_model'].get('model_name', 'N/A')}")
                    print(f"  - 상태: {status['health'].get('status', 'N/A')}")
                    
                    # 하이브리드 상태 표시
                    hybrid_info = status.get('hybrid_status', {})
                    print(f"  - 하이브리드 검색: {'활성' if hybrid_info.get('bm25_enabled', False) else '비활성'}")
                    print(f"  - BM25 인덱스: {hybrid_info.get('bm25_index_size', 0)}개 문서")
                    print(f"  - 가중치 비율: {hybrid_info.get('hybrid_alpha', 0.7):.1f}:{ 1-hybrid_info.get('hybrid_alpha', 0.7):.1f}")
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
                            print(f"도메인: {result['domain']}")
                            print(f"내용: {result['content'][:200]}...")
                    else:
                        print("검색 결과가 없습니다.")
                else:
                    print("검색어를 입력해주세요. 예: /search 시뮬레이션")
            
            elif user_input.startswith('/hybrid'):
                query = user_input[7:].strip()
                if query:
                    print(f"\n🔹 하이브리드 검색: '{query}'")
                    results = rag_system.search_documents(query, use_hybrid=True)
                    
                    if results:
                        print(f"📚 하이브리드 검색 결과: {len(results)}개 문서")
                        for i, result in enumerate(results, 1):
                            print(f"  [{i}] {result['source']} (도메인: {result['domain']})")
                    else:
                        print("하이브리드 검색 결과가 없습니다.")
                else:
                    print("검색어를 입력해주세요. 예: /hybrid HLA 아키텍처")
            
            elif user_input.startswith('/vector'):
                query = user_input[7:].strip()
                if query:
                    print(f"\n🔸 벡터 검색: '{query}'")
                    results = rag_system.search_documents(query, use_hybrid=False)
                    
                    if results:
                        print(f"📚 벡터 검색 결과: {len(results)}개 문서")
                        for i, result in enumerate(results, 1):
                            print(f"  [{i}] {result['source']} (도메인: {result['domain']})")
                    else:
                        print("벡터 검색 결과가 없습니다.")
                else:
                    print("검색어를 입력해주세요. 예: /vector 시뮬레이션 검증")
            
            elif user_input.startswith('/compare'):
                query = user_input[8:].strip()
                if query:
                    print(f"\n⚖️ 검색 방법 비교: '{query}'")
                    comparison = rag_system.compare_search_methods(query)
                    
                    if 'error' not in comparison:
                        vec_info = comparison['vector_search']
                        hyb_info = comparison['hybrid_search']
                        perf_diff = comparison['performance_diff']
                        
                        print(f"\n🔸 벡터 검색:")
                        print(f"  응답시간: {vec_info['response_time']:.2f}초")
                        print(f"  문서 수: {vec_info['source_count']}개")
                        print(f"  신뢰도: {vec_info['confidence']:.2f}")
                        
                        print(f"\n🔹 하이브리드 검색:")
                        print(f"  응답시간: {hyb_info['response_time']:.2f}초")
                        print(f"  문서 수: {hyb_info['source_count']}개")
                        print(f"  신뢰도: {hyb_info['confidence']:.2f}")
                        
                        print(f"\n📊 성능 차이:")
                        print(f"  시간 차이: {perf_diff['time_delta']:+.2f}초")
                        print(f"  문서 수 차이: {perf_diff['source_diff']:+d}개")
                        print(f"  신뢰도 차이: {perf_diff['confidence_diff']:+.2f}")
                    else:
                        print(f"비교 실패: {comparison['error']}")
                else:
                    print("검색어를 입력해주세요. 예: /compare HLA 아키텍처")
            
            elif user_input.startswith('/explain'):
                query = user_input[8:].strip()
                if query:
                    print(f"\n🔍 검색 과정 설명: '{query}'")
                    explanation = rag_system.get_search_explanation(query)
                    
                    if 'error' not in explanation:
                        print(f"원본 쿼리: {explanation.get('original_query', '')}")
                        print(f"전처리된 쿼리: {explanation.get('preprocessed_query', '')}")
                        print(f"식별된 도메인: {explanation.get('identified_domains', [])}")
                        print(f"검색 방법: {explanation.get('search_method', '')}")
                        if explanation.get('enhanced_query'):
                            print(f"확장된 쿼리: {explanation['enhanced_query']}")
                        if explanation.get('hybrid_ratio'):
                            print(f"하이브리드 비율: {explanation['hybrid_ratio']}")
                        print(f"인덱스된 문서: {explanation.get('total_indexed_docs', 0)}개")
                    else:
                        print(f"설명 생성 실패: {explanation['error']}")
                else:
                    print("검색어를 입력해주세요. 예: /explain 전투 시뮬레이션")
            
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
                # 일반 질문 처리 (하이브리드 기본 활성화)
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
                
                if 'search_info' in result:
                    search_info = result['search_info']
                    print(f"  - 검색 방법: {search_info['search_method']}")
                
                if 'generation_info' in result:
                    gen_info = result['generation_info']
                    print(f"  - 응답 시간: {gen_info['total_time']:.2f}초")
        
        except KeyboardInterrupt:
            print("\n\n시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


def main():
    """메인 함수 (하이브리드 옵션 추가)"""
    parser = argparse.ArgumentParser(description="국방 M&S RAG 시스템 (하이브리드 검색 지원)")
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
        '--domain',
        help='도메인 필터 (지상전, 해상전, 공중전, 우주전, 사이버전, 합동작전)'
    )
    parser.add_argument(
        '--hybrid', 
        action='store_true',
        help='하이브리드 검색 강제 활성화'
    )
    parser.add_argument(
        '--vector-only', 
        action='store_true',
        help='벡터 검색만 사용 (하이브리드 비활성화)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='벡터 vs 하이브리드 검색 성능 비교'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='검색할 문서 수 (기본값: 5)'
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
                
                # 하이브리드 상태 정보
                hybrid_info = status.get('hybrid_status', {})
                print(f"  - 하이브리드 검색: {'활성' if hybrid_info.get('bm25_enabled', False) else '비활성'}")
                print(f"  - BM25 인덱스: {hybrid_info.get('bm25_index_size', 0)}개 문서")
                print(f"  - 가중치 비율: {hybrid_info.get('hybrid_alpha', 0.7):.1f}:{1-hybrid_info.get('hybrid_alpha', 0.7):.1f}")
            else:
                print(f"  ❌ 오류: {status['error']}")
            return
        
        # 검색 방법 결정
        use_hybrid = None
        if args.hybrid:
            use_hybrid = True
            print("🔹 하이브리드 검색 모드 활성화")
        elif args.vector_only:
            use_hybrid = False
            print("🔸 벡터 검색 전용 모드 활성화")
        
        # 단일 질문 처리
        if args.question:
            print(f"🤔 질문: {args.question}")
            
            # 성능 비교 모드
            if args.compare:
                print("⚖️ 검색 방법 성능 비교 중...")
                comparison = rag_system.compare_search_methods(args.question, args.domain)
                
                if 'error' not in comparison:
                    vec_info = comparison['vector_search']
                    hyb_info = comparison['hybrid_search']
                    perf_diff = comparison['performance_diff']
                    
                    print(f"\n🔸 벡터 검색:")
                    print(f"  응답시간: {vec_info['response_time']:.2f}초")
                    print(f"  문서 수: {vec_info['source_count']}개")
                    print(f"  신뢰도: {vec_info['confidence']:.2f}")
                    
                    print(f"\n🔹 하이브리드 검색:")
                    print(f"  응답시간: {hyb_info['response_time']:.2f}초")
                    print(f"  문서 수: {hyb_info['source_count']}개") 
                    print(f"  신뢰도: {hyb_info['confidence']:.2f}")
                    
                    print(f"\n📊 성능 차이:")
                    time_change = ((perf_diff['time_ratio'] - 1) * 100)
                    print(f"  시간 변화: {time_change:+.1f}%")
                    print(f"  문서 수 차이: {perf_diff['source_diff']:+d}개")
                    print(f"  신뢰도 차이: {perf_diff['confidence_diff']:+.2f}")
                else:
                    print(f"비교 실패: {comparison['error']}")
            else:
                # 일반 질문 처리
                result = rag_system.ask_question(
                    args.question, 
                    domain=args.domain,
                    use_hybrid=use_hybrid,
                    k=args.top_k
                )
                
                print(f"\n💡 답변:")
                print(result['answer'])
                
                # 상세 정보 출력
                if 'search_info' in result:
                    search_info = result['search_info']
                    print(f"\n🔍 검색 정보:")
                    print(f"  - 검색 방법: {search_info['search_method']}")
                    print(f"  - 하이브리드 사용: {'예' if search_info['hybrid_used'] else '아니오'}")
                
                if 'context_info' in result:
                    context_info = result['context_info']
                    print(f"📚 참고 정보:")
                    print(f"  - 참조 문서: {context_info['source_count']}개")
                    print(f"  - 관련 영역: {', '.join(context_info['domains'])}")
                    print(f"  - 신뢰도: {context_info['confidence']:.2f}")
                
                if 'generation_info' in result:
                    gen_info = result['generation_info']
                    print(f"⏱️ 성능 정보:")
                    print(f"  - 총 응답시간: {gen_info['total_time']:.2f}초")
                    print(f"  - LLM 생성시간: {gen_info.get('generation_time', 0):.2f}초")
            
            return
        
        # 대화형 모드
        interactive_mode(rag_system)
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        print(f"❌ 시스템 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()