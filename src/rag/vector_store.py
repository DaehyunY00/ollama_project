"""
국방 M&S RAG 시스템용 벡터 스토어 모듈
Chroma DB를 사용한 문서 임베딩 저장 및 검색
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

# LangChain 최신 import 방식 사용
try:
    from langchain_chroma import Chroma
except ImportError:
    # 백업용 - 구버전 호환성
    from langchain_community.vectorstores import Chroma

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ollama 임베딩 (최신 버전)
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    # 백업용 - 구버전 호환성
    from langchain_community.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


class DefenseVectorStore:
    """국방 M&S 문서용 벡터 스토어 클래스"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        벡터 스토어 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.vector_store = None
        self.embeddings = None
        
        # 설정에서 파라미터 로드
        vector_config = self.config.get('vector_store', {})
        self.persist_directory = vector_config.get('persist_directory', './data/chroma_db')
        self.collection_name = vector_config.get('collection_name', 'defense_ms_docs')
        
        # 임베딩 설정
        embedding_config = self.config.get('embeddings', {})
        self.embedding_model = embedding_config.get('model', 'nomic-embed-text')
        
        # 디렉토리 생성
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 초기화
        self._initialize_embeddings()
        self._initialize_vector_store()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'vector_store': {
                'persist_directory': './data/chroma_db',
                'collection_name': 'defense_ms_docs'
            },
            'embeddings': {
                'model': 'nomic-embed-text',
                'base_url': 'http://localhost:11434'
            }
        }
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            embedding_config = self.config.get('embeddings', {})
            base_url = embedding_config.get('base_url', 'http://localhost:11434')
            
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=base_url
            )
            
            # 임베딩 테스트
            test_result = self.embeddings.embed_query("테스트")
            logger.info(f"임베딩 모델 초기화 완료: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            raise
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화 (기존 DB 처리 개선)"""
        try:
            # 기존 컬렉션 확인 및 안전한 로드
            if self._collection_exists():
                try:
                    # 기존 컬렉션을 로드 시도
                    self.vector_store = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    logger.info(f"기존 벡터 스토어 로드 완료: {self.collection_name}")
                    
                except Exception as load_error:
                    logger.warning(f"기존 벡터 스토어 로드 실패: {load_error}")
                    logger.info("새로운 벡터 스토어를 생성합니다...")
                    
                    # 기존 데이터 백업 후 삭제
                    self._backup_and_reset()
                    
                    # 새 벡터 스토어 생성
                    self.vector_store = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    logger.info("새 벡터 스토어 생성 완료")
            else:
                # 새 벡터 스토어 생성
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"새 벡터 스토어 생성 완료: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 실패: {e}")
            raise
    
    def _collection_exists(self) -> bool:
        """컬렉션 존재 여부 확인"""
        db_path = Path(self.persist_directory)
        return db_path.exists() and any(db_path.iterdir())
    
    def _backup_and_reset(self):
        """기존 데이터 백업 후 리셋"""
        try:
            backup_path = f"{self.persist_directory}_backup_{int(time.time())}"
            if Path(self.persist_directory).exists():
                shutil.move(self.persist_directory, backup_path)
                logger.info(f"기존 데이터를 백업했습니다: {backup_path}")
            
            # 새 디렉토리 생성
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"백업 및 리셋 실패: {e}")
            # 강제로 삭제
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """문서를 벡터 스토어에 추가"""
        try:
            if not documents:
                logger.warning("추가할 문서가 없습니다.")
                return []
            
            # 메타데이터 전처리
            processed_docs = []
            for doc in documents:
                # 메타데이터에서 None 값 제거
                clean_metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if v is not None and str(v).strip()
                }
                
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata=clean_metadata
                )
                processed_docs.append(processed_doc)
            
            # 문서 추가
            doc_ids = self.vector_store.add_documents(processed_docs)
            
            logger.info(f"문서 {len(doc_ids)}개를 벡터 스토어에 추가했습니다.")
            return doc_ids
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """유사도 검색"""
        try:
            if not self.vector_store:
                logger.error("벡터 스토어가 초기화되지 않았습니다.")
                return []
            
            # 필터가 있는 경우
            if filter:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.debug(f"검색 결과: {len(results)}개 문서")
            return results
            
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """점수와 함께 유사도 검색"""
        try:
            if not self.vector_store:
                logger.error("벡터 스토어가 초기화되지 않았습니다.")
                return []
            
            if filter:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            return results
            
        except Exception as e:
            logger.error(f"점수 포함 검색 실패: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            if not self.vector_store:
                return {'error': '벡터 스토어가 초기화되지 않음'}
            
            # Chroma 클라이언트에서 정보 가져오기
            collection = self.vector_store._collection
            
            info = {
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'embedding_model': self.embedding_model,
                'document_count': collection.count() if collection else 0,
                'status': 'active'
            }
            
            return info
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {'error': str(e)}
    
    def reset_collection(self) -> bool:
        """컬렉션 재설정"""
        try:
            # 기존 벡터 스토어 정리
            if self.vector_store:
                try:
                    self.vector_store._client.reset()
                except:
                    pass
            
            # 디렉토리 삭제 및 재생성
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)
            
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # 벡터 스토어 재초기화
            self._initialize_vector_store()
            
            logger.info("컬렉션 재설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 재설정 실패: {e}")
            return False
    
    def export_collection_info(self, output_path: str) -> bool:
        """컬렉션 정보를 파일로 내보내기"""
        try:
            info = self.get_collection_info()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(info, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"컬렉션 정보를 {output_path}에 저장했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 정보 내보내기 실패: {e}")
            return False


# time 모듈 import 추가
import time