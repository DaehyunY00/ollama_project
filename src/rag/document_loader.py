"""
국방 M&S RAG 시스템용 문서 로더 모듈
다양한 형식의 문서를 로드하고 전처리하는 기능 제공
"""

import os
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import yaml
from urllib.parse import urlparse

# LangChain 문서 로더들
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    WebBaseLoader
)

logger = logging.getLogger(__name__)


class DefenseDocumentLoader:
    """국방 M&S 문서 로더 클래스"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        문서 로더 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # 설정에서 파라미터 로드
        doc_config = self.config.get('document_loader', {})
        
        # 텍스트 분할기 설정
        chunk_size = doc_config.get('chunk_size', 1000)
        chunk_overlap = doc_config.get('chunk_overlap', 200)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 지원 형식
        self.supported_formats = doc_config.get('supported_formats', [
            'pdf', 'txt', 'md', 'docx', 'webpage'
        ])
        
        logger.info(f"문서 로더 초기화 완료 - 지원 형식: {self.supported_formats}")
    
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
            'document_loader': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'supported_formats': ['pdf', 'txt', 'md', 'docx', 'webpage']
            }
        }
    
    def load_single_file(self, file_path: str) -> str:
        """단일 파일 로드"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            file_extension = file_path.suffix.lower().lstrip('.')
            
            if file_extension == 'pdf':
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                text = "\n\n".join([page.page_content for page in pages])
                
            elif file_extension == 'txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                text = documents[0].page_content
                
            elif file_extension == 'md':
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
                text = documents[0].page_content
                
            elif file_extension == 'docx':
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
                text = documents[0].page_content
                
            else:
                # 텍스트 파일로 시도
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # UTF-8이 안되면 다른 인코딩 시도
                    with open(file_path, 'r', encoding='cp949') as f:
                        text = f.read()
            
            logger.debug(f"파일 로드 완료: {file_path} ({len(text)} 문자)")
            return text
            
        except Exception as e:
            logger.error(f"파일 로드 실패 {file_path}: {e}")
            return ""
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """디렉토리의 모든 지원 파일 로드"""
        try:
            directory = Path(directory_path)
            
            if not directory.exists():
                logger.error(f"디렉토리를 찾을 수 없습니다: {directory_path}")
                return []
            
            documents = []
            supported_extensions = [f".{fmt}" for fmt in self.supported_formats if fmt != 'webpage']
            
            # 재귀적으로 파일 찾기
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        text = self.load_single_file(str(file_path))
                        
                        if text.strip():
                            # 파일 메타데이터
                            metadata = {
                                'source': str(file_path),
                                'filename': file_path.name,
                                'file_type': file_path.suffix.lower().lstrip('.'),
                                'file_size': file_path.stat().st_size,
                                'directory': str(file_path.parent),
                                'domain': self._extract_domain_from_filename(file_path.name)
                            }
                            
                            # 텍스트를 청크로 분할
                            chunks = self.text_splitter.split_text(text)
                            
                            for i, chunk in enumerate(chunks):
                                chunk_metadata = metadata.copy()
                                chunk_metadata['chunk_id'] = i
                                chunk_metadata['total_chunks'] = len(chunks)
                                
                                documents.append(Document(
                                    page_content=chunk,
                                    metadata=chunk_metadata
                                ))
                        
                    except Exception as e:
                        logger.warning(f"파일 로드 실패 {file_path}: {e}")
            
            logger.info(f"디렉토리 로드 완료: {len(documents)}개 문서 청크")
            return documents
            
        except Exception as e:
            logger.error(f"디렉토리 로드 실패: {e}")
            return []
    
    def load_webpage(self, url: str) -> List[Document]:
        """웹페이지 로드"""
        try:
            logger.debug(f"웹페이지 로드 시작: {url}")
            
            # URL 유효성 검사
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"유효하지 않은 URL: {url}")
            
            # 웹페이지 로드
            loader = WebBaseLoader([url])
            documents = loader.load()
            
            processed_documents = []
            
            for doc in documents:
                # 메타데이터 추가
                metadata = doc.metadata.copy()
                metadata.update({
                    'source': url,
                    'source_type': 'webpage',
                    'domain': self._extract_domain_from_url(url)
                })
                
                # 텍스트 청크로 분할
                chunks = self.text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_id'] = i
                    chunk_metadata['total_chunks'] = len(chunks)
                    
                    processed_documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            
            logger.info(f"웹페이지 로드 완료: {url} ({len(processed_documents)} 청크)")
            return processed_documents
            
        except Exception as e:
            logger.error(f"웹페이지 로드 실패 {url}: {e}")
            return []
    
    def load_urls_from_file(self, urls_file: str) -> List[Document]:
        """URL 파일에서 여러 웹페이지 로드"""
        try:
            urls_path = Path(urls_file)
            
            if not urls_path.exists():
                logger.error(f"URL 파일을 찾을 수 없습니다: {urls_file}")
                return []
            
            # URL 목록 읽기
            urls = []
            with open(urls_path, 'r', encoding='utf-8') as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith('#'):  # 주석 제외
                        urls.append(url)
            
            if not urls:
                logger.warning(f"URL 파일에 유효한 URL이 없습니다: {urls_file}")
                return []
            
            # 모든 웹페이지 로드
            all_documents = []
            for url in urls:
                try:
                    docs = self.load_webpage(url)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"URL 로드 실패 {url}: {e}")
            
            logger.info(f"URL 파일 로드 완료: {len(urls)}개 URL, {len(all_documents)}개 문서 청크")
            return all_documents
            
        except Exception as e:
            logger.error(f"URL 파일 로드 실패: {e}")
            return []
    
    def load_mixed_sources(
        self, 
        directory_path: Optional[str] = None,
        urls_file: Optional[str] = None
    ) -> List[Document]:
        """파일과 웹페이지를 혼합하여 로드"""
        try:
            all_documents = []
            
            # 디렉토리에서 파일 로드
            if directory_path:
                logger.info(f"디렉토리에서 문서 로드: {directory_path}")
                file_docs = self.load_directory(directory_path)
                all_documents.extend(file_docs)
                logger.info(f"파일 문서 로드 완료: {len(file_docs)}개 청크")
            
            # URL 파일에서 웹페이지 로드
            if urls_file:
                logger.info(f"URL 파일에서 웹페이지 로드: {urls_file}")
                web_docs = self.load_urls_from_file(urls_file)
                all_documents.extend(web_docs)
                logger.info(f"웹페이지 로드 완료: {len(web_docs)}개 청크")
            
            logger.info(f"혼합 소스 로드 총계: {len(all_documents)}개 문서 청크")
            return all_documents
            
        except Exception as e:
            logger.error(f"혼합 소스 로드 실패: {e}")
            return []
    
    def _extract_domain_from_filename(self, filename: str) -> str:
        """파일명에서 도메인 추출"""
        filename_lower = filename.lower()
        
        # 국방 M&S 관련 키워드 매핑
        domain_keywords = {
            'simulation': '시뮬레이션',
            'modeling': '모델링',
            'combat': '전투효과도',
            'weapon': '무기체계',
            'analysis': '작전분석',
            'training': '교육훈련',
            'doctrine': '교리',
            'strategy': '전략',
            'tactics': '전술',
            'logistics': '군수',
            'c4i': 'C4I',
            'intel': '정보',
            'cyber': '사이버',
            'electronic': '전자전'
        }
        
        for keyword, domain in domain_keywords.items():
            if keyword in filename_lower:
                return domain
        
        return '일반'
    
    def _extract_domain_from_url(self, url: str) -> str:
        """URL에서 도메인 추출"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # 도메인별 분류
            if 'mnd.go.kr' in domain:
                return '국방부'
            elif 'add.re.kr' in domain:
                return '국방과학연구소'
            elif 'dtaq.re.kr' in domain:
                return '국방기술품질원'
            elif 'wikipedia' in domain:
                return '위키피디아'
            elif 'kida.re.kr' in domain:
                return '국방연구원'
            elif any(keyword in domain for keyword in ['simulation', 'modeling', 'defense']):
                return '시뮬레이션'
            else:
                return '웹자료'
                
        except Exception:
            return '웹자료'
    
    def get_loader_statistics(self) -> Dict[str, Any]:
        """로더 통계 정보 반환"""
        return {
            'supported_formats': self.supported_formats,
            'chunk_size': self.text_splitter._chunk_size,
            'chunk_overlap': self.text_splitter._chunk_overlap,
            'separators': self.text_splitter._separators
        }