"""
RAG 검색 시스템
문서 검색, 컨텍스트 생성, 답변 생성을 담당하는 모듈
하이브리드 검색 (벡터 + BM25) 지원
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import yaml
import re
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict

from langchain.schema import Document

# 프로젝트 모듈
from .vector_store import DefenseVectorStore

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseRAGRetriever:
    """국방 M&S 문서용 RAG 검색기 (하이브리드 검색 지원)"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        RAG 검색기 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # 검색 설정
        self.top_k = self.config['rag']['retrieval']['top_k']
        self.search_type = self.config['rag']['retrieval']['search_type']
        self.score_threshold = self.config['rag']['retrieval']['score_threshold']
        self.max_context_length = self.config['rag']['generation']['max_context_length']
        
        # 하이브리드 검색 설정
        self.hybrid_alpha = self.config['rag']['retrieval'].get('hybrid_alpha', 0.7)
        self.use_bm25 = self.config['rag']['retrieval'].get('use_bm25', True)
        self.use_reranking = self.config['rag']['retrieval'].get('use_reranking', False)
        
        # 벡터 스토어 초기화
        self.vector_store = DefenseVectorStore(config_path)
        
        # 국방 M&S 도메인 키워드
        self.domain_keywords = self._load_domain_keywords()
        
        # BM25 인덱스 관련
        self.bm25_index = None
        self.documents = []
        self.document_texts = []
        
        # BM25 인덱스 초기화
        if self.use_bm25:
            self._initialize_bm25()
        
        logger.info(f"RAG 검색기 초기화 완료 (하이브리드: {self.use_bm25}, 가중치: {self.hybrid_alpha})")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'rag': {
                'retrieval': {
                    'top_k': 5,
                    'search_type': 'hybrid',
                    'score_threshold': 0.7,
                    'hybrid_alpha': 0.7,
                    'use_bm25': True,
                    'use_reranking': False
                },
                'generation': {
                    'max_context_length': 4000
                }
            }
        }
    
    def _initialize_bm25(self):
        """BM25 인덱스 초기화"""
        try:
            logger.info("BM25 인덱스 구축 시작...")
            
            # 벡터 스토어에서 모든 문서 가져오기
            self.documents = self.vector_store.get_all_documents()
            
            if not self.documents:
                logger.warning("BM25 인덱스 구축을 위한 문서가 없습니다.")
                self.use_bm25 = False
                return
            
            # 문서 텍스트 추출
            self.document_texts = [doc.page_content for doc in self.documents]
            
            # 토큰화 (한국어 + 영어 혼합 처리)
            tokenized_docs = []
            for text in self.document_texts:
                tokens = self._tokenize_text(text)
                tokenized_docs.append(tokens)
            
            # BM25 인덱스 생성
            self.bm25_index = BM25Okapi(tokenized_docs)
            
            logger.info(f"BM25 인덱스 구축 완료: {len(self.documents)}개 문서")
            
        except Exception as e:
            logger.error(f"BM25 인덱스 초기화 실패: {e}")
            self.use_bm25 = False
    
    def _tokenize_text(self, text: str) -> List[str]:
        """텍스트 토큰화 (국방 M&S 도메인 특화)"""
        # 기본 공백 기반 토큰화
        tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            # 특수문자 제거 및 소문자 변환
            cleaned_token = re.sub(r'[^\w가-힣]', '', token).lower()
            
            if len(cleaned_token) > 1:  # 한 글자 토큰 제외
                processed_tokens.append(cleaned_token)
                
                # 영문 약어의 경우 원본도 추가
                if re.match(r'^[a-z]+$', cleaned_token) and len(cleaned_token) <= 5:
                    processed_tokens.append(cleaned_token.upper())
        
        return processed_tokens
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """도메인별 키워드 사전 로드"""
        return {
            '지상전': [
                '지상작전', '육군', '전차', '장갑차', '보병', '기갑', '포병',
                '지상군', '육상전투', '기동작전', '방어작전', '공격작전'
            ],
            '해상전': [
                '해상작전', '해군', '함정', '잠수함', '해상전투', '함대',
                '해전', '수상함', '해상봉쇄', '상륙작전'
            ],
            '공중전': [
                '공중작전', '공군', '항공기', '전투기', '공중전투', '항공',
                '공중우세', '공대공', '공대지', '항공우세'
            ],
            '우주전': [
                '우주작전', '위성', '우주전', '우주자산', '우주감시',
                '우주상황인식', '우주군'
            ],
            '사이버전': [
                '사이버작전', '사이버전', '정보전', '전자전', '사이버보안',
                '사이버공격', '사이버방어', '정보보호'
            ],
            '합동작전': [
                '합동작전', '연합작전', '통합작전', '합동', '연합',
                '다영역작전', '통합화력', '합동기동'
            ]
        }
    
    def preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 기본 전처리
        query = query.strip()
        
        # 국방 M&S 용어 정규화
        replacements = {
            r'M&S|m&s': '모델링 및 시뮬레이션',
            r'VV&A|vv&a': '검증 확인 및 인정',
            r'HLA|hla': 'High Level Architecture',
            r'DIS|dis': 'Distributed Interactive Simulation',
            r'C4I|c4i': 'Command Control Communications Computers Intelligence'
        }
        
        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def identify_query_domain(self, query: str) -> List[str]:
        """쿼리에서 관련 도메인 식별"""
        query_lower = query.lower()
        identified_domains = []
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    if domain not in identified_domains:
                        identified_domains.append(domain)
                    break
        
        return identified_domains if identified_domains else ['일반']
    
    def enhance_query(self, query: str) -> str:
        """쿼리 강화 (동의어, 관련 용어 추가)"""
        enhanced_terms = []
        
        # 원본 쿼리
        enhanced_terms.append(query)
        
        # 도메인 관련 용어 추가
        domains = self.identify_query_domain(query)
        for domain in domains:
            if domain in self.domain_keywords:
                # 각 도메인의 주요 키워드 일부 추가
                domain_terms = self.domain_keywords[domain][:3]
                enhanced_terms.extend(domain_terms)
        
        # 일반적인 M&S 용어 추가
        if '시뮬레이션' in query.lower():
            enhanced_terms.extend(['모델링', '분석', '평가'])
        if '모델링' in query.lower():
            enhanced_terms.extend(['시뮬레이션', '모델', '구현'])
        if '검증' in query.lower():
            enhanced_terms.extend(['확인', '인정', 'VV&A', '테스트'])
        
        return ' '.join(set(enhanced_terms))
    
    def vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """벡터 검색 실행"""
        try:
            if self.search_type == "similarity_with_score":
                results = self.vector_store.similarity_search_with_score(query, k)
                return results
            else:
                documents = self.vector_store.similarity_search(query, k)
                # 점수를 1.0으로 설정 (점수가 없는 경우)
                return [(doc, 1.0) for doc in documents]
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def bm25_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """BM25 키워드 검색 실행"""
        try:
            if not self.use_bm25 or not self.bm25_index:
                return []
            
            # 쿼리 토큰화
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                return []
            
            # BM25 점수 계산
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 상위 k개 문서 선택
            top_indices = np.argsort(scores)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and scores[idx] > 0:
                    results.append((self.documents[idx], float(scores[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 검색 실패: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int) -> List[Document]:
        """하이브리드 검색 (벡터 + BM25)"""
        try:
            # 1. 벡터 검색
            vector_results = self.vector_search(query, k * 2)
            
            # 2. BM25 검색 (활성화된 경우만)
            bm25_results = []
            if self.use_bm25:
                bm25_results = self.bm25_search(query, k * 2)
            
            # 3. 결과 결합
            if not self.use_bm25 or not bm25_results:
                # BM25가 비활성화되었거나 결과가 없으면 벡터 검색만 사용
                return [doc for doc, score in vector_results[:k]]
            
            # 4. 하이브리드 점수 계산 및 결합
            return self._combine_search_results(vector_results, bm25_results, k)
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            # 폴백: 벡터 검색만 사용
            return self.vector_store.similarity_search(query, k)
    
    def _combine_search_results(
        self, 
        vector_results: List[Tuple[Document, float]], 
        bm25_results: List[Tuple[Document, float]], 
        k: int
    ) -> List[Document]:
        """검색 결과 결합"""
        combined_scores = defaultdict(float)
        all_docs = {}
        
        # 벡터 검색 점수 정규화 및 추가
        if vector_results:
            max_vector_score = max(score for _, score in vector_results)
            if max_vector_score > 0:
                for doc, score in vector_results:
                    doc_key = self._get_document_key(doc)
                    all_docs[doc_key] = doc
                    normalized_score = score / max_vector_score
                    combined_scores[doc_key] += self.hybrid_alpha * normalized_score
        
        # BM25 점수 정규화 및 추가
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            if max_bm25_score > 0:
                for doc, score in bm25_results:
                    doc_key = self._get_document_key(doc)
                    all_docs[doc_key] = doc
                    normalized_score = score / max_bm25_score
                    combined_scores[doc_key] += (1 - self.hybrid_alpha) * normalized_score
        
        # 점수순 정렬
        sorted_items = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 상위 k개 문서 반환
        result_docs = []
        for doc_key, score in sorted_items[:k]:
            if doc_key in all_docs:
                result_docs.append(all_docs[doc_key])
        
        return result_docs
    
    def _get_document_key(self, doc: Document) -> str:
        """문서 고유 키 생성"""
        # 문서 내용의 첫 100자를 키로 사용 (중복 방지)
        content_key = doc.page_content[:100].strip()
        
        # 메타데이터가 있으면 추가 정보 포함
        if doc.metadata:
            filename = doc.metadata.get('filename', '')
            chunk_id = doc.metadata.get('chunk_id', '')
            if filename or chunk_id:
                content_key += f"_{filename}_{chunk_id}"
        
        return content_key
    
    def retrieve_documents(
        self, 
        query: str, 
        k: Optional[int] = None,
        domain_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        use_hybrid: Optional[bool] = None
    ) -> List[Document]:
        """문서 검색 (하이브리드 옵션 추가)"""
        try:
            # 기본값 설정
            k = k or self.top_k
            if use_hybrid is None:
                use_hybrid = (self.search_type == 'hybrid' and self.use_bm25)
            
            # 쿼리 전처리
            processed_query = self.preprocess_query(query)
            logger.info(f"검색 쿼리: '{processed_query}' (하이브리드: {use_hybrid})")
            
            # 도메인/파일 타입 필터링 검색 (기존 로직 유지)
            if domain_filter:
                documents = self.vector_store.search_by_domain(processed_query, domain_filter, k)
            elif file_type_filter:
                documents = self.vector_store.search_by_file_type(processed_query, file_type_filter, k)
            else:
                # 하이브리드 검색 또는 일반 검색
                if use_hybrid:
                    documents = self.hybrid_search(processed_query, k)
                else:
                    # 기존 벡터 검색 로직
                    if self.search_type == "similarity_with_score":
                        results = self.vector_store.similarity_search_with_score(
                            processed_query, k, self.score_threshold
                        )
                        documents = [doc for doc, score in results]
                    else:
                        documents = self.vector_store.similarity_search(processed_query, k)
            
            # 쿼리 강화 후 추가 검색 (결과가 부족한 경우)
            if len(documents) < k // 2:
                enhanced_query = self.enhance_query(processed_query)
                if enhanced_query != processed_query:
                    logger.info("쿼리 강화 후 추가 검색 실행")
                    
                    if use_hybrid:
                        additional_docs = self.hybrid_search(enhanced_query, k)
                    else:
                        additional_docs = self.vector_store.similarity_search(enhanced_query, k)
                    
                    # 중복 제거하면서 추가
                    existing_contents = {doc.page_content[:100] for doc in documents}
                    for doc in additional_docs:
                        if doc.page_content[:100] not in existing_contents and len(documents) < k:
                            documents.append(doc)
                            existing_contents.add(doc.page_content[:100])
            
            logger.info(f"최종 검색된 문서 수: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def create_context(self, documents: List[Document], max_length: Optional[int] = None) -> str:
        """검색된 문서들로부터 컨텍스트 생성"""
        if not documents:
            return ""
        
        max_length = max_length or self.max_context_length
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # 문서 헤더 생성
            source = doc.metadata.get('filename', '알 수 없는 파일')
            domain = doc.metadata.get('domain', '일반')
            chunk_id = doc.metadata.get('chunk_id', 0)
            
            header = f"[문서 {i+1}: {source} - {domain} 영역 - 구간 {chunk_id+1}]"
            content = f"{header}\n{doc.page_content}\n"
            
            # 길이 제한 확인
            if current_length + len(content) > max_length:
                if current_length == 0:  # 첫 번째 문서도 너무 긴 경우
                    # 잘라서 포함
                    remaining = max_length - len(header) - 10
                    truncated_content = doc.page_content[:remaining] + "..."
                    content = f"{header}\n{truncated_content}\n"
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        context = "\n".join(context_parts)
        
        # 컨텍스트 품질 정보 추가
        search_method = "하이브리드" if self.use_bm25 else "벡터"
        quality_info = f"\n[검색 결과 ({search_method}): 총 {len(documents)}개 문서에서 {len(context_parts)}개 구간 선택]"
        
        return context + quality_info
    
    def get_relevant_context(
        self, 
        query: str, 
        k: Optional[int] = None,
        domain_filter: Optional[str] = None,
        include_metadata: bool = True,
        use_hybrid: Optional[bool] = None
    ) -> Dict[str, Any]:
        """쿼리에 대한 관련 컨텍스트 생성 (메타데이터 포함)"""
        try:
            # 문서 검색
            documents = self.retrieve_documents(query, k, domain_filter, use_hybrid=use_hybrid)
            
            if not documents:
                return {
                    'context': "관련 문서를 찾을 수 없습니다.",
                    'source_count': 0,
                    'domains': [],
                    'sources': [],
                    'confidence': 0.0,
                    'search_method': '하이브리드' if use_hybrid else '벡터'
                }
            
            # 컨텍스트 생성
            context = self.create_context(documents)
            
            # 메타데이터 분석
            sources = list(set([doc.metadata.get('filename', '') for doc in documents]))
            domains = list(set([doc.metadata.get('domain', '') for doc in documents]))
            
            # 신뢰도 계산
            confidence = min(1.0, len(documents) / self.top_k)
            
            result = {
                'context': context,
                'source_count': len(sources),
                'domains': domains,
                'sources': sources,
                'confidence': confidence,
                'search_method': '하이브리드' if (use_hybrid or self.use_bm25) else '벡터'
            }
            
            if include_metadata:
                result['documents'] = documents
            
            return result
            
        except Exception as e:
            logger.error(f"컨텍스트 생성 실패: {e}")
            return {
                'context': "컨텍스트 생성 중 오류가 발생했습니다.",
                'source_count': 0,
                'domains': [],
                'sources': [],
                'confidence': 0.0,
                'search_method': 'error'
            }
    
    def get_domain_specific_context(self, query: str, domain: str) -> Dict[str, Any]:
        """특정 도메인에 특화된 컨텍스트 검색"""
        return self.get_relevant_context(query, domain_filter=domain)
    
    def search_similar_questions(self, query: str, k: int = 3) -> List[str]:
        """유사한 질문 찾기 (FAQ 기능)"""
        try:
            # 질문 형태의 문서나 제목 검색
            documents = self.retrieve_documents(f"질문 {query}", k)
            
            similar_questions = []
            for doc in documents:
                # 문서에서 질문 형태의 텍스트 추출
                content = doc.page_content
                
                # 간단한 질문 패턴 매칭
                question_patterns = [
                    r'질문[:\s]*([^.\n]*\?)',
                    r'Q[:\s]*([^.\n]*\?)',
                    r'([^.\n]*는\s+무엇[^.\n]*\?)',
                    r'([^.\n]*어떻게[^.\n]*\?)',
                    r'([^.\n]*왜[^.\n]*\?)'
                ]
                
                for pattern in question_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    for match in matches:
                        question = match.strip()
                        if len(question) > 10 and question not in similar_questions:
                            similar_questions.append(question)
                        if len(similar_questions) >= k:
                            break
                    if len(similar_questions) >= k:
                        break
            
            return similar_questions[:k]
            
        except Exception as e:
            logger.error(f"유사 질문 검색 실패: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """검색 시스템 통계 정보"""
        try:
            collection_info = self.vector_store.get_collection_info()
            
            return {
                'total_documents': collection_info.get('document_count', 0),
                'collection_name': collection_info.get('collection_name', ''),
                'search_settings': {
                    'top_k': self.top_k,
                    'search_type': self.search_type,
                    'score_threshold': self.score_threshold,
                    'max_context_length': self.max_context_length,
                    'hybrid_alpha': self.hybrid_alpha,
                    'use_bm25': self.use_bm25,
                    'use_reranking': self.use_reranking
                },
                'supported_domains': list(self.domain_keywords.keys()),
                'bm25_index_size': len(self.documents) if self.documents else 0
            }
            
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {}
    
    def validate_setup(self) -> Dict[str, bool]:
        """시스템 설정 검증"""
        validation_results = {}
        
        try:
            # 벡터 스토어 연결 확인
            collection_info = self.vector_store.get_collection_info()
            validation_results['vector_store'] = collection_info.get('document_count', 0) > 0
            
            # 임베딩 모델 확인
            try:
                test_embedding = self.vector_store.embeddings.embed_query("테스트")
                validation_results['embedding_model'] = len(test_embedding) > 0
            except:
                validation_results['embedding_model'] = False
            
            # 설정 파일 확인
            validation_results['config_file'] = bool(self.config)
            
            # 도메인 키워드 확인
            validation_results['domain_keywords'] = len(self.domain_keywords) > 0
            
            # BM25 인덱스 확인
            validation_results['bm25_index'] = (
                self.use_bm25 and 
                self.bm25_index is not None and 
                len(self.documents) > 0
            )
            
            # 전체 시스템 상태
            validation_results['hybrid_search_ready'] = (
                validation_results.get('vector_store', False) and
                (not self.use_bm25 or validation_results.get('bm25_index', False))
            )
            
        except Exception as e:
            logger.error(f"설정 검증 실패: {e}")
            validation_results['system_error'] = True
        
        return validation_results
    
    def rebuild_bm25_index(self):
        """BM25 인덱스 재구축"""
        logger.info("BM25 인덱스 재구축 시작...")
        self._initialize_bm25()
        logger.info("BM25 인덱스 재구축 완료")
    
    def get_search_explanation(self, query: str) -> Dict[str, Any]:
        """검색 과정 설명"""
        domains = self.identify_query_domain(query)
        enhanced_query = self.enhance_query(query)
        
        return {
            'original_query': query,
            'preprocessed_query': self.preprocess_query(query),
            'identified_domains': domains,
            'enhanced_query': enhanced_query if enhanced_query != query else None,
            'search_method': 'hybrid' if self.use_bm25 else 'vector_only',
            'hybrid_ratio': f"{self.hybrid_alpha:.1f}:{1-self.hybrid_alpha:.1f}" if self.use_bm25 else None,
            'bm25_enabled': self.use_bm25,
            'total_indexed_docs': len(self.documents)
        }