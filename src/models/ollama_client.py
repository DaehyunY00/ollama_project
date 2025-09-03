"""
Ollama LLM 클라이언트
국방 M&S 특화 프롬프트와 응답 생성을 담당하는 모듈
실제 파일명 기반 출처 표시 기능 포함
"""

import logging
from typing import Dict, Any, Optional, List, Iterator
import yaml
import time
import ollama
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseOllamaClient:
    """국방 M&S 특화 Ollama 클라이언트"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        Ollama 클라이언트 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # Ollama 설정
        self.host = self.config['ollama']['host']
        self.model = self.config['ollama']['model']
        self.temperature = self.config['ollama']['temperature']
        self.max_tokens = self.config['ollama']['max_tokens']
        self.timeout = self.config['ollama']['timeout']
        
        # 프롬프트 템플릿
        self.system_prompt = self.config['rag']['generation']['system_prompt_template']
        self.user_prompt = self.config['rag']['generation']['user_prompt_template']
        
        # Ollama 클라이언트 초기화
        self.client = ollama.Client(host=self.host)
        
        # 모델 가용성 확인
        self._verify_model()
        
        logger.info(f"Ollama 클라이언트 초기화 완료 - 모델: {self.model}")
    
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
            'ollama': {
                'host': 'http://localhost:11434',
                'model': 'llama3:8b',
                'temperature': 0.7,
                'max_tokens': 2048,
                'timeout': 60
            },
            'rag': {
                'generation': {
                    'system_prompt_template': "당신은 국방 M&S 전문가입니다.",
                    'user_prompt_template': "질문: {question}\n컨텍스트: {context}\n\n답변:"
                }
            }
        }
    
    def _verify_model(self):
        """모델 가용성 확인"""
        try:
            # 간단한 테스트 메시지로 모델 확인
            response = self.client.generate(
                model=self.model,
                prompt="테스트",
                options={'num_predict': 10}
            )
            logger.info("모델 연결 확인 완료")
        except Exception as e:
            logger.error(f"모델 연결 실패: {e}")
            logger.error(f"모델 다운로드 명령어: ollama pull {self.model}")
            raise
    
    def _extract_sources_from_context(self, context: str) -> List[Dict[str, str]]:
        """컨텍스트에서 소스 정보 추출"""
        sources = []
        
        try:
            # 문서 헤더 패턴: [문서 N: 파일명 - 도메인 영역 - 구간 M]
            pattern = r'\[문서\s+(\d+):\s+([^-]+)\s+-\s+([^-]+)\s+영역\s+-\s+구간\s+(\d+)\]'
            matches = re.findall(pattern, context)
            
            for match in matches:
                doc_num, filename, domain, chunk_id = match
                sources.append({
                    'doc_number': doc_num.strip(),
                    'filename': filename.strip(),
                    'domain': domain.strip(),
                    'chunk_id': chunk_id.strip()
                })
            
            # 중복 파일명 제거
            unique_sources = []
            seen_files = set()
            for source in sources:
                if source['filename'] not in seen_files and source['filename'] != "알 수 없는 파일":
                    unique_sources.append(source)
                    seen_files.add(source['filename'])
            
            return unique_sources
            
        except Exception as e:
            logger.error(f"소스 정보 추출 실패: {e}")
            return []
    
    def _format_source_list(self, sources: List[Dict[str, str]]) -> str:
        """소스 목록 형식화"""
        if not sources:
            return "참고할 수 있는 문서가 없습니다."
        
        formatted = []
        for i, source in enumerate(sources, 1):
            filename = source['filename']
            domain = source['domain']
            formatted.append(f"{i}. {filename} ({domain} 영역)")
        
        return "\n".join(formatted)
    
    def _enhance_source_citations(self, answer: str, sources: List[Dict[str, str]]) -> str:
        """답변의 출처 표시 강화"""
        try:
            # LLM이 이미 출처를 잘 표시했는지 확인
            has_proper_sources = False
            
            if "**참고문서:**" in answer or "**출처:**" in answer or "**참고자료:**" in answer:
                # 파일명이 올바르게 인식되었는지 검증
                for source in sources:
                    filename = source['filename']
                    if filename in answer and filename != "알 수 없는 파일":
                        has_proper_sources = True
                        break
            
            # 출처가 없거나 부정확한 경우 수동 추가
            if not has_proper_sources and sources:
                source_list = []
                for source in sources:
                    filename = source['filename']
                    domain = source['domain']
                    if filename and filename != "알 수 없는 파일":
                        source_list.append(f"- {filename} ({domain} 영역)")
                
                if source_list:
                    citation = f"\n\n**참고문서:**\n" + "\n".join(source_list)
                    return answer + citation
            
            return answer
            
        except Exception as e:
            logger.error(f"출처 표시 강화 실패: {e}")
            return answer
    
    def generate_response(
        self, 
        query: str, 
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """응답 생성"""
        try:
            # 파라미터 설정
            temp = temperature or self.temperature
            max_tok = max_tokens or self.max_tokens
            
            # 프롬프트 구성
            if context:
                user_message = self.user_prompt.format(
                    question=query,
                    context=context
                )
            else:
                user_message = f"질문: {query}\n\n답변:"
            
            # 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ]
            
            logger.info(f"응답 생성 시작 - 온도: {temp}, 최대 토큰: {max_tok}")
            start_time = time.time()
            
            if stream:
                return self._generate_stream(messages, temp, max_tok)
            else:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        'temperature': temp,
                        'num_predict': max_tok,
                        'top_p': 0.9,
                        'repeat_penalty': 1.1
                    }
                )
                
                generation_time = time.time() - start_time
                logger.info(f"응답 생성 완료 - 소요 시간: {generation_time:.2f}초")
                
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _generate_stream(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> Iterator[str]:
        """스트리밍 응답 생성"""
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 실패: {e}")
            yield f"스트리밍 응답 생성 중 오류: {str(e)}"
    
    def generate_defense_response(
        self, 
        query: str, 
        context: str = "",
        domain: str = "일반"
    ) -> Dict[str, Any]:
        """국방 M&S 특화 응답 생성 (개선된 출처 표시)"""
        try:
            start_time = time.time()
            
            # 컨텍스트에서 소스 정보 추출
            sources = self._extract_sources_from_context(context)
            logger.info(f"추출된 소스 정보: {len(sources)}개 파일")
            
            # 도메인별 시스템 프롬프트 조정
            enhanced_system_prompt = self._get_domain_specific_prompt(domain)
            
            # 출처 인식을 강화한 사용자 프롬프트
            if context:
                enhanced_user_prompt = f"""다음 참고 문서들을 바탕으로 질문에 답변해주세요.

**참고 문서들:**
{context}

**질문:** {query}
**영역:** {domain}

**답변 작성 지침:**
1. 위 참고 문서의 내용을 바탕으로 정확하고 전문적인 답변을 작성하세요.
2. 답변 마지막에 반드시 아래 형식으로 참고한 실제 파일명들을 명시하세요:

**참고문서:**
{self._format_source_list(sources)}

**중요 사항:**
- "Document1", "문서1" 같은 일반적 표현은 절대 사용하지 마세요
- 반드시 위에 제시된 실제 파일명을 사용하세요
- 참고하지 않은 문서는 출처에 포함하지 마세요
- 답변 내용과 출처가 일치하도록 작성하세요"""
            else:
                enhanced_user_prompt = f"""**질문 분석**
영역: {domain}
질문: {query}

**요청사항**
위 질문에 대해 국방 M&S 전문가로서 다음 구조로 답변해주세요:

1. **핵심 답변** (2-3문장 요약)
2. **기술적 세부설명**
   - 주요 개념 및 원리
   - 적용 방법론
   - 기술적 고려사항
3. **실무 적용방안**
   - 구체적 구현 방법
   - 도구 및 표준 활용
   - 모범사례 및 사례연구
4. **주의사항 및 제한사항**
   - 기술적 한계
   - 운용상 고려사항
   - 보안 및 정책적 이슈
5. **참고자료** (관련 표준, 문서, 추가 학습 자료)

**답변 품질 기준:**
- 기술적 정확성 우선
- 실무 적용 가능성 중시
- 국방 특수성 반영
- 구체적이고 actionable한 정보 제공"""
            
            # 메시지 구성
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": enhanced_user_prompt}
            ]
            
            logger.info(f"응답 생성 시작 (도메인: {domain}, 소스: {len(sources)}개)")
            
            # LLM 호출
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            generation_time = time.time() - start_time
            answer = response['message']['content']
            
            # 출처 정보 후처리 및 강화
            formatted_answer = self._enhance_source_citations(answer, sources)
            
            result = {
                'answer': formatted_answer,
                'model': self.model,
                'domain': domain,
                'generation_time': generation_time,
                'sources_used': [s['filename'] for s in sources],
                'context_length': len(context),
                'source_count': len(sources)
            }
            
            logger.info(f"응답 생성 완료 - 소요시간: {generation_time:.2f}초, 소스: {len(sources)}개")
            return result
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            return {
                'answer': f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                'model': self.model,
                'domain': domain,
                'error': str(e),
                'generation_time': 0,
                'sources_used': [],
                'source_count': 0
            }
    
    def _get_domain_specific_prompt(self, domain: str) -> str:
        """도메인별 특화 시스템 프롬프트"""
        base_prompt = """당신은 국방 모델링 및 시뮬레이션(M&S) 분야의 최고 전문가입니다.

**전문 역량:**
- 20년 이상의 국방 M&S 실무 경험
- 국내외 M&S 표준 및 기술 동향에 대한 깊은 이해
- 전투 효과 분석, 시뮬레이션 설계, VV&A 전문성
- HLA, DIS, SISO 표준 전문가

**답변 원칙:**
1. 정확하고 기술적으로 정밀한 정보 제공
2. 실무에 직접 적용 가능한 구체적 가이드
3. 관련 표준, 절차, 모범사례 포함
4. 불확실한 정보는 명시적으로 구분
5. 한국의 국방 환경과 요구사항 고려

**출처 표시 규칙:**
- 답변 끝에 반드시 "**참고문서:**" 섹션 포함
- 실제 파일명을 정확히 기재 (예: "HLA표준문서.pdf", "전투시뮬레이션가이드.pdf")
- "Document1", "문서1" 같은 일반적 표현 절대 금지
- 참고하지 않은 문서는 출처에 포함하지 않음"""

        domain_prompts = {
            '지상전': f"""{base_prompt}

**특화 전문 분야: 지상전 M&S**
- 육군 무기체계 모델링 (전차, 장갑차, 포병)
- 지상전 전술 시뮬레이션
- 기동 및 화력 효과 분석
- 지형 분석 및 기상 영향
- 보급 및 유지정비 모델링

육군 작전과 지상 무기체계에 대한 깊은 전문지식을 바탕으로 정확하고 실무적인 답변을 제공합니다.""",
            
            '해상전': f"""{base_prompt}

**특화 전문 분야: 해상전 M&S**
- 해군 함정 및 잠수함 모델링
- 해상전 시나리오 시뮬레이션
- 대함/대잠 무기체계 효과 분석
- 해상 환경 요소 (조류, 수온, 염분도)
- 상륙작전 및 해상봉쇄 시뮬레이션

해군 작전과 해상 무기체계의 특성을 정확히 반영한 전문적인 답변을 제공합니다.""",
            
            '공중전': f"""{base_prompt}

**특화 전문 분야: 공중전 M&S**
- 항공기 성능 모델링 (전투기, 수송기, 헬기)
- 공중전 교전 시뮬레이션
- 공대공/공대지 무기체계 분석
- 항공작전 계획 및 효과 분석
- 공중급유, 공중조기경보 모델링

공군 작전과 항공 무기체계의 복잡한 동역학을 정확히 모델링한 전문 지식을 제공합니다.""",
            
            '사이버전': f"""{base_prompt}

**특화 전문 분야: 사이버전 M&S**
- 사이버 공격/방어 시나리오 모델링
- 정보시스템 취약성 분석
- 네트워크 보안 시뮬레이션
- 전자전 및 정보전 모델링
- 사이버 위협 효과 분석

사이버 도메인의 특수성과 정보보안 요구사항을 고려한 전문적인 답변을 제공합니다.""",
            
            '합동작전': f"""{base_prompt}

**특화 전문 분야: 합동작전 M&S**
- 다군 연합 작전 시뮬레이션
- 합동화력 조정 및 효과 분석
- C4I 체계 모델링
- 상호운용성 및 통합 시뮬레이션
- 다영역작전(MDO) 모델링

육해공군의 통합 작전과 연합전력의 시너지 효과를 정확히 분석한 전문 답변을 제공합니다."""
        }
        
        return domain_prompts.get(domain, base_prompt)
    
    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """기존 호환성을 위한 답변 생성 메서드"""
        return self.generate_defense_response(query, context, "일반")
    
    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """문서 요약"""
        try:
            prompt = f"""다음 국방 M&S 관련 문서를 {max_length}자 이내로 요약해주세요.
핵심 내용과 주요 개념을 포함하여 간결하게 요약해주세요.

문서 내용:
{text}

요약:"""
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': max_length,
                    'top_p': 0.8
                }
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"문서 요약 실패: {e}")
            return "문서 요약 중 오류가 발생했습니다."
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출"""
        try:
            prompt = f"""다음 국방 M&S 관련 텍스트에서 핵심 키워드 {max_keywords}개를 추출해주세요.
기술 용어, 무기체계명, 작전 개념 등을 우선적으로 선별해주세요.
키워드는 쉼표로 구분하여 나열해주세요.

텍스트:
{text}

키워드:"""
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 200,
                    'top_p': 0.8
                }
            )
            
            keywords_text = response['response'].strip()
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    def translate_query(self, query: str, target_lang: str = "영어") -> str:
        """쿼리 번역"""
        try:
            prompt = f"""다음 국방 M&S 관련 질문을 {target_lang}로 번역해주세요.
전문 용어는 정확하게 번역하고, 필요시 원문을 병기해주세요.

원문: {query}
번역:"""
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 200
                }
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"쿼리 번역 실패: {e}")
            return query
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        try:
            # Ollama 모델 목록 조회
            models = self.client.list()
            
            current_model_info = None
            for model in models['models']:
                if model['name'] == self.model:
                    current_model_info = model
                    break
            
            return {
                'model_name': self.model,
                'host': self.host,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'model_info': current_model_info,
                'available_models': [model['name'] for model in models['models']]
            }
            
        except Exception as e:
            logger.error(f"모델 정보 조회 실패: {e}")
            return {
                'model_name': self.model,
                'host': self.host,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        try:
            start_time = time.time()
            
            # 간단한 테스트 생성
            response = self.client.generate(
                model=self.model,
                prompt="안녕하세요",
                options={'num_predict': 10}
            )
            
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'model': self.model,
                'host': self.host,
                'response_time': response_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.model,
                'host': self.host,
                'error': str(e),
                'timestamp': time.time()
            }