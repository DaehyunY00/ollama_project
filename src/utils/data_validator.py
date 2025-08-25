"""
데이터 검증기
생성된 국방 M&S 데이터 파일들의 형식과 내용을 검증
"""

import json
import csv
import xml.etree.ElementTree as ET
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from datetime import datetime
import jsonschema

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseDataValidator:
    """국방 M&S 데이터 검증기"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        데이터 검증기 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # 검증 규칙 로드
        self.xml_schema = self._load_xml_validation_rules()
        self.json_schema = self._load_json_validation_rules()
        self.csv_schema = self._load_csv_validation_rules()
        
        # 국방 M&S 도메인 특화 규칙
        self.domain_rules = self._load_domain_rules()
        
        logger.info("데이터 검증기 초기화 완료")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류: {e}")
            return {}
    
    def _load_xml_validation_rules(self) -> Dict[str, Any]:
        """XML 검증 규칙 로드"""
        return {
            'required_elements': [
                'metadata', 'parameters', 'terrain', 'forces'
            ],
            'metadata_fields': [
                'name', 'version', 'description', 'created_date'
            ],
            'parameter_fields': [
                'duration', 'time_step', 'resolution'
            ],
            'terrain_fields': [
                'type', 'size_x', 'size_y'
            ],
            'unit_fields': [
                'type', 'name', 'position_x', 'position_y', 'status'
            ]
        }
    
    def _load_json_validation_rules(self) -> Dict[str, Any]:
        """JSON 검증 스키마 로드"""
        return {
            "type": "object",
            "required": ["model_info", "weapon_systems"],
            "properties": {
                "model_info": {
                    "type": "object",
                    "required": ["name", "version", "type"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "version": {"type": "string", "pattern": r"^\d+\.\d+$"},
                        "type": {"type": "string", "enum": ["combat_model", "logistics_model", "command_model"]},
                        "created_date": {"type": "string", "format": "date-time"}
                    }
                },
                "weapon_systems": {
                    "type": "object",
                    "patternProperties": {
                        "^[A-Za-z0-9_-]+$": {
                            "type": "object",
                            "required": ["name", "type"],
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "max_speed": {"type": "number", "minimum": 0},
                                "effective_range": {"type": "number", "minimum": 0},
                                "reload_time": {"type": "number", "minimum": 0}
                            }
                        }
                    }
                },
                "movement_parameters": {
                    "type": "object",
                    "properties": {
                        "max_speed": {"type": "number", "minimum": 0, "maximum": 500},
                        "acceleration": {"type": "number", "minimum": 0},
                        "turn_rate": {"type": "number", "minimum": 0, "maximum": 180}
                    }
                },
                "combat_parameters": {
                    "type": "object",
                    "properties": {
                        "effective_range": {"type": "number", "minimum": 0},
                        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                        "damage": {"type": "number", "minimum": 0}
                    }
                }
            }
        }
    
    def _load_csv_validation_rules(self) -> Dict[str, Any]:
        """CSV 검증 규칙 로드"""
        return {
            'required_columns': [
                'unit_id', 'unit_type', 'force', 'x_coord', 'y_coord', 'status'
            ],
            'column_types': {
                'unit_id': str,
                'unit_type': str,
                'force': str,
                'x_coord': (int, float),
                'y_coord': (int, float),
                'z_coord': (int, float),
                'heading': (int, float),
                'speed': (int, float),
                'fuel_level': (int, float),
                'ammo_count': int,
                'crew_count': int,
                'morale': float
            },
            'valid_values': {
                'unit_type': ['infantry', 'armor', 'artillery', 'aviation', 'naval', 'special'],
                'force': ['RED', 'BLUE', 'GREEN', 'NEUTRAL'],
                'status': ['active', 'moving', 'engaged', 'damaged', 'destroyed', 'resupply']
            },
            'value_ranges': {
                'x_coord': (0, 100000),
                'y_coord': (0, 100000),
                'z_coord': (0, 10000),
                'heading': (0, 360),
                'speed': (0, 500),
                'fuel_level': (0, 100),
                'ammo_count': (0, 1000),
                'crew_count': (1, 100),
                'morale': (0.0, 1.0)
            }
        }
    
    def _load_domain_rules(self) -> Dict[str, Any]:
        """국방 M&S 도메인 특화 규칙"""
        return {
            'weapon_systems': {
                'land': ['tank', 'artillery', 'missile', 'rifle'],
                'naval': ['ship', 'submarine', 'torpedo', 'cannon'],
                'air': ['fighter', 'bomber', 'helicopter', 'missile'],
                'space': ['satellite', 'interceptor'],
                'cyber': ['virus', 'firewall', 'encryption']
            },
            'coordinate_systems': {
                'utm': {'x_range': (-180, 180), 'y_range': (-90, 90)},
                'mgrs': {'pattern': r'^[0-9]{1,2}[A-Z]{3}[0-9]{1,10}$'},
                'local': {'x_range': (0, 100000), 'y_range': (0, 100000)}
            },
            'time_formats': [
                r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$',
                r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z?$',
                r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
            ]
        }
    
    def validate_xml_file(self, file_path: str) -> Dict[str, Any]:
        """XML 파일 검증"""
        try:
            logger.info(f"XML 파일 검증 시작: {file_path}")
            
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'xml'
            }
            
            # 파일 존재 확인
            if not Path(file_path).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("파일을 찾을 수 없습니다")
                return validation_result
            
            # XML 파싱
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
            except ET.ParseError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"XML 파싱 오류: {e}")
                return validation_result
            
            # 루트 엘리먼트 확인
            if root.tag != 'simulation_config':
                validation_result['warnings'].append(f"예상되지 않은 루트 엘리먼트: {root.tag}")
            
            # 필수 엘리먼트 확인
            required_elements = self.xml_schema['required_elements']
            for element in required_elements:
                if root.find(element) is None:
                    validation_result['errors'].append(f"필수 엘리먼트 누락: {element}")
                    validation_result['valid'] = False
            
            # 메타데이터 검증
            metadata = root.find('metadata')
            if metadata is not None:
                for field in self.xml_schema['metadata_fields']:
                    field_elem = metadata.find(field)
                    if field_elem is None:
                        validation_result['warnings'].append(f"메타데이터 필드 누락: {field}")
                    elif not field_elem.text or not field_elem.text.strip():
                        validation_result['warnings'].append(f"메타데이터 필드 값 없음: {field}")
            
            # 파라미터 검증
            parameters = root.find('parameters')
            if parameters is not None:
                # 시간 검증
                duration_elem = parameters.find('duration')
                if duration_elem is not None:
                    try:
                        duration = int(duration_elem.text)
                        if duration <= 0:
                            validation_result['errors'].append("duration은 양수여야 합니다")
                            validation_result['valid'] = False
                        elif duration > 86400:  # 24시간
                            validation_result['warnings'].append("duration이 24시간을 초과합니다")
                    except ValueError:
                        validation_result['errors'].append("duration은 숫자여야 합니다")
                        validation_result['valid'] = False
            
            # 세력 및 부대 검증
            forces = root.find('forces')
            if forces is not None:
                force_count = 0
                unit_count = 0
                
                for force in forces.findall('force'):
                    force_count += 1
                    force_id = force.get('id')
                    if not force_id:
                        validation_result['warnings'].append("세력 ID가 없습니다")
                    
                    units = force.find('units')
                    if units is not None:
                        for unit in units.findall('unit'):
                            unit_count += 1
                            unit_id = unit.get('id')
                            if not unit_id:
                                validation_result['warnings'].append("부대 ID가 없습니다")
                            
                            # 좌표 검증
                            for coord in ['position_x', 'position_y']:
                                coord_elem = unit.find(coord)
                                if coord_elem is not None:
                                    try:
                                        coord_val = float(coord_elem.text)
                                        if coord_val < 0:
                                            validation_result['warnings'].append(f"음수 좌표: {coord} = {coord_val}")
                                    except ValueError:
                                        validation_result['errors'].append(f"좌표는 숫자여야 합니다: {coord}")
                                        validation_result['valid'] = False
                
                validation_result['statistics'] = {
                    'force_count': force_count,
                    'unit_count': unit_count
                }
            
            logger.info(f"XML 검증 완료: {file_path} - {'성공' if validation_result['valid'] else '실패'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"XML 검증 실패: {e}")
            return {
                'valid': False,
                'errors': [f"검증 중 오류 발생: {str(e)}"],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'xml'
            }
    
    def validate_json_file(self, file_path: str) -> Dict[str, Any]:
        """JSON 파일 검증"""
        try:
            logger.info(f"JSON 파일 검증 시작: {file_path}")
            
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'json'
            }
            
            # 파일 존재 확인
            if not Path(file_path).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("파일을 찾을 수 없습니다")
                return validation_result
            
            # JSON 파싱
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"JSON 파싱 오류: {e}")
                return validation_result
            
            # JSON 스키마 검증
            try:
                jsonschema.validate(data, self.json_schema)
            except jsonschema.ValidationError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"스키마 검증 실패: {e.message}")
            except jsonschema.SchemaError as e:
                validation_result['warnings'].append(f"스키마 오류: {e.message}")
            
            # 도메인 특화 검증
            if 'weapon_systems' in data:
                weapon_count = 0
                for weapon_id, weapon_data in data['weapon_systems'].items():
                    weapon_count += 1
                    
                    # 무기체계 유형 검증
                    weapon_type = weapon_data.get('type', '').lower()
                    found_domain = False
                    for domain, types in self.domain_rules['weapon_systems'].items():
                        if any(wtype in weapon_type for wtype in types):
                            found_domain = True
                            break
                    
                    if not found_domain:
                        validation_result['warnings'].append(f"알 수 없는 무기체계 유형: {weapon_type}")
                    
                    # 수치 검증
                    if 'max_speed' in weapon_data:
                        speed = weapon_data['max_speed']
                        if speed < 0 or speed > 3000:  # 마하 3 정도까지
                            validation_result['warnings'].append(f"비정상적인 최대속도: {speed}")
                    
                    if 'effective_range' in weapon_data:
                        range_val = weapon_data['effective_range']
                        if range_val < 0 or range_val > 100000:  # 100km
                            validation_result['warnings'].append(f"비정상적인 사거리: {range_val}")
                
                validation_result['statistics'] = {
                    'weapon_systems_count': weapon_count
                }
            
            # 날짜 형식 검증
            if 'model_info' in data and 'created_date' in data['model_info']:
                date_str = data['model_info']['created_date']
                valid_format = False
                for pattern in self.domain_rules['time_formats']:
                    if re.match(pattern, date_str):
                        valid_format = True
                        break
                
                if not valid_format:
                    validation_result['warnings'].append(f"비표준 날짜 형식: {date_str}")
            
            logger.info(f"JSON 검증 완료: {file_path} - {'성공' if validation_result['valid'] else '실패'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"JSON 검증 실패: {e}")
            return {
                'valid': False,
                'errors': [f"검증 중 오류 발생: {str(e)}"],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'json'
            }
    
    def validate_csv_file(self, file_path: str) -> Dict[str, Any]:
        """CSV 파일 검증"""
        try:
            logger.info(f"CSV 파일 검증 시작: {file_path}")
            
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'csv'
            }
            
            # 파일 존재 확인
            if not Path(file_path).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("파일을 찾을 수 없습니다")
                return validation_result
            
            # CSV 읽기
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # CSV 방언 감지
                    sample = f.read(1024)
                    f.seek(0)
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    
                    reader = csv.DictReader(f, dialect=dialect)
                    headers = reader.fieldnames
                    
                    if not headers:
                        validation_result['valid'] = False
                        validation_result['errors'].append("CSV 헤더를 찾을 수 없습니다")
                        return validation_result
                    
                    rows = list(reader)
                    
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"CSV 읽기 오류: {e}")
                return validation_result
            
            # 헤더 검증
            required_columns = self.csv_schema['required_columns']
            missing_columns = [col for col in required_columns if col not in headers]
            if missing_columns:
                validation_result['errors'].append(f"필수 컬럼 누락: {missing_columns}")
                validation_result['valid'] = False
            
            # 데이터 타입 검증
            column_types = self.csv_schema['column_types']
            valid_values = self.csv_schema['valid_values']
            value_ranges = self.csv_schema['value_ranges']
            
            row_count = 0
            error_count = 0
            
            for i, row in enumerate(rows, start=2):  # 1은 헤더
                row_count += 1
                
                for column, expected_type in column_types.items():
                    if column in row and row[column]:
                        value = row[column]
                        
                        # 타입 검증
                        try:
                            if expected_type == str:
                                pass  # 문자열은 항상 유효
                            elif expected_type == int:
                                int(value)
                            elif expected_type == float:
                                float(value)
                            elif isinstance(expected_type, tuple):
                                # 여러 타입 허용
                                valid_type = False
                                for t in expected_type:
                                    try:
                                        if t == int:
                                            int(value)
                                        elif t == float:
                                            float(value)
                                        valid_type = True
                                        break
                                    except:
                                        continue
                                
                                if not valid_type:
                                    validation_result['errors'].append(
                                        f"행 {i}, 컬럼 '{column}': 타입 오류 - '{value}'"
                                    )
                                    error_count += 1
                                    
                        except ValueError:
                            validation_result['errors'].append(
                                f"행 {i}, 컬럼 '{column}': 타입 오류 - '{value}'"
                            )
                            error_count += 1
                        
                        # 유효값 검증
                        if column in valid_values:
                            if value not in valid_values[column]:
                                validation_result['warnings'].append(
                                    f"행 {i}, 컬럼 '{column}': 비표준 값 - '{value}'"
                                )
                        
                        # 범위 검증
                        if column in value_ranges:
                            try:
                                num_value = float(value)
                                min_val, max_val = value_ranges[column]
                                if not (min_val <= num_value <= max_val):
                                    validation_result['warnings'].append(
                                        f"행 {i}, 컬럼 '{column}': 범위 벗어남 - {num_value} (범위: {min_val}-{max_val})"
                                    )
                            except ValueError:
                                pass  # 이미 타입 검증에서 처리됨
                
                # 논리적 검증
                # 연료량이 0인데 이동 중인 경우
                if (row.get('fuel_level') == '0' and 
                    row.get('status') in ['moving', 'active']):
                    validation_result['warnings'].append(
                        f"행 {i}: 연료 없이 활동 중 (unit_id: {row.get('unit_id', 'N/A')})"
                    )
                
                # 탄약이 0인데 교전 중인 경우
                if (row.get('ammo_count') == '0' and 
                    row.get('status') == 'engaged'):
                    validation_result['warnings'].append(
                        f"행 {i}: 탄약 없이 교전 중 (unit_id: {row.get('unit_id', 'N/A')})"
                    )
            
            # 오류가 너무 많으면 유효하지 않은 것으로 판정
            if error_count > row_count * 0.1:  # 10% 이상 오류
                validation_result['valid'] = False
                validation_result['errors'].append(f"오류율이 너무 높습니다: {error_count}/{row_count}")
            
            validation_result['statistics'] = {
                'total_rows': row_count,
                'total_columns': len(headers),
                'error_count': error_count,
                'columns': headers
            }
            
            logger.info(f"CSV 검증 완료: {file_path} - {'성공' if validation_result['valid'] else '실패'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"CSV 검증 실패: {e}")
            return {
                'valid': False,
                'errors': [f"검증 중 오류 발생: {str(e)}"],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'csv'
            }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """파일 형식에 따른 자동 검증"""
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension == '.xml':
            return self.validate_xml_file(file_path)
        elif file_extension == '.json':
            return self.validate_json_file(file_path)
        elif file_extension == '.csv':
            return self.validate_csv_file(file_path)
        else:
            return {
                'valid': False,
                'errors': [f"지원하지 않는 파일 형식: {file_extension}"],
                'warnings': [],
                'file_path': file_path,
                'file_type': 'unknown'
            }
    
    def validate_directory(self, directory_path: str) -> Dict[str, Any]:
        """디렉토리 내 모든 데이터 파일 검증"""
        try:
            directory_path_obj = Path(directory_path)
            
            if not directory_path_obj.exists():
                return {
                    'valid': False,
                    'error': f"디렉토리를 찾을 수 없습니다: {directory_path}"
                }
            
            # 지원하는 파일들 찾기
            supported_extensions = ['.xml', '.json', '.csv']
            files_to_validate = []
            
            for ext in supported_extensions:
                files_to_validate.extend(directory_path_obj.glob(f"*{ext}"))
            
            validation_results = {
                'directory': str(directory_path),
                'total_files': len(files_to_validate),
                'valid_files': 0,
                'invalid_files': 0,
                'files': []
            }
            
            for file_path in files_to_validate:
                result = self.validate_file(str(file_path))
                validation_results['files'].append(result)
                
                if result['valid']:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['invalid_files'] += 1
            
            validation_results['success_rate'] = (
                validation_results['valid_files'] / validation_results['total_files'] * 100
                if validation_results['total_files'] > 0 else 0
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"디렉토리 검증 실패: {e}")
            return {
                'valid': False,
                'error': f"검증 중 오류 발생: {str(e)}"
            }
    
    def generate_validation_report(
        self, 
        validation_results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Optional[str] = None
    ) -> str:
        """검증 결과 보고서 생성"""
        try:
            if isinstance(validation_results, dict):
                # 단일 파일 결과
                results = [validation_results]
            else:
                # 여러 파일 결과
                results = validation_results
            
            report_lines = []
            report_lines.append("="*60)
            report_lines.append("국방 M&S 데이터 파일 검증 보고서")
            report_lines.append("="*60)
            report_lines.append(f"검증 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"검증 파일 수: {len(results)}")
            report_lines.append("")
            
            # 전체 통계
            total_files = len(results)
            valid_files = sum(1 for r in results if r.get('valid', False))
            invalid_files = total_files - valid_files
            
            report_lines.append("📊 전체 통계")
            report_lines.append("-" * 30)
            report_lines.append(f"총 파일 수: {total_files}")
            report_lines.append(f"유효한 파일: {valid_files}")
            report_lines.append(f"무효한 파일: {invalid_files}")
            report_lines.append(f"성공률: {valid_files/total_files*100:.1f}%" if total_files > 0 else "성공률: N/A")
            report_lines.append("")
            
            # 파일별 상세 결과
            for i, result in enumerate(results, 1):
                report_lines.append(f"📄 파일 {i}: {Path(result['file_path']).name}")
                report_lines.append("-" * 50)
                report_lines.append(f"경로: {result['file_path']}")
                report_lines.append(f"형식: {result.get('file_type', 'unknown').upper()}")
                report_lines.append(f"상태: {'✅ 유효' if result.get('valid', False) else '❌ 무효'}")
                
                if result.get('errors'):
                    report_lines.append("🔴 오류:")
                    for error in result['errors']:
                        report_lines.append(f"  - {error}")
                
                if result.get('warnings'):
                    report_lines.append("🟡 경고:")
                    for warning in result['warnings']:
                        report_lines.append(f"  - {warning}")
                
                if result.get('statistics'):
                    report_lines.append("📈 통계:")
                    for key, value in result['statistics'].items():
                        report_lines.append(f"  - {key}: {value}")
                
                report_lines.append("")
            
            # 권장사항
            report_lines.append("💡 권장사항")
            report_lines.append("-" * 30)
            if invalid_files > 0:
                report_lines.append("- 무효한 파일들을 수정하여 재검증해주세요")
                report_lines.append("- 오류 메시지를 참고하여 데이터 형식을 확인해주세요")
            else:
                report_lines.append("- 모든 파일이 유효합니다!")
            
            if any(r.get('warnings') for r in results):
                report_lines.append("- 경고 메시지를 검토하여 데이터 품질을 향상시켜주세요")
            
            report_lines.append("- 정기적인 데이터 검증을 권장합니다")
            report_lines.append("")
            report_lines.append("="*60)
            
            report_content = "\n".join(report_lines)
            
            # 파일로 저장
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"검증 보고서 저장: {output_path}")
            
            return report_content
            
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")
            return f"보고서 생성 중 오류 발생: {str(e)}"