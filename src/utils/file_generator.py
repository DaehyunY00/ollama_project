"""
데이터 파일 생성기
XML, JSON, CSV 등 다양한 형식의 국방 M&S 데이터 파일 생성
"""

import json
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import random
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseFileGenerator:
    """국방 M&S 데이터 파일 생성기"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        파일 생성기 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['file_generation']['output_directory'])
        self.templates_dir = Path(self.config['file_generation']['templates_directory'])
        
        # 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # 국방 M&S 데이터 템플릿
        self.weapon_systems = self._load_weapon_systems()
        self.unit_types = self._load_unit_types()
        self.terrain_types = self._load_terrain_types()
        
        logger.info("파일 생성기 초기화 완료")
    
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
            'file_generation': {
                'output_directory': './data/output',
                'templates_directory': './data/templates',
                'formats': {
                    'xml': {'encoding': 'utf-8', 'pretty_print': True},
                    'json': {'encoding': 'utf-8', 'indent': 2, 'ensure_ascii': False},
                    'csv': {'encoding': 'utf-8', 'delimiter': ',', 'quoting': 'minimal'}
                }
            }
        }
    
    def _load_weapon_systems(self) -> Dict[str, Dict[str, Any]]:
        """무기체계 데이터 로드"""
        return {
            "K2_MBT": {
                "name": "K2 흑표",
                "type": "주력전차",
                "caliber": 120,
                "max_speed": 70,
                "crew": 3,
                "armor": "복합장갑",
                "effective_range": 3000,
                "reload_time": 8.5
            },
            "K9_SPH": {
                "name": "K9 자주포",
                "type": "자주포",
                "caliber": 155,
                "max_speed": 67,
                "crew": 5,
                "range": 30000,
                "reload_time": 15
            },
            "KF-21": {
                "name": "KF-21 보라매",
                "type": "전투기",
                "max_speed": 1800,
                "crew": 1,
                "combat_radius": 740,
                "ceiling": 17000
            },
            "KSS-III": {
                "name": "도산안창호급",
                "type": "잠수함",
                "displacement": 3358,
                "crew": 50,
                "max_speed": 37,
                "diving_depth": 400
            }
        }
    
    def _load_unit_types(self) -> Dict[str, Dict[str, Any]]:
        """부대 유형 데이터 로드"""
        return {
            "infantry": {
                "name": "보병",
                "size": "소대",
                "personnel": 30,
                "mobility": "도보",
                "primary_weapon": "소총"
            },
            "armor": {
                "name": "기갑",
                "size": "중대",
                "personnel": 14,
                "mobility": "궤도",
                "primary_weapon": "전차포"
            },
            "artillery": {
                "name": "포병",
                "size": "포대",
                "personnel": 80,
                "mobility": "자주",
                "primary_weapon": "곡사포"
            },
            "aviation": {
                "name": "항공",
                "size": "편대",
                "personnel": 4,
                "mobility": "항공",
                "primary_weapon": "공대공미사일"
            }
        }
    
    def _load_terrain_types(self) -> List[str]:
        """지형 유형 로드"""
        return [
            "평지", "구릉지", "산악지", "사막", "숲",
            "도시", "강변", "해안", "습지", "고원"
        ]
    
    def generate_simulation_config_xml(
        self, 
        scenario_name: str = "기본_시나리오",
        duration: int = 3600,
        participants: int = 100
    ) -> str:
        """시뮬레이션 설정 XML 파일 생성"""
        try:
            # 루트 엘리먼트 생성
            root = ET.Element("simulation_config")
            root.set("version", "1.0")
            root.set("xmlns", "http://defense.mil.kr/ms/config")
            
            # 메타데이터
            metadata = ET.SubElement(root, "metadata")
            ET.SubElement(metadata, "name").text = scenario_name
            ET.SubElement(metadata, "version").text = "1.0"
            ET.SubElement(metadata, "description").text = f"{scenario_name} 시뮬레이션 설정"
            ET.SubElement(metadata, "created_date").text = datetime.now().isoformat()
            ET.SubElement(metadata, "created_by").text = "Defense RAG System"
            
            # 시뮬레이션 파라미터
            parameters = ET.SubElement(root, "parameters")
            ET.SubElement(parameters, "duration").text = str(duration)
            ET.SubElement(parameters, "time_step").text = "1.0"
            ET.SubElement(parameters, "resolution").text = "high"
            ET.SubElement(parameters, "weather").text = random.choice(["clear", "cloudy", "rain", "snow"])
            ET.SubElement(parameters, "visibility").text = str(random.randint(5000, 15000))
            
            # 지형 설정
            terrain = ET.SubElement(root, "terrain")
            ET.SubElement(terrain, "type").text = random.choice(self._load_terrain_types())
            ET.SubElement(terrain, "size_x").text = str(random.randint(10000, 50000))
            ET.SubElement(terrain, "size_y").text = str(random.randint(10000, 50000))
            ET.SubElement(terrain, "elevation_data").text = "elevation.dat"
            
            # 참가 세력
            forces = ET.SubElement(root, "forces")
            
            # RED 세력
            red_force = ET.SubElement(forces, "force")
            red_force.set("id", "RED")
            red_force.set("name", "적군")
            red_force.set("color", "#FF0000")
            
            red_units = ET.SubElement(red_force, "units")
            for i in range(participants // 2):
                unit = ET.SubElement(red_units, "unit")
                unit.set("id", f"RED_{i+1:03d}")
                
                unit_type = random.choice(list(self.unit_types.keys()))
                unit_data = self.unit_types[unit_type]
                
                ET.SubElement(unit, "type").text = unit_type
                ET.SubElement(unit, "name").text = unit_data["name"]
                ET.SubElement(unit, "personnel").text = str(unit_data["personnel"])
                ET.SubElement(unit, "position_x").text = str(random.randint(1000, 20000))
                ET.SubElement(unit, "position_y").text = str(random.randint(1000, 20000))
                ET.SubElement(unit, "heading").text = str(random.randint(0, 359))
                ET.SubElement(unit, "status").text = "active"
            
            # BLUE 세력
            blue_force = ET.SubElement(forces, "force")
            blue_force.set("id", "BLUE")
            blue_force.set("name", "아군")
            blue_force.set("color", "#0000FF")
            
            blue_units = ET.SubElement(blue_force, "units")
            for i in range(participants // 2):
                unit = ET.SubElement(blue_units, "unit")
                unit.set("id", f"BLUE_{i+1:03d}")
                
                unit_type = random.choice(list(self.unit_types.keys()))
                unit_data = self.unit_types[unit_type]
                
                ET.SubElement(unit, "type").text = unit_type
                ET.SubElement(unit, "name").text = unit_data["name"]
                ET.SubElement(unit, "personnel").text = str(unit_data["personnel"])
                ET.SubElement(unit, "position_x").text = str(random.randint(30000, 49000))
                ET.SubElement(unit, "position_y").text = str(random.randint(1000, 20000))
                ET.SubElement(unit, "heading").text = str(random.randint(0, 359))
                ET.SubElement(unit, "status").text = "active"
            
            # 이벤트 스케줄
            events = ET.SubElement(root, "events")
            for i in range(5):
                event = ET.SubElement(events, "event")
                event.set("id", f"EVENT_{i+1}")
                ET.SubElement(event, "time").text = str(random.randint(300, duration-300))
                ET.SubElement(event, "type").text = random.choice(["contact", "fire", "move", "resupply"])
                ET.SubElement(event, "description").text = f"시나리오 이벤트 {i+1}"
            
            # XML 문자열 생성
            xml_str = ET.tostring(root, encoding='unicode')
            
            # 보기 좋게 포맷팅
            if self.config['file_generation']['formats']['xml']['pretty_print']:
                dom = minidom.parseString(xml_str)
                xml_str = dom.toprettyxml(indent="  ", encoding=None)
                # 첫 번째 라인의 빈 줄 제거
                xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
            
            # 파일 저장
            output_path = self.output_dir / f"{scenario_name}_config.xml"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            logger.info(f"시뮬레이션 설정 XML 생성 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"XML 파일 생성 실패: {e}")
            raise
    
    def generate_model_parameters_json(
        self,
        model_name: str = "기본_모델",
        weapon_system: Optional[str] = None
    ) -> str:
        """모델 파라미터 JSON 파일 생성"""
        try:
            # 무기체계 선택
            if weapon_system and weapon_system in self.weapon_systems:
                weapon_data = self.weapon_systems[weapon_system]
            else:
                weapon_system = random.choice(list(self.weapon_systems.keys()))
                weapon_data = self.weapon_systems[weapon_system]
            
            # JSON 데이터 구성
            model_data = {
                "model_info": {
                    "name": model_name,
                    "version": "1.0",
                    "type": "combat_model",
                    "created_date": datetime.now().isoformat(),
                    "description": f"{model_name} 전투 모델 파라미터"
                },
                "weapon_systems": {
                    weapon_system: weapon_data
                },
                "movement_parameters": {
                    "max_speed": weapon_data.get("max_speed", 50),
                    "acceleration": round(random.uniform(1.5, 3.0), 2),
                    "deceleration": round(random.uniform(2.0, 4.0), 2),
                    "turn_rate": random.randint(20, 45),
                    "fuel_consumption": round(random.uniform(0.5, 2.0), 2),
                    "terrain_factor": {
                        "road": 1.0,
                        "offroad": 0.7,
                        "forest": 0.4,
                        "urban": 0.6,
                        "mountain": 0.3
                    }
                },
                "combat_parameters": {
                    "effective_range": weapon_data.get("effective_range", 2000),
                    "reload_time": weapon_data.get("reload_time", 10.0),
                    "accuracy": round(random.uniform(0.7, 0.95), 3),
                    "penetration": random.randint(200, 800),
                    "damage": random.randint(100, 500),
                    "ammunition_capacity": random.randint(20, 60)
                },
                "survivability": {
                    "armor_thickness": random.randint(50, 200),
                    "armor_type": weapon_data.get("armor", "복합장갑"),
                    "crew_protection": round(random.uniform(0.8, 0.98), 3),
                    "stealth_factor": round(random.uniform(0.1, 0.9), 2),
                    "electronic_warfare": {
                        "ecm_capability": random.choice([True, False]),
                        "eccm_capability": random.choice([True, False]),
                        "radar_signature": round(random.uniform(0.1, 10.0), 2)
                    }
                },
                "logistics": {
                    "fuel_capacity": random.randint(500, 2000),
                    "maintenance_interval": random.randint(100, 500),
                    "repair_time": random.randint(30, 180),
                    "supply_requirements": {
                        "fuel": round(random.uniform(1.0, 5.0), 2),
                        "ammunition": random.randint(1, 10),
                        "spare_parts": round(random.uniform(0.1, 1.0), 2)
                    }
                },
                "environmental_factors": {
                    "weather_sensitivity": {
                        "rain": round(random.uniform(0.8, 1.0), 2),
                        "snow": round(random.uniform(0.6, 0.9), 2),
                        "fog": round(random.uniform(0.7, 0.95), 2),
                        "wind": round(random.uniform(0.9, 1.0), 2)
                    },
                    "day_night_factor": {
                        "day": 1.0,
                        "night": round(random.uniform(0.7, 0.9), 2),
                        "thermal_vision": random.choice([True, False]),
                        "night_vision": random.choice([True, False])
                    }
                }
            }
            
            # 파일 저장
            output_path = self.output_dir / f"{model_name}_parameters.json"
            
            json_config = self.config['file_generation']['formats']['json']
            with open(output_path, 'w', encoding=json_config['encoding']) as f:
                json.dump(
                    model_data, 
                    f, 
                    indent=json_config['indent'],
                    ensure_ascii=json_config['ensure_ascii'],
                    default=str
                )
            
            logger.info(f"모델 파라미터 JSON 생성 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"JSON 파일 생성 실패: {e}")
            raise
    
    def generate_scenario_data_csv(
        self,
        scenario_name: str = "기본_시나리오",
        num_units: int = 50
    ) -> str:
        """시나리오 데이터 CSV 파일 생성"""
        try:
            # CSV 헤더
            headers = [
                'unit_id', 'unit_type', 'force', 'weapon_system',
                'x_coord', 'y_coord', 'z_coord', 'heading',
                'speed', 'fuel_level', 'ammo_count', 'crew_count',
                'status', 'mission', 'last_contact', 'morale'
            ]
            
            # 데이터 생성
            csv_data = []
            for i in range(num_units):
                force = random.choice(['RED', 'BLUE'])
                unit_type = random.choice(list(self.unit_types.keys()))
                unit_data = self.unit_types[unit_type]
                weapon_system = random.choice(list(self.weapon_systems.keys()))
                
                row = {
                    'unit_id': f"{force}_{i+1:03d}",
                    'unit_type': unit_type,
                    'force': force,
                    'weapon_system': weapon_system,
                    'x_coord': random.randint(1000, 49000),
                    'y_coord': random.randint(1000, 49000),
                    'z_coord': random.randint(100, 500),
                    'heading': random.randint(0, 359),
                    'speed': random.randint(0, 70),
                    'fuel_level': random.randint(30, 100),
                    'ammo_count': random.randint(10, 100),
                    'crew_count': unit_data['personnel'],
                    'status': random.choice(['active', 'moving', 'engaged', 'damaged', 'resupply']),
                    'mission': random.choice(['patrol', 'attack', 'defend', 'recon', 'support']),
                    'last_contact': (datetime.now() - timedelta(minutes=random.randint(0, 120))).isoformat(),
                    'morale': round(random.uniform(0.5, 1.0), 2)
                }
                csv_data.append(row)
            
            # 파일 저장
            output_path = self.output_dir / f"{scenario_name}_data.csv"
            
            csv_config = self.config['file_generation']['formats']['csv']
            with open(output_path, 'w', newline='', encoding=csv_config['encoding']) as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=headers,
                    delimiter=csv_config['delimiter'],
                    quoting=getattr(csv, f"QUOTE_{csv_config['quoting'].upper()}")
                )
                
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"시나리오 데이터 CSV 생성 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"CSV 파일 생성 실패: {e}")
            raise
    
    def generate_complete_scenario(
        self,
        scenario_name: str = "통합_시나리오",
        num_units: int = 100,
        duration: int = 7200
    ) -> Dict[str, str]:
        """완전한 시나리오 파일 세트 생성"""
        try:
            logger.info(f"통합 시나리오 생성 시작: {scenario_name}")
            
            # 각 형식별 파일 생성
            files = {
                'xml': self.generate_simulation_config_xml(scenario_name, duration, num_units),
                'json': self.generate_model_parameters_json(f"{scenario_name}_모델"),
                'csv': self.generate_scenario_data_csv(scenario_name, num_units)
            }
            
            # 메타데이터 파일 생성
            metadata = {
                'scenario_name': scenario_name,
                'created_date': datetime.now().isoformat(),
                'files': files,
                'parameters': {
                    'num_units': num_units,
                    'duration': duration,
                    'terrain': random.choice(self._load_terrain_types())
                },
                'description': f"{scenario_name} 통합 시나리오 파일 세트"
            }
            
            metadata_path = self.output_dir / f"{scenario_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            files['metadata'] = str(metadata_path)
            
            logger.info(f"통합 시나리오 생성 완료: {scenario_name}")
            return files
            
        except Exception as e:
            logger.error(f"통합 시나리오 생성 실패: {e}")
            raise
    
    def create_template_files(self) -> Dict[str, str]:
        """템플릿 파일들 생성"""
        try:
            templates = {}
            
            # XML 템플릿
            xml_template = """<?xml version="1.0" encoding="UTF-8"?>
<simulation_config version="1.0" xmlns="http://defense.mil.kr/ms/config">
    <metadata>
        <name>{{SCENARIO_NAME}}</name>
        <version>{{VERSION}}</version>
        <description>{{DESCRIPTION}}</description>
        <created_date>{{CREATED_DATE}}</created_date>
        <created_by>{{CREATED_BY}}</created_by>
    </metadata>
    
    <parameters>
        <duration>{{DURATION}}</duration>
        <time_step>{{TIME_STEP}}</time_step>
        <resolution>{{RESOLUTION}}</resolution>
        <weather>{{WEATHER}}</weather>
        <visibility>{{VISIBILITY}}</visibility>
    </parameters>
    
    <terrain>
        <type>{{TERRAIN_TYPE}}</type>
        <size_x>{{SIZE_X}}</size_x>
        <size_y>{{SIZE_Y}}</size_y>
        <elevation_data>{{ELEVATION_FILE}}</elevation_data>
    </terrain>
    
    <forces>
        <!-- 세력 정보 -->
    </forces>
    
    <events>
        <!-- 이벤트 스케줄 -->
    </events>
</simulation_config>"""
            
            xml_template_path = self.templates_dir / "simulation_config_template.xml"
            with open(xml_template_path, 'w', encoding='utf-8') as f:
                f.write(xml_template)
            templates['xml'] = str(xml_template_path)
            
            # JSON 템플릿
            json_template = {
                "model_info": {
                    "name": "{{MODEL_NAME}}",
                    "version": "{{VERSION}}",
                    "type": "{{MODEL_TYPE}}",
                    "created_date": "{{CREATED_DATE}}",
                    "description": "{{DESCRIPTION}}"
                },
                "weapon_systems": {},
                "movement_parameters": {},
                "combat_parameters": {},
                "survivability": {},
                "logistics": {},
                "environmental_factors": {}
            }
            
            json_template_path = self.templates_dir / "model_parameters_template.json"
            with open(json_template_path, 'w', encoding='utf-8') as f:
                json.dump(json_template, f, indent=2, ensure_ascii=False)
            templates['json'] = str(json_template_path)
            
            # CSV 템플릿 (헤더만)
            csv_headers = [
                'unit_id', 'unit_type', 'force', 'weapon_system',
                'x_coord', 'y_coord', 'z_coord', 'heading',
                'speed', 'fuel_level', 'ammo_count', 'crew_count',
                'status', 'mission', 'last_contact', 'morale'
            ]
            
            csv_template_path = self.templates_dir / "scenario_data_template.csv"
            with open(csv_template_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
                # 샘플 데이터 한 줄 추가
                writer.writerow([
                    'SAMPLE_001', 'armor', 'BLUE', 'K2_MBT',
                    '25000', '25000', '200', '90',
                    '45', '80', '40', '3',
                    'active', 'patrol', '2024-01-01T12:00:00', '0.85'
                ])
            templates['csv'] = str(csv_template_path)
            
            logger.info("템플릿 파일 생성 완료")
            return templates
            
        except Exception as e:
            logger.error(f"템플릿 파일 생성 실패: {e}")
            raise
    
    def validate_generated_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """생성된 파일들 검증"""
        validation_results = {}
        
        for file_path in file_paths:
            try:
                file_path_obj = Path(file_path)
                
                if not file_path_obj.exists():
                    validation_results[str(file_path)] = False
                    continue
                
                # 파일 형식별 검증
                if file_path_obj.suffix.lower() == '.xml':
                    ET.parse(file_path)  # XML 파싱 테스트
                elif file_path_obj.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)  # JSON 파싱 테스트
                elif file_path_obj.suffix.lower() == '.csv':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        csv.reader(f)  # CSV 읽기 테스트
                
                validation_results[str(file_path)] = True
                
            except Exception as e:
                logger.error(f"파일 검증 실패 {file_path}: {e}")
                validation_results[str(file_path)] = False
        
        return validation_results
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """파일 생성 통계"""
        try:
            stats = {
                'output_directory': str(self.output_dir),
                'templates_directory': str(self.templates_dir),
                'total_files': 0,
                'file_types': {},
                'total_size': 0,
                'latest_files': []
            }
            
            if self.output_dir.exists():
                files = list(self.output_dir.iterdir())
                stats['total_files'] = len(files)
                
                for file_path in files:
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                        stats['total_size'] += file_path.stat().st_size
                
                # 최근 파일 5개
                recent_files = sorted(
                    files, 
                    key=lambda x: x.stat().st_mtime, 
                    reverse=True
                )[:5]
                
                stats['latest_files'] = [
                    {
                        'name': f.name,
                        'size': f.stat().st_size,
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in recent_files if f.is_file()
                ]
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {'error': str(e)}