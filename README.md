# MAR_project

MuJoCo 기반 식사 보조 로봇 시뮬레이션 프로젝트입니다. 숟가락으로 음식을 뜨는 동작을 만들기 위해 tray 영역별 scoop primitive를 생성하고, IK로 검증한 뒤 LUT로 저장하고 replay하는 파이프라인을 포함합니다.

## 주요 구성

- `meal_assist/`: 현재 사용 중인 모듈형 Python 패키지
- `eating_scoop_system_v11.py`: v11 단일 파일 버전
- `robot_model_v5_scene.xml`: MuJoCo scene XML
- `model/`: XML에서 참조하는 STL mesh 파일
- `scoop_lut_output_v11/`: 생성된 scoop primitive LUT 및 connector cache
- `requirements.txt`: pip 설치용 의존성 목록
- `environment.yml`: conda 환경 생성용 파일

## 설치

Python 3.11 사용을 권장합니다.

### pip 사용

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### conda 사용

```bash
conda env create -f environment.yml
conda activate meal-assist-robot
```

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
python -m meal_assist --mode test_run
```

LUT를 새로 생성하려면:

```bash
python -m meal_assist --mode build_lut
```

특정 region만 생성하려면:

```bash
python -m meal_assist --mode build_lut --regions 1 2 3
```

저장된 LUT에서 primitive를 선택하고 viewer로 replay하려면:

```bash
python -m meal_assist --mode replay --region 1 --food_xy 0.12 0.18 --viewer
```

단일 파일 버전도 직접 실행할 수 있습니다.

```bash
python eating_scoop_system_v11.py --mode test_run
```

## GitHub 업로드 메모

STL mesh 파일은 크기가 크기 때문에 Git LFS 사용을 권장합니다.

```bash
git lfs install
git add .
git commit -m "Add meal assist robot workspace"
git push -u origin main
```

## 참고

- robot_model_v5_scene.xml 파일의 충돌 모델 조금 더 정교하게 업데이트 했습니다. 26.06.23
- MuJoCo XML은 `model/` 폴더의 STL mesh를 참조하므로, `robot_model_v5_scene.xml`과 `model/`은 같은 루트 구조를 유지해야 합니다.
- 기본 출력 폴더는 `scoop_lut_output_v11/`입니다.
- `scoop_lut_output_v11/`은 현재 repository에 포함하도록 설정되어 있습니다.
