# 🔊 Voice2Face-MLops

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/docker--compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/mlflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"> <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white"> <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=PostgreSQL&logoColor=white"> <img src="https://img.shields.io/badge/MinIO-F5370D?style=for-the-badge&logo=MinIO&logoColor=white"> <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=Grafana&logoColor=white"> <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=white">

## Project Overview
이 프로젝트에서는 MLflow와 MinIO를 활용하여 모델의 학습, 버전 관리 및 로그 기록을 체계적으로 수행했습니다. Docker와 Docker Compose를 이용해 SF2F와 SwimSwap 모델의 서빙 환경을 구축하였으며, PostgreSQL을 통해 메타데이터를 안정적으로 저장했습니다. 또한, Prometheus와 Grafana를 통합하여 시스템 및 모델의 성능을 실시간으로 모니터링하고, Alertmanager를 통해 이상 징후 발생 시 즉각적인 알림을 받을 수 있도록 설정했습니다. 이러한 MLOps 파이프라인을 통해 모델의 개발부터 배포, 모니터링까지 전 과정이 자동화되고 효율적으로 관리될 수 있도록 설계하였습니다.

## Project Structure
```
voice2face-mlops
 ┣ docker
 ┃ ┣ mlflow
 ┃ ┃ ┣ docker-compose_mlflow.yaml
 ┃ ┃ ┗ DockerFile.mlflow
 ┃ ┣ monitoring
 ┃ ┃ ┣ alertmanager
 ┃ ┃ ┃ ┗ config
 ┃ ┃ ┃ ┃ ┗ alertmanager.yml
 ┃ ┃ ┣ prometheus
 ┃ ┃ ┃ ┗ config
 ┃ ┃ ┃ ┃ ┣ prometheus.yml
 ┃ ┃ ┃ ┃ ┗ rule.yml
 ┃ ┃ ┣ docker-compose_monitoring.yaml
 ┃ ┃ ┗ docker-compose_node_exporter.yaml
 ┃ ┗ pipeline
 ┃ ┃ ┣ docker-compose_serving.yaml
 ┃ ┃ ┣ Dockerfile.sf2f
 ┃ ┃ ┗ Dockerfile.swimswap
 ┣ mlflow
 ┃ ┣ registry
 ┃ ┃ ┗ Swimswap
 ┃ ┃ ┃ ┗ model_registry.py
 ┃ ┗ train
 ┃ ┃ ┗ sf2f
 ┃ ┃ ┃ ┣ inference_fuser.py
 ┃ ┃ ┃ ┣ model_registry.py
 ┃ ┃ ┃ ┣ train.py
 ┃ ┃ ┃ ┗ train_registry.py
 ┣ serving
 ┃ ┣ sf2f
 ┃ ┃ ┣ app.py
 ┃ ┃ ┣ config.py
 ┃ ┃ ┣ Dockerfile.sf2f
 ┃ ┃ ┣ inference.py
 ┃ ┃ ┗ requirement.txt
 ┃ ┣ SwimSwap
 ┃ ┃ ┣ app.py
 ┃ ┃ ┣ config.py
 ┃ ┃ ┣ Dockerfile.swimswap
 ┃ ┃ ┣ inference.py
 ┃ ┃ ┗ requirement.txt
 ┃ ┣ docker-compose.yaml
 ┃ ┗ requirements.txt
```

## Usage
### `docker`
- **`mlflow`**: MLflow 및 MinIO 관련된 설정 파일들이 모여 있는 디렉토리입니다.
  - `docker-compose_mlflow.yaml`: MLflow 및 MinIO 환경을 Docker로 실행하기 위한 설정 파일입니다.
  - `DockerFile.mlflow`: MLflow 서버의 Docker 이미지를 생성하기 위한 스크립트입니다.

- **`monitoring`**: 모니터링 시스템 구축을 위한 파일들이 모여 있습니다.
  - `alertmanager/config/alertmanager.yml`: Alertmanager 설정 파일로, 알람 조건을 정의합니다.
  - `prometheus/config/prometheus.yml`: Prometheus 모니터링 설정 파일로, 모니터링 대상과 관련된 설정을 포함합니다.
  - `prometheus/config/rule.yml`: Prometheus에서 사용할 규칙 파일로, 경고 조건을 정의합니다.
  - `docker-compose_monitoring.yaml`: Prometheus, Grafana 및 Alertmanager 서비스를 Docker로 실행하는 설정 파일입니다.
  - `docker-compose_node_exporter.yaml`: 시스템 메트릭을 수집하는 Node Exporter 설정 파일입니다.

- **`pipeline`**: 모델 서빙과 관련된 파이프라인 설정을 담고 있습니다.
  - `docker-compose_serving.yaml`: SF2F 및 SwimSwap 모델 서빙을 위한 Docker 설정 파일입니다.
  - `Dockerfile.sf2f`: SF2F 모델 서빙을 위한 Docker 이미지 생성 스크립트입니다.
  - `Dockerfile.swimswap`: SwimSwap 모델 서빙을 위한 Docker 이미지 생성 스크립트입니다.

### `mlflow`
- **`registry`**: MLflow에 모델을 등록하기 위한 파일들이 모여 있습니다.
  - `Swimswap/model_registry.py`: SwimSwap 모델을 MLflow에 등록하기 위한 스크립트입니다.

- **`train`**: 모델 학습과 관련된 스크립트들이 모여 있습니다.
  - `sf2f/inference_fuser.py`: SF2F 모델의 추론을 위한 데이터 병합 기능을 구현한 스크립트입니다.
  - `sf2f/model_registry.py`: SF2F 모델을 MLflow에 등록하는 스크립트입니다.
  - `sf2f/train.py`: SF2F 모델을 학습시키는 메인 스크립트입니다.
  - `sf2f/train_registry.py`: 모델을 학습시키고 MLflow에 관련 정보를 등록하는 스크립트입니다.

### `serving`
- **`sf2f`**: SF2F 모델 서빙과 관련된 파일들이 모여 있습니다.
  - `app.py`: SF2F 모델을 서비스하기 위한 애플리케이션 엔트리 포인트입니다.
  - `config.py`: SF2F 서비스 환경 설정을 위한 파일입니다.
  - `Dockerfile.sf2f`: SF2F 모델 서빙을 위한 Docker 이미지 생성 스크립트입니다.
  - `inference.py`: SF2F 모델의 추론 기능을 구현한 스크립트입니다.
  - `requirement.txt`: SF2F 서비스에 필요한 Python 패키지 목록입니다.

- **`SwimSwap`**: SwimSwap 모델 서빙과 관련된 파일들입니다.
  - `app.py`: SwimSwap 모델을 서비스하기 위한 애플리케이션 엔트리 포인트입니다.
  - `config.py`: SwimSwap 서비스 환경 설정을 위한 파일입니다.
  - `Dockerfile.swimswap`: SwimSwap 모델 서빙을 위한 Docker 이미지 생성 스크립트입니다.
  - `inference.py`: SwimSwap 모델의 추론 기능을 구현한 스크립트입니다.
  - `requirement.txt`: SwimSwap 서비스에 필요한 Python 패키지 목록입니다.

- **`docker-compose.yaml`**: 전체 모델 서빙을 위한 Docker Compose 설정 파일입니다.
- **`requirements.txt`**: 전체 프로젝트에 필요한 패키지 목록을 정의한 파일입니다.

## Getting Started
### Setup
#### Model Inference
- After training, the model can be served using Docker. Navigate to the `serving/sf2f` folder and run the following command to launch the inference service:

  ```bash
  docker-compose -f docker-compose_serving.yaml up
  ```

#### Monitoring and Logging
- To monitor system metrics, run the Grafana, Prometheus and Alertmanager services. These are located in the `docker/monitoring` folder:

  ```bash
  docker-compose -f docker-compose_monitoring.yaml up
  ```

- You can also track model versions and logs using MLFlow, store the data Using Minio and store metadata using PostgreSQL. These are located in the the `docker/mlflow` folder:

  ```bash
  docker-compose -f docker-compose_mlflow.yaml up
  ```

## Training and Logging the SF2F Model
Navigate to the mlflow/train/sf2f directory and run train.py to train the Voice2Face(SF2F) model at MLFlow:
```
python train.py
```
## Registry SIMSWAP Model
Navigate to mlflow/registry/Swimswap directory and run model_registry.py to registry SimSwap Model at MLflow
```
python model_registry.py
``` 
## License

This project is licensed under the MIT License.