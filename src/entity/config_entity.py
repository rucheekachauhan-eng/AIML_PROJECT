from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float = 0.2
    random_state: int = 42

@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_obj_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: Path
    expected_score: float = 0.6

@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    test_data_path: Path
    metric_file_path: Path
