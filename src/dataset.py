from __future__ import annotations

from typing import ClassVar

import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor


class SpeechOceanDataset:
    DATA_SPLITS: ClassVar[list[str]] = ['train', 'test']

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "whisper-tiny": "openai/whisper-tiny",
        "whisper-base": "openai/whisper-base",
        "whisper-small": "openai/whisper-small",
        "whisper-medium": "openai/whisper-medium",
        "whisper-large-v3": "openai/whisper-large-v3",
    }

    def __init__(
        self,
        split: str = 'train',
        backbone: str = 'whisper-large-v3'
    ):
        if split not in self.DATA_SPLITS:
            raise ValueError(
                f"Unknown Split: {split}."  # 에러 메시지 수정 (Backbone -> Split)
                f"Choose from: {self.DATA_SPLITS}" # 오타 수정 (DATA_SPLIT -> DATA_SPLITS)
            )
        if backbone not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown Backbone: {backbone}."
                f"Choose from: {list(self.SUPPORTED_MODELS)}"
            )

        model_id = self.SUPPORTED_MODELS[backbone]

        # 1. 전처리 도구 로드
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.sr = self.feature_extractor.sampling_rate

        # 2. 데이터셋 로드
        self.dataset = load_dataset("mispeech/speechocean762", split=split)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=self.sr))

        # 3. 데이터셋에 전처리 함수 적용
        self.dataset = self.dataset.map(
            self.preprocess_function, 
            remove_columns=self.dataset.column_names,
        ) 
        self.dataset.set_format(type="torch", columns=["input_features", "labels"])

    def preprocess_function(self, example):
        # 개별 샘플(example)에서 오디오 배열 추출
        audio = example["audio"]
        
        # Whisper 전용 로그-멜 스펙트로그램으로 변환
        processed = self.feature_extractor(
            audio["array"], 
            sampling_rate=self.sr, 
            return_tensors="pt"
        )
        
        # 스펙트로그램 피처와 타겟 점수를 딕셔너리로 반환
        return {
            "input_features": processed.input_features[0],
            # Speechocean762의 실제 레이블 구조(accuracy, prosodic 등)에 맞게 매핑
            "labels": [float(example["accuracy"]), float(example["prosodic"])] 
        }