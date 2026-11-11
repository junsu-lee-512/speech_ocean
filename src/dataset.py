from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
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
            self.preprocess, 
            remove_columns=self.dataset.column_names,
        ) 
        self.dataset.set_format(type="torch", columns=["input_features", "labels"])

    def preprocess(self, example):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def pad_audio(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    스펙트로그램을 입력으로 받아, 시간축을 기준으로 제로패딩 수행하는 함수.
    제로패딩된 스펙트로그램은 배치 텐서로 묶여 반환
    """
    input_ndim = spectrogram[0].ndim

    # 스펙트로그램의 차원 확인
    if input_ndim != 2:
        raise TypeError(
            "The spectrogram should be two-dimensional",
            f"ndim:{input_ndim}"
        )

    # 배치에서 가장 긴 오디오 길이
    max_len = max(s.shape[-1] for s in spectrogram)

    padded_list = []
    for s in spectrogram:
        # 현재 스펙트로그램이 채워야 할 패딩 크기 계산
        pad_right = max_len - s.shape[-1]

        # 시간축에 제로패딩
        # F.pad(tensor, (left_pad, right_pad, top_pad, bottom_pad ...))
        padded_s = F.pad(s, (0, pad_right))
        padded_list.append(padded_s)

    # 4. 리스트로 모인 텐서들을 하나의 배치 텐서로 묶기 ([Batch, Channels, Max_Len])
    return torch.stack(padded_list)    


def collate_fn(batch):
    inputs = [item['input_features'] for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    return pad_audio(inputs), torch.stack(labels)


def build_dataloader(
    dataset: SpeechOceanDataset,
    batch_size: int = 8,
    shuffle: bool = False, # train일 때 True, test일 때 False
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        # pin_memory가 True이고, cuda가 있을 때
        pin_memory=pin_memory and torch.cuda.is_available()
    )