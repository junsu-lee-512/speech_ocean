from __future__ import annotations

import logging
from typing import ClassVar

import torch
from torch import nn
from transformers import WhisperModel


class WhisperPronunciationModel(nn.Module):

    # 선택 가능한 모델
    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "whisper-tiny": "openai/whisper-tiny",
        "whisper-base": "openai/whisper-base",
        "whisper-small": "openai/whisper-small",
        "whisper-medium": "openai/whisper-medium",
        "whisper-large-v3": "openai/whisper-large-v3",
    }

    def __init__(
        self,
        backbone:str = 'whisper-large-v3',
        hidden_dim: int = 256,
        freeze_backbone: bool = True,       
        dropout: float = 0.1 
    ) -> None:
        super().__init__()

        # SUPPORTED_MODELS에 있는 모델만 허용
        if backbone not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown Backbone: {backbone}."
                f"Choose from: {list(self.SUPPORTED_MODELS)}"
            )

        # 모델 호출
        model_id = self.SUPPORTED_MODELS[backbone]
        _model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
        self.encoder = _model.encoder.to(torch.float32)     # float16는 attn 연산 시 생기는 작은 값을 표현할 수 없음 (안정성이 떨어짐), 다운스트림 레이어와의 호환성 문제 있음
        del _model      # 메모리 확보를 위해 전체 모델 삭제

        # freeze 구현
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
    
        # Whisper 백본의 히든 차원 크기 가져오기 (large-v3 기준 1280)
        backbone_hidden_size = self.encoder.config.d_model

        # regression head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(backbone_hidden_size),
            nn.Linear(backbone_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),   # [Accuracy, Prosodic] 예측
        )

        self.hidden_dim = hidden_dim
        self.backbone_name = backbone


    def forward(
            self,
            input_feature: torch.Tensor
    ) -> torch.Tensor:

        # Whisper Encoder
        en_outputs = self.encoder(input_feature)
        en_hidden_states = en_outputs.last_hidden_state

        # 
        
        
        pass