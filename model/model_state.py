"""
WillTiboReset - 模型状态持久化

ModelState 保存从 reset_history.json 学习到的自适应参数，
使 ResetPredictor 不必每次使用硬编码系数。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelState(BaseModel):
    """
    自适应 Bayesian Survival Model 的持久化状态。

    包含从历史 reset 数据计算出的间隔统计量、
    先验权重以及模型系数。
    """

    average_interval_hours: float = Field(
        ..., description="观测到的平均 reset 间隔（小时）"
    )
    median_interval_hours: Optional[float] = Field(
        default=None, description="中位 reset 间隔（小时）"
    )
    std_interval_hours: Optional[float] = Field(
        default=None, description="reset 间隔标准差（小时）"
    )
    min_interval_hours: Optional[float] = Field(
        default=None, description="最小 reset 间隔（小时）"
    )
    max_interval_hours: Optional[float] = Field(
        default=None, description="最大 reset 间隔（小时）"
    )
    interval_uncertainty: Optional[float] = Field(
        default=None, ge=0.0,
        description="reset 间隔估计的不确定性（小时），用于平滑 time_pressure"
    )
    sample_count: int = Field(
        ..., ge=0, description="用于估计的 interval 样本数"
    )
    interval_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="对平均间隔估计的置信度"
    )
    prior_weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="先验默认周期在 posterior 中的权重"
    )
    params: dict[str, float] = Field(
        default_factory=dict,
        description="模型参数（如 alpha, beta_time, beta_tibo 等）"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="状态更新时间（UTC）"
    )

    def get_param(self, name: str, default: float) -> float:
        """读取参数，缺失时返回 default。"""
        return self.params.get(name, default)


class ModelStateManager:
    """管理 model_state.json 的读写。"""

    def __init__(self, state_path: Path):
        self._state_path = state_path

    def load(self) -> Optional[ModelState]:
        """从磁盘加载 ModelState，文件不存在时返回 None。"""
        if not self._state_path.exists():
            return None
        try:
            return ModelState.model_validate_json(
                self._state_path.read_text(encoding="utf-8")
            )
        except Exception:
            return None

    def save(self, state: ModelState) -> None:
        """保存 ModelState 到磁盘。"""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
