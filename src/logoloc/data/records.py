from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x2, self.y2)

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclasses.dataclass(frozen=True)
class LogoInstance:
    class_name: str
    bbox: BBox


@dataclasses.dataclass
class ImageRecord:
    image_path: Path
    split: str
    width: int
    height: int
    instances: list[LogoInstance] = dataclasses.field(default_factory=list)
    image_id: str = ""

    @property
    def is_no_logo(self) -> bool:
        return len(self.instances) == 0

    @property
    def class_names(self) -> set[str]:
        return {inst.class_name for inst in self.instances}
