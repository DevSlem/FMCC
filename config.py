from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
  sample_rate: int = 16000
  folder_name: str = "data"