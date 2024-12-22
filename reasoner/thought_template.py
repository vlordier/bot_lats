from pydantic import BaseModel
from typing import List, Dict


class ThoughtTemplate(BaseModel):
    id: str
    nodes: List[str]
    edges: List[tuple]
    labels: Dict[str, str]
    metadata: Dict[str, str]
