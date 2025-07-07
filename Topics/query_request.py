from pydantic import BaseModel, Field
from enum import Enum

class DatasetEnum(str, Enum):
    antique = "antique/train"
    quora = "beir/quora/test"

class SearchMode(str, Enum):
    tfidf = "tfidf"
    bert = "bert"
    hybrid = "hybrid"

class QueryRequest(BaseModel):
    dataset: DatasetEnum = Field(..., title="📚 مجموعة البيانات")
    query: str = Field(..., title="📝 الاستعلام")
    mode: SearchMode = Field(..., title="⚙️ نوع التمثيل")
