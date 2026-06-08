from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig
from ..milvus.config import IndexType, MilvusIndexConfig


class ZillizCloudConfig(DBConfig):
    uri: SecretStr
    user: str
    password: SecretStr
    num_shards: int = 1
    collection_name: str = "ZillizCloudVDBBench"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user,
            "password": self.password.get_secret_value(),
            "num_shards": self.num_shards,
            "collection_name": self.collection_name,
        }


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX
    level: int = 1
    num_shards: int = 1
    cardinal_expr_downpush: bool = False
    cardinal_expr_threshold: int | None = None
    cardinal_expr_filtered_out_count: int | None = None
    cardinal_expr_filter_ratio: float | None = None

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"shardsNum": self.num_shards},
        }

    def search_param(self) -> dict:
        params = {
            "level": self.level,
        }
        if self.cardinal_expr_downpush:
            params["cardinal_expr_downpush"] = True
            if self.cardinal_expr_threshold is not None:
                params["cardinal_expr_threshold"] = self.cardinal_expr_threshold
            if self.cardinal_expr_filtered_out_count is not None:
                params["cardinal_expr_filtered_out_count"] = self.cardinal_expr_filtered_out_count
            if self.cardinal_expr_filter_ratio is not None:
                params["cardinal_expr_filter_ratio"] = self.cardinal_expr_filter_ratio

        return {
            "metric_type": self.parse_metric(),
            "params": params,
        }
