"""Wrapper around the ZillizCloud vector database over VectorDB"""

import logging

from ..api import DBCaseConfig
from ..milvus.milvus import Milvus

log = logging.getLogger(__name__)
ZILLIZ_CLOUD_FORCE_MERGE_TARGET_SIZE = 6 * 1024 * 1024 * 1024


class ZillizCloud(Milvus):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVDBBench",
        drop_old: bool = False,
        name: str = "ZillizCloud",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
            name=name,
            **kwargs,
        )

    def _optimize(self):
        log.info(f"{self.name} optimizing before search")
        try:
            self.client.flush(self.collection_name)
            self._wait_for_index()
            if self.case_config.is_gpu_index:
                log.debug("skip force merge compaction for gpu index type.")
            else:
                try:
                    compaction_id = self.client.compact(
                        self.collection_name,
                        target_size=ZILLIZ_CLOUD_FORCE_MERGE_TARGET_SIZE,
                    )
                    if compaction_id > 0:
                        self._wait_for_compaction(compaction_id)
                    log.info(f"{self.name} force merge compaction completed.")
                    self._wait_for_index()
                except Exception as e:
                    log.warning(f"{self.name} compact error: {e}")
                    code = getattr(e, "code", None)
                    if callable(code):
                        code = code()
                    if getattr(code, "name", None) == "PERMISSION_DENIED":
                        log.warning("Skip compact due to permission denied.")
                    else:
                        raise e from None
            self.client.refresh_load(self.collection_name)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None
