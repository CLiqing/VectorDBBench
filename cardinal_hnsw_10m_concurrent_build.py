#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pathlib
import time
from datetime import datetime

from pymilvus import MilvusClient

from vectordb_bench.backend.clients import MetricType
from vectordb_bench.backend.clients.milvus.config import HNSWConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import Dataset
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner.concurrent_runner import ConcurrentInsertRunner


LOG = logging.getLogger("cardinal_hnsw_10m_concurrent_build")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_summary(path: str, summary: dict):
    if not path:
        return
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")


def wait_index(client: MilvusClient, collection: str, index_name: str, interval: int = 30):
    while True:
        info = client.describe_index(collection, index_name)
        LOG.info("index %s status: %s", index_name, info)
        if (
            str(info.get("state")).lower() == "finished"
            and int(info.get("indexed_rows", 0)) >= int(info.get("total_rows", 0))
            and int(info.get("total_rows", 0)) > 0
        ):
            return info
        time.sleep(interval)


def wait_compaction(client: MilvusClient, job_id: int, interval: int = 10):
    while True:
        state = client.get_compaction_state(job_id)
        LOG.info("compaction %s state: %s", job_id, state)
        if str(state) in {"Completed", "Failed", "Timeout"}:
            return str(state)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", default="db_admin")
    parser.add_argument("--password", default="Milvus123")
    parser.add_argument("--collection", default="VDBBenchStatic10M")
    parser.add_argument("--load-concurrency", type=int, default=16)
    parser.add_argument("--m", type=int, default=48)
    parser.add_argument("--ef-construction", type=int, default=400)
    parser.add_argument("--ef-search", type=int, default=300)
    parser.add_argument("--compact-target-mb", type=int, default=6144)
    parser.add_argument("--summary", default="")
    parser.add_argument("--no-drop-existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("VDBBENCH_STREAM_TRAIN_DOWNLOAD", "1")

    summary = {
        "started_at": utc_now(),
        "collection": args.collection,
        "uri": args.uri,
        "dataset": "Cohere 10M",
        "load_concurrency": args.load_concurrency,
        "index": {
            "type": "HNSW",
            "metric": "COSINE",
            "M": args.m,
            "efConstruction": args.ef_construction,
            "ef": args.ef_search,
        },
        "compact_target_mb": args.compact_target_mb,
        "steps": [],
    }
    write_summary(args.summary, summary)

    manager = Dataset.COHERE.manager(10_000_000)
    manager.prepare(DatasetSource[os.getenv("DATASET_SOURCE", "S3")])
    summary["dim"] = manager.data.dim
    summary["metric_type"] = str(manager.data.metric_type)
    summary["train_file_count"] = manager.data.file_count
    summary["steps"].append({"step": "prepare_dataset", "finished_at": utc_now()})
    write_summary(args.summary, summary)

    db_config = {
        "uri": args.uri,
        "user": args.user,
        "password": args.password,
        "num_shards": 1,
        "replica_number": 1,
    }
    index_config = HNSWConfig(
        metric_type=MetricType.COSINE,
        M=args.m,
        efConstruction=args.ef_construction,
        ef=args.ef_search,
    )
    db = Milvus(
        dim=manager.data.dim,
        db_config=db_config,
        db_case_config=index_config,
        collection_name=args.collection,
        drop_old=not args.no_drop_existing,
        with_scalar_labels=False,
    )
    summary["steps"].append({"step": "create_collection", "finished_at": utc_now()})
    write_summary(args.summary, summary)

    runner = ConcurrentInsertRunner(
        db=db,
        dataset=manager,
        normalize=False,
        filters=non_filter,
        timeout=manager.data.size,
        max_workers=args.load_concurrency,
    )
    start = time.perf_counter()
    inserted = runner.run()
    insert_dur = time.perf_counter() - start
    summary["inserted"] = inserted
    summary["insert_duration_seconds"] = insert_dur
    summary["insert_rows_per_second"] = inserted / insert_dur
    summary["steps"].append({"step": "concurrent_insert", "finished_at": utc_now()})
    write_summary(args.summary, summary)
    LOG.info("concurrent insert finished: count=%s dur=%.2fs rows/s=%.2f", inserted, insert_dur, inserted / insert_dur)

    client = MilvusClient(uri=args.uri, user=args.user, password=args.password, timeout=180)
    try:
        LOG.info("flush start")
        client.flush(args.collection, timeout=1800)
        LOG.info("flush done")
        summary["steps"].append({"step": "flush", "finished_at": utc_now()})
        write_summary(args.summary, summary)

        vector_idx = wait_index(client, args.collection, "vector_idx")
        id_idx = wait_index(client, args.collection, "id_sort_idx")
        summary["vector_idx_before_compact"] = vector_idx
        summary["id_idx_before_compact"] = id_idx
        summary["steps"].append({"step": "wait_index_before_compact", "finished_at": utc_now()})
        write_summary(args.summary, summary)

        LOG.info("compact start target=%smb", args.compact_target_mb)
        job_id = client.compact(
            args.collection,
            target_size=args.compact_target_mb,
            target_size_unit="mb",
            timeout=120,
        )
        summary["compaction_job_id"] = job_id
        if job_id:
            summary["compaction_state"] = wait_compaction(client, job_id)
        summary["steps"].append({"step": "compact", "finished_at": utc_now()})
        write_summary(args.summary, summary)

        vector_idx = wait_index(client, args.collection, "vector_idx")
        id_idx = wait_index(client, args.collection, "id_sort_idx")
        summary["vector_idx_after_compact"] = vector_idx
        summary["id_idx_after_compact"] = id_idx
        summary["steps"].append({"step": "wait_index_after_compact", "finished_at": utc_now()})
        write_summary(args.summary, summary)

        LOG.info("load/refresh start")
        try:
            client.refresh_load(args.collection, timeout=1800)
        except Exception:
            client.load_collection(args.collection, replica_number=1, timeout=1800)
        LOG.info("load/refresh done")
        summary["steps"].append({"step": "load", "finished_at": utc_now()})

        summary["stats"] = client.get_collection_stats(args.collection)
        summary["load_state"] = str(client.get_load_state(args.collection))
        summary["vector_idx_final"] = client.describe_index(args.collection, "vector_idx")
        summary["id_idx_final"] = client.describe_index(args.collection, "id_sort_idx")
        summary["finished_at"] = utc_now()
    finally:
        client.close()
        write_summary(args.summary, summary)

    LOG.info("summary: %s", json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
