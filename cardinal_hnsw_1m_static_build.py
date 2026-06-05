#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pathlib
import time
from datetime import datetime

import numpy as np
from pymilvus import DataType, MilvusClient

from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import Dataset


LOG = logging.getLogger("cardinal_hnsw_1m_static_build")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def wait_index(client: MilvusClient, collection: str, index_name: str, interval: int = 15):
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


def wait_compaction(client: MilvusClient, job_id: int, interval: int = 5):
    while True:
        state = client.get_compaction_state(job_id)
        LOG.info("compaction %s state: %s", job_id, state)
        if str(state) in {"Completed", "Failed", "Timeout"}:
            return str(state)
        time.sleep(interval)


def prepare_dataset():
    manager = Dataset.COHERE.manager(1_000_000)
    manager.prepare(DatasetSource[os.getenv("DATASET_SOURCE", "S3")])
    return manager


def create_collection(client: MilvusClient, collection: str, dim: int, m: int, ef_construction: int):
    if client.has_collection(collection):
        LOG.info("drop existing collection: %s", collection)
        client.drop_collection(collection)

    schema = MilvusClient.create_schema()
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("id", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector_idx",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": m, "efConstruction": ef_construction},
    )
    index_params.add_index(
        field_name="id",
        index_name="id_sort_idx",
        index_type="STL_SORT",
    )

    LOG.info("create collection without load: %s", collection)
    client.create_collection(
        collection_name=collection,
        schema=schema,
        num_shards=1,
        consistency_level="Session",
    )
    client.create_index(collection, index_params)


def insert_dataset(client: MilvusClient, collection: str, manager, batch_size: int):
    inserted = 0
    start = time.perf_counter()
    for frame in manager:
        ids = frame[manager.data.train_id_field].tolist()
        emb = np.stack(frame[manager.data.train_vector_field])
        emb = emb / np.linalg.norm(emb, axis=1)[:, np.newaxis]
        vectors = emb.tolist()
        del emb

        for offset in range(0, len(ids), batch_size):
            end = min(offset + batch_size, len(ids))
            rows = [
                {"pk": ids[i], "id": ids[i], "vector": vectors[i]}
                for i in range(offset, end)
            ]
            res = client.insert(collection, rows)
            inserted += int(res.get("insert_count", len(rows)))
            if inserted % 100000 == 0:
                LOG.info("inserted rows: %s", inserted)
    dur = time.perf_counter() - start
    LOG.info("insert finished: count=%s dur=%.2fs", inserted, dur)
    return inserted, dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--token", default="db_admin:Milvus123")
    parser.add_argument("--collection", default="VDBBenchStatic1M")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--m", type=int, default=48)
    parser.add_argument("--ef-construction", type=int, default=400)
    parser.add_argument("--compact-target-mb", type=int, default=6144)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    summary = {
        "started_at": utc_now(),
        "collection": args.collection,
        "uri": args.uri,
        "batch_size": args.batch_size,
        "index": {"type": "HNSW", "metric": "COSINE", "M": args.m, "efConstruction": args.ef_construction},
        "steps": [],
    }

    manager = prepare_dataset()
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    try:
        create_collection(client, args.collection, manager.data.dim, args.m, args.ef_construction)
        summary["steps"].append({"step": "create_collection", "finished_at": utc_now()})

        inserted, insert_dur = insert_dataset(client, args.collection, manager, args.batch_size)
        summary["inserted"] = inserted
        summary["insert_duration_seconds"] = insert_dur
        summary["steps"].append({"step": "insert", "finished_at": utc_now()})

        LOG.info("flush start")
        client.flush(args.collection, timeout=600)
        LOG.info("flush done")
        summary["steps"].append({"step": "flush", "finished_at": utc_now()})

        vector_idx = wait_index(client, args.collection, "vector_idx")
        id_idx = wait_index(client, args.collection, "id_sort_idx")
        summary["vector_idx_before_compact"] = vector_idx
        summary["id_idx_before_compact"] = id_idx
        summary["steps"].append({"step": "wait_index", "finished_at": utc_now()})

        try:
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
        except Exception as exc:
            LOG.exception("compact failed")
            summary["compact_error"] = repr(exc)

        LOG.info("load start")
        client.load_collection(args.collection, replica_number=1, timeout=1800)
        LOG.info("load done")
        summary["steps"].append({"step": "load", "finished_at": utc_now()})

        summary["stats"] = client.get_collection_stats(args.collection)
        summary["load_state"] = str(client.get_load_state(args.collection))
        summary["vector_idx_final"] = client.describe_index(args.collection, "vector_idx")
        summary["id_idx_final"] = client.describe_index(args.collection, "id_sort_idx")
        summary["finished_at"] = utc_now()
    finally:
        client.close()

    if args.summary:
        path = pathlib.Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    LOG.info("summary: %s", json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
