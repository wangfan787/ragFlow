"""T2 评测回归：分块映射、评分、缓存完整性和断点恢复。"""
import hashlib
import json

import pytest


class FakeEmbedding:
    model = "fake-v1"
    dimensions = 2

    def __init__(self):
        self.inputs: list[str] = []

    def embed_documents(self, texts):
        self.inputs.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_t2_preparation_emits_v2_chunk_mapping_with_fingerprint(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dataset import embed_t2retrieval
    from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter

    pq.write_table(pa.Table.from_pylist([
        {"_id": "plain", "title": "Plain", "text": "First paragraph.\n\nSecond paragraph."},
        {"_id": "html", "title": "HTML", "text": "<h1>Title</h1><p>Answer text.</p>"},
    ]), tmp_path / "corpus.parquet")
    monkeypatch.setattr(embed_t2retrieval, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(embed_t2retrieval, "OUTPUT_DIR", tmp_path)
    manifest, progress = embed_t2retrieval._prepare_manifest(
        "corpus", ProductionRagAdapter(), 2
    )
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert progress["complete"] is True
    assert progress["schema_version"] == "t2-production-rag-v2"
    assert records and all(row["chunk_id"] and row["doc_id"] for row in records)
    assert {row["doc_id"] for row in records} == {"plain", "html"}
    assert all(row["embedding_text"] for row in records)


def test_t2_family_evaluator_means_hit_children_before_document_ranking():
    import numpy as np
    from dataset.evaluate_t2retrieval import _family_document_ranking

    mapping = [
        {"chunk_id": "a1", "parent_id": "pa", "doc_id": "A"},
        {"chunk_id": "a2", "parent_id": "pa", "doc_id": "A"},
        {"chunk_id": "b1", "parent_id": "pb", "doc_id": "B"},
    ]
    # A has the highest individual Child but family mean .5; B wins with .7.
    ranked = _family_document_ranking(
        np.array([0, 1, 2]),
        np.array([0.9, 0.1, 0.7]),
        mapping,
        aggregation="max",
        top_n=3,
        top_k=2,
    )
    assert ranked == ["B", "A"]


def test_t2_evaluator_rejects_incomplete_artifact_and_ann_matches_exact(tmp_path, monkeypatch):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset import evaluate_t2retrieval

    monkeypatch.setattr(evaluate_t2retrieval, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(evaluate_t2retrieval, "DATASET_DIR", tmp_path)

    def artifact(name, vectors, mapping):
        vectors = np.asarray(vectors, dtype="float32")
        mmap_path = tmp_path / f"{name}.f32.mmap"
        mmap = np.memmap(mmap_path, dtype="float32", mode="w+", shape=vectors.shape)
        mmap[:] = vectors
        mmap.flush()
        vectors_bytes = mmap_path.stat().st_size
        vectors_sha256 = hashlib.sha256(mmap_path.read_bytes()).hexdigest()
        mapping_text = "\n".join(json.dumps(row) for row in mapping) + "\n"
        mapping_bytes = len(mapping_text.encode("utf-8"))
        mapping_sha256 = hashlib.sha256(mapping_text.encode("utf-8")).hexdigest()
        exclusions_path = tmp_path / f"{name}.exclusions.jsonl"
        exclusions_path.write_text("", encoding="utf-8")
        contract = {
            "schema_version": "t2-production-rag-v2", "complete": True,
            "algorithm_version": "production-adapter-v2.3-child-body",
            "input_fingerprint": "fp", "profile_hash": "profile", "model": "m",
            "source_file": f"{name}.parquet", "source_file_sha256": "source-sha",
            "source_total_rows": len({str(row["doc_id"]) for row in mapping}),
            "source_start_row": 0,
            "source_end_row": len({str(row["doc_id"]) for row in mapping}),
            "source_selection": "all" if name == "corpus" else "prefix",
            "backend": "b", "dimension": vectors.shape[1], "shape": list(vectors.shape),
            "dtype": "float32", "vector_rows": vectors.shape[0], "document_rows": len({
                str(row["doc_id"]) for row in mapping
            }),
            "mapping_file": f"{name}.chunks.jsonl",
            "mapping_bytes": mapping_bytes,
            "mapping_sha256": mapping_sha256,
            "excluded_documents": 0,
            "exclusions_file": exclusions_path.name,
            "exclusions_bytes": 0,
            "exclusions_sha256": hashlib.sha256(b"").hexdigest(),
            "vectors_bytes": vectors_bytes,
            "vectors_sha256": vectors_sha256,
        }
        (tmp_path / f"{name}.metadata.json").write_text(json.dumps(contract), encoding="utf-8")
        (tmp_path / f"{name}.embedding.progress.json").write_text(
            json.dumps({**contract, "next_vector_row": vectors.shape[0], "complete": True}),
            encoding="utf-8",
        )
        (tmp_path / f"{name}.prepare.progress.json").write_text(
            json.dumps({
                "complete": True, "schema_version": contract["schema_version"],
                "algorithm_version": contract["algorithm_version"],
                "input_fingerprint": contract["input_fingerprint"],
                "profile_hash": contract["profile_hash"], "artifact_rows": vectors.shape[0],
                "source_file": contract["source_file"],
                "source_file_sha256": contract["source_file_sha256"],
                "source_total_rows": contract["source_total_rows"],
                "source_start_row": contract["source_start_row"],
                "source_end_row": contract["source_end_row"],
                "source_selection": contract["source_selection"],
                "mapping_bytes": mapping_bytes, "mapping_sha256": mapping_sha256,
                "excluded_documents": 0, "exclusions_file": exclusions_path.name,
                "exclusions_bytes": 0, "exclusions_sha256": hashlib.sha256(b"").hexdigest(),
            }), encoding="utf-8"
        )
        (tmp_path / f"{name}.chunks.jsonl").write_text(
            mapping_text, encoding="utf-8"
        )

    artifact(
        "corpus",
        [[1, 0], [0.8, 0.2], [0, 1]],
        [
            {"chunk_id": "a1", "parent_id": "pa", "doc_id": "A"},
            {"chunk_id": "a2", "parent_id": "pa", "doc_id": "A"},
            {"chunk_id": "b1", "parent_id": "pb", "doc_id": "B"},
        ],
    )
    artifact("queries", [[1, 0]], [{"chunk_id": "q", "doc_id": "q1"}])
    pq.write_table(
        pa.table({"query-id": ["q1"], "corpus-id": ["A"], "score": [1]}),
        tmp_path / "qrels.parquet",
    )
    common = dict(top_k=2, candidate_top_k=10, query_batch=1, corpus_batch=2,
                  aggregation="max", top_n=3, enforce_scope=False, expected_query_count=None)
    exact = evaluate_t2retrieval.evaluate(**common, engine="exact")
    ann = evaluate_t2retrieval.evaluate(**common, engine="faiss-hnsw")
    assert exact["Recall@2"] == ann["Recall@2"] == 1.0
    assert exact["qrels_coverage"] == 1.0

    pq.write_table(
        pa.table({"query-id": ["q1"], "corpus-id": ["B"], "score": [1]}),
        tmp_path / "qrels.parquet",
    )
    below_recall_cutoff = evaluate_t2retrieval.evaluate(
        **{**common, "top_k": 1}, engine="exact"
    )
    assert below_recall_cutoff["Recall@1"] == 0.0
    assert below_recall_cutoff["MRR@10"] == 0.5
    assert below_recall_cutoff["nDCG@10"] > 0.0

    exclusions_path = tmp_path / "corpus.exclusions.jsonl"
    exclusions_path.write_text("corrupt", encoding="utf-8")
    try:
        evaluate_t2retrieval._load_artifact("corpus")
    except ValueError as exc:
        assert "exclusions digest/byte contract" in str(exc)
    else:
        raise AssertionError("corrupt exclusions must not be evaluated")
    exclusions_path.write_text("", encoding="utf-8")

    progress_path = tmp_path / "corpus.embedding.progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["complete"] = False
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    try:
        evaluate_t2retrieval._load_artifact("corpus")
    except ValueError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("incomplete vector progress must not be evaluated")


def test_t2_embedding_resume_rejects_missing_mmap_and_short_committed_jsonl(tmp_path, monkeypatch):
    import pytest
    from dataset import embed_t2retrieval

    monkeypatch.setattr(embed_t2retrieval, "OUTPUT_DIR", tmp_path)
    manifest_path = tmp_path / "corpus.chunks.jsonl"
    mapping_text = json.dumps({"embedding_text": "content"}) + "\n"
    manifest_path.write_text(mapping_text, encoding="utf-8")
    exclusions_path = tmp_path / "corpus.exclusions.jsonl"
    exclusions_path.write_text("", encoding="utf-8")
    prepared = {
        "algorithm_version": "production-adapter-v2.3-child-body",
        "input_fingerprint": "fp",
        "profile_hash": "profile",
        "source_file": "corpus.parquet",
        "source_file_sha256": "source-sha",
        "source_total_rows": 1,
        "source_start_row": 0,
        "source_end_row": 1,
        "source_selection": "all",
        "total_documents": 1,
        "artifact_rows": 1,
        "mapping_bytes": len(mapping_text.encode("utf-8")),
        "mapping_sha256": hashlib.sha256(mapping_text.encode("utf-8")).hexdigest(),
        "excluded_documents": 0,
        "exclusions_file": exclusions_path.name,
        "exclusions_bytes": 0,
        "exclusions_sha256": hashlib.sha256(b"").hexdigest(),
    }
    model = FakeEmbedding()
    embed_t2retrieval._embed_manifest("corpus", model, manifest_path, prepared, batch_size=1)
    mmap_path = tmp_path / "corpus.f32.mmap"
    with mmap_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x01]))
    with pytest.raises(ValueError, match="digest does not match committed contract"):
        embed_t2retrieval._embed_manifest("corpus", model, manifest_path, prepared, batch_size=1)

    mmap_path.unlink()
    with pytest.raises(ValueError, match="mmap is missing"):
        embed_t2retrieval._embed_manifest("corpus", model, manifest_path, prepared, batch_size=1)

    short_path = tmp_path / "short.jsonl"
    short_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="shorter than committed progress"):
        embed_t2retrieval._require_committed_prefix(short_path, 10, "fixture")


def test_t2_prepare_recovers_uncommitted_suffix_and_never_recertifies_complete_mapping(
    tmp_path, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from backend.src.apps.services.benchmark_adapter import ProductionRagAdapter
    from dataset import embed_t2retrieval

    monkeypatch.setattr(embed_t2retrieval, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(embed_t2retrieval, "OUTPUT_DIR", tmp_path / "out")
    pq.write_table(
        pa.table(
            {
                "_id": ["one", "two", "bad"],
                "text": ["第一篇有效正文。", "第二篇有效正文。", " </pre>"],
                "title": ["", "", ""],
            }
        ),
        tmp_path / "corpus.parquet",
    )
    base = ProductionRagAdapter()

    class FailSecondDocument:
        chunk_config = base.chunk_config
        text_builder = base.text_builder

        def __init__(self):
            self.calls = 0

        def child_inputs(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("unexpected adapter failure")
            return base.child_inputs(**kwargs)

    with pytest.raises(ValueError, match="unexpected adapter failure"):
        embed_t2retrieval._prepare_manifest("corpus", FailSecondDocument(), None)

    manifest_path = tmp_path / "out/corpus.chunks.jsonl"
    exclusions_path = tmp_path / "out/corpus.exclusions.jsonl"
    with manifest_path.open("ab") as handle:
        handle.write(b'{"uncommitted":')
    with exclusions_path.open("ab") as handle:
        handle.write(b'{"uncommitted":')

    _, progress = embed_t2retrieval._prepare_manifest("corpus", base, None)
    assert progress["complete"] is True
    assert progress["excluded_documents"] == 1
    assert json.loads(exclusions_path.read_text(encoding="utf-8"))["doc_id"] == "bad"

    original = manifest_path.read_bytes()
    corrupted = original.replace(b'"embedding_text"', b'"embedding_Text"', 1)
    assert len(corrupted) == len(original) and corrupted != original
    manifest_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="digest does not match committed contract"):
        embed_t2retrieval._prepare_manifest("corpus", base, None)
