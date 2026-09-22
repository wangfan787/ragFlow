"""共享小型合成评测数据，不读取真实语料或运行历史。"""
import json
from pathlib import Path

import pytest
from evaluation.build_dataset import BuildConfig, TASKS, build_dataset


def _source_fixture(root: Path) -> Path:
    source = root / "CRUD_RAG"
    split_dir = source / "data" / "crud_split"
    distractor_dir = source / "data" / "80000_docs"
    split_dir.mkdir(parents=True)
    distractor_dir.mkdir(parents=True)
    rows = {}
    for task_index, task in enumerate(TASKS, start=1):
        doc_count = task_index
        task_rows = []
        for row_index in range(3):
            row = {
                "ID": f"event-{task_index}-{row_index}",
                "event": f"事件 {task_index}-{row_index}",
                "questions": f"任务 {task_index} 问题 {row_index}",
                "answers": f"任务 {task_index} 答案 {row_index}",
            }
            for doc_index in range(1, doc_count + 1):
                row[f"news{doc_index}"] = f"正例 {task_index}-{row_index}-{doc_index} 的独立事实"
            task_rows.append(row)
        rows[task] = task_rows
    (split_dir / "split_merged.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    distractors = "\n".join(f"干扰新闻 {index} 包含任务主题但事实不同" for index in range(60)) + "\n"
    (distractor_dir / "part-1").write_text(distractors, encoding="utf-8")
    return source


@pytest.fixture
def source_dataset(tmp_path):
    return _source_fixture(tmp_path)


@pytest.fixture
def mini_dataset(tmp_path, source_dataset):
    output = tmp_path / "mini"
    build_dataset(
        source_dataset, output,
        BuildConfig(seed=42, per_task=2, dev_per_task=1, corpus_size=20, hard_negative_count=3),
    )
    return output
