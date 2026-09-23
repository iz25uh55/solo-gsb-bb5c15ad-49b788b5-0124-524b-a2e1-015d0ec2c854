"""端到端行为测试。"""
from __future__ import annotations

import hashlib
import random
import time

from fastapi.testclient import TestClient

from app import settings
from app.cleanup import run_cleanup
from app import storage


def _create(client: TestClient, data: bytes, name: str = "data.bin", chunk_size: int = 0):
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "chunk_size": chunk_size,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def test_out_of_order_upload_and_download(client: TestClient) -> None:
    rng = random.Random(42)
    data = bytes(rng.randrange(256) for _ in range(1024 * 1024))  # 1 MiB
    sid = _create(client, data, chunk_size=256 * 1024)

    # 完成前下载必须 404——不得存在可下载成品。
    assert client.get(f"/api/v1/sessions/{sid}/download").status_code == 404

    # 切成大小不一的 9 片，打乱顺序提交。
    cuts = [0, 37, 120_000, 120_001, 260_000, 500_000, 700_000, 999_999, 1_000_000]
    ranges = list(zip(cuts, cuts[1:] + [len(data)]))
    rng.shuffle(ranges)
    for start, end in ranges:
        resp = client.put(
            f"/api/v1/sessions/{sid}/chunks?start={start}&end={end}",
            content=data[start:end],
        )
        assert resp.status_code == 200, (start, end, resp.text)
        assert resp.json()["received"] == {"start": start, "end": end}

    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["fully_received"] is True
    assert status["missing_ranges"] == []
    assert status["status"] == "uploading"
    assert status["bytes_received"] == len(data)

    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["status"] == "completed"
    assert body["complete"] is True

    dl = client.get(f"/api/v1/sessions/{sid}/download")
    assert dl.status_code == 200
    assert dl.content == data
    assert "attachment" in dl.headers["content-disposition"]

    # 已完成的会话不再接受分片；complete 幂等。
    dup = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=1", content=data[:1]
    )
    assert dup.status_code == 409
    again = client.post(f"/api/v1/sessions/{sid}/complete")
    assert again.status_code == 200
    assert again.json()["status"] == "completed"


def test_reject_out_of_bounds_overlap_and_length(client: TestClient) -> None:
    data = b"abcdefghij" * 100
    sid = _create(client, data)
    total = len(data)

    # 越界（end 超过 total_size）
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=990&end=1001",
        content=data[990:1000] + b"x",
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "chunk range exceeds total_size"

    # 非法区间 start >= end
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=10&end=10", content=b""
    )
    assert r.status_code == 400

    # content-length 与区间不一致
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=10", content=b"short"
    )
    assert r.status_code == 400
    assert "content-length" in r.json()["detail"]["error"]

    # 正常收一片
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=100", content=data[:100]
    )
    assert r.status_code == 200, r.text

    # 完全重叠
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=50&end=150",
        content=data[50:150],
    )
    assert r.status_code == 409
    assert "overlaps" in r.json()["detail"]["error"]

    # 边界相邻不算重叠：[100,200) 合法
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=100&end=200",
        content=data[100:200],
    )
    assert r.status_code == 200, r.text

    # 状态接口返回的缺失区间必须准确
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["missing_ranges"] == [{"start": 200, "end": total}]


def test_sha256_mismatch_keeps_session_unfinished(client: TestClient) -> None:
    data = bytes(range(256)) * 4
    # 用错误 sha 创建会话并提交正确字节
    bad = client.post(
        "/api/v1/sessions",
        json={
            "filename": "bad.bin",
            "total_size": len(data),
            "sha256": "0" * 64,
        },
    ).json()["session_id"]
    client.put(
        f"/api/v1/sessions/{bad}/chunks?start=0&end={len(data)}", content=data
    )
    r = client.post(f"/api/v1/sessions/{bad}/complete")
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "sha256 mismatch"

    # 失败后仍不可下载，会话仍是 uploading
    assert client.get(f"/api/v1/sessions/{bad}/download").status_code == 404
    status = client.get(f"/api/v1/sessions/{bad}").json()
    assert status["status"] == "uploading"
    assert status["fully_received"] is True


def test_complete_with_gaps_reports_missing(client: TestClient) -> None:
    data = bytes(r % 251 for r in range(5000))
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=100&end=1000",
        content=data[100:1000],
    )
    r = client.post(f"/api/v1/sessions/{sid}/complete")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["missing_ranges"] == [
        {"start": 0, "end": 100},
        {"start": 1000, "end": 5000},
    ]
    assert client.get(f"/api/v1/sessions/{sid}/download").status_code == 404


def test_abort_session(client: TestClient) -> None:
    data = b"hello"
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=5", content=data
    )
    r = client.delete(f"/api/v1/sessions/{sid}")
    assert r.status_code == 204
    # 删除幂等
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404
    assert not storage.parts_dir(sid).exists()


def test_empty_file_roundtrip(client: TestClient) -> None:
    sid = _create(client, b"", name="empty.txt")
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["fully_received"] is True
    r = client.post(f"/api/v1/sessions/{sid}/complete")
    assert r.status_code == 200, r.text
    dl = client.get(f"/api/v1/sessions/{sid}/download")
    assert dl.status_code == 200
    assert dl.content == b""


def test_expired_session_is_rejected_and_cleaned(client: TestClient, monkeypatch) -> None:
    data = b"x" * 1000
    # 创建前就把 TTL 调小：expires_at 在创建/分片提交时按当前 TTL 计算。
    monkeypatch.setattr(settings, "ttl_seconds", 1)
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=100", content=data[:100]
    )

    time.sleep(1.1)
    # 后台轮询间隔很长，直接驱动一轮清理（生产中由 janitor 周期调用）。
    expired = run_cleanup()
    assert sid in expired

    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=100&end=200",
        content=data[100:200],
    )
    assert r.status_code == 410
    status = client.get(f"/api/v1/sessions/{sid}")
    assert status.status_code == 410
    assert client.get(f"/api/v1/sessions/{sid}/download").status_code == 404
    assert not storage.parts_dir(sid).exists()


def test_unknown_session_errors(client: TestClient) -> None:
    sid = "deadbeef" * 4
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=1", content=b"x"
    )
    assert r.status_code == 404
    assert client.post(f"/api/v1/sessions/{sid}/complete").status_code == 404


def test_invalid_create_requests(client: TestClient) -> None:
    r = client.post(
        "/api/v1/sessions",
        json={"filename": "x.bin", "total_size": -1, "sha256": "a" * 64},
    )
    assert r.status_code == 422

    r = client.post(
        "/api/v1/sessions",
        json={"filename": "x.bin", "total_size": 1, "sha256": "xyz"},
    )
    assert r.status_code == 422

    r = client.post(
        "/api/v1/sessions",
        json={"filename": "   ", "total_size": 1, "sha256": "a" * 64},
    )
    assert r.status_code == 422


def test_concurrent_overlapping_chunks_only_one_wins(client: TestClient) -> None:
    import threading

    data = b"".join(bytes([i % 251]) for i in range(100))
    sid = _create(client, data)
    results: list[int] = []
    barrier = threading.Barrier(5)

    def worker() -> None:
        barrier.wait()  # 尽量放大同时提交同一区间的竞态窗口
        r = client.put(
            f"/api/v1/sessions/{sid}/chunks?start=0&end=100", content=data
        )
        results.append(r.status_code)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results).count(200) == 1
    assert all(code == 409 for code in results if code != 200)
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["bytes_received"] == 100
    assert status["fully_received"] is True


def test_storage_summary(client: TestClient) -> None:
    import uuid

    from app import db

    def fetch() -> dict:
        resp = client.get("/api/v1/storage/summary")
        assert resp.status_code == 200, resp.text
        return resp.json()

    def row_for(body: dict, status: str) -> dict:
        return next(r for r in body["statuses"] if r["status"] == status)

    # 数据库为整个测试会话共享，一律用基线差值断言。
    baseline = fetch()
    # 固定返回四种状态，即使计数为零也保留该行；总计等于状态行求和。
    assert [r["status"] for r in baseline["statuses"]] == [
        "uploading",
        "completed",
        "aborted",
        "expired",
    ]
    for key in ("session_count", "declared_bytes", "stored_bytes"):
        assert baseline["total"][key] == sum(r[key] for r in baseline["statuses"])

    # uploading：declared=1000，stored 取 bytes_received=100。
    up_data = b"u" * 1000
    up_sid = _create(client, up_data)
    r = client.put(
        f"/api/v1/sessions/{up_sid}/chunks?start=0&end=100", content=up_data[:100]
    )
    assert r.status_code == 200, r.text
    before = client.get(f"/api/v1/sessions/{up_sid}").json()

    # completed：declared=stored=total_size=5。
    done_data = b"hello"
    done_sid = _create(client, done_data)
    client.put(f"/api/v1/sessions/{done_sid}/chunks?start=0&end=5", content=done_data)
    assert client.post(f"/api/v1/sessions/{done_sid}/complete").status_code == 200

    # aborted/expired：主流程里 aborted 会被硬删除、expired 由后台清理翻转，
    # 这里直接落库两行，专注验证聚合口径（stored 恒为 0）。
    now_iso = storage.iso(storage.utcnow())
    with db.immediate() as tx:
        for sid, status, declared, received in (
            (uuid.uuid4().hex, "aborted", 700, 300),
            (uuid.uuid4().hex, "expired", 900, 400),
        ):
            tx.execute(
                "INSERT INTO upload_sessions ("
                "id, filename, total_size, sha256_expected, chunk_size, "
                "status, bytes_received, created_at, expires_at, last_access_at"
                ") VALUES (?, 'ops.bin', ?, ?, 0, ?, ?, ?, ?, ?)",
                (sid, declared, "a" * 64, status, received, now_iso, now_iso, now_iso),
            )

    after = fetch()
    expected = {
        "uploading": (1, 1000, 100),
        "completed": (1, 5, 5),
        "aborted": (1, 700, 0),
        "expired": (1, 900, 0),
    }
    for status, (count, declared, stored) in expected.items():
        base, row = row_for(baseline, status), row_for(after, status)
        assert row["session_count"] - base["session_count"] == count
        assert row["declared_bytes"] - base["declared_bytes"] == declared
        assert row["stored_bytes"] - base["stored_bytes"] == stored

    # 总计仍由状态行求和得到。
    for key in ("session_count", "declared_bytes", "stored_bytes"):
        assert after["total"][key] == sum(r[key] for r in after["statuses"])

    # 只读汇总不得滑动 TTL 或更新 last_access_at。
    again = client.get(f"/api/v1/sessions/{up_sid}").json()
    assert again["last_access_at"] == before["last_access_at"]
    assert again["expires_at"] == before["expires_at"]
