"""展厅开放批次合规核验：单事务核验、凭证固化、request_key 幂等与逐件失败原因。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

BASE = 1_700_000_000  # 2023-11-14T22:13:20Z


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    from app.main import app

    with TestClient(app) as c:
        yield c, db_file


def make_exhibit(client: TestClient, eid: str,
                 limit: float = 100000.0, interval: float = 3600.0) -> str:
    resp = client.post("/exhibits", json={
        "id": eid, "name": eid,
        "dose_limit_luxh": limit, "max_interval_seconds": interval,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["canonical_id"]


def set_calendar_24h(client: TestClient, eid: str) -> None:
    """全天 24 小时开放（00:00–00:00 跨午夜到次日同一时刻）。"""
    resp = client.put(
        f"/exhibits/{eid}/calendar",
        json={
            "timezone": "UTC",
            "weekly": [{"weekday": d, "start": "00:00", "end": "00:00"}
                       for d in range(1, 8)],
        },
    )
    assert resp.status_code == 200, resp.text


def send(client: TestClient, exhibit_id: str, readings: list[dict]) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/readings", json={"readings": readings})
    assert resp.status_code == 200, resp.text
    return resp.json()


def steady_readings(n: int = 13, step: float = 1800.0, lux: float = 100.0) -> list[dict]:
    """等间隔平稳读数（间隔 30 分钟 ≤ 3600 秒上限，恰好覆盖 6 小时核验窗）。"""
    return [{"id": f"r{i+1}", "ts": iso(BASE + i * step), "lux": lux} for i in range(n)]


def reconcile(client: TestClient, round_id: str, body: dict,
              expected_status: int = 200) -> dict:
    resp = client.post(f"/inventory-rounds/{round_id}/reconcile", json=body)
    assert resp.status_code == expected_status, resp.text
    return resp.json()


def round_body(expected: list[dict], events: list[dict]) -> dict:
    return {
        "start_at": iso(BASE),
        "window_seconds": 60,
        "rssi_delta_threshold": 5,
        "expected": expected,
        "events": events,
    }


def ev(exhibit_id: str, area: str, dt: float, rssi: float) -> dict:
    return {"exhibit_id": exhibit_id, "area": area, "ts": iso(dt), "rssi": rssi}


def check_body(round_id: str, exhibit_ids: list[str], key: str = "chk-1",
               start: float = BASE, end: float = BASE + 6 * 3600, **kw) -> dict:
    body = {
        "request_key": key,
        "round_id": round_id,
        "start": iso(start),
        "end": iso(end),
        "exhibit_ids": exhibit_ids,
        "max_rate_lux_per_second": 1.0,
        "stuck_tolerance_lux": 2.0,
        "min_stuck_seconds": 86400.0,  # 默认卡死阈值拉到一天，平稳读数不误判 stuck
    }
    body.update(kw)
    return body


def check(client: TestClient, body: dict, expected_status: int = 200) -> dict:
    resp = client.post("/compliance-checks", json=body)
    assert resp.status_code == expected_status, resp.text
    return resp.json()


def compliant_exhibit(client: TestClient, eid: str, area: str = "A区",
                      round_id: str = "R1") -> str:
    """造一件全合规展品：24h 日历 + 平稳读数 + 盘点定位到预期区域。"""
    make_exhibit(client, eid)
    set_calendar_24h(client, eid)
    send(client, eid, steady_readings())
    reconcile(client, round_id, round_body(
        expected=[{"exhibit_id": eid, "area": area}],
        events=[ev(eid, area, BASE + 5, -50)],
    ))
    return eid


# ---------- 合规固化与查询 ----------

def test_compliant_credential_solidified_and_queryable(env):
    client, _ = env
    eid = compliant_exhibit(client, "EX-1")
    out = check(client, check_body("R1", [eid]))
    assert out["status"] == "compliant"
    assert out["already_committed"] is False
    assert out["credential_id"] and len(out["digest"]) == 64
    assert out["created_at"] is not None
    assert out["round_id"] == "R1" and len(out["round_digest"]) == 64

    item, = out["exhibits"]
    assert item["requested_id"] == eid and item["canonical_id"] == eid
    assert item["compliant"] is True and item["reasons"] == []
    assert item["location"] == {
        "status": "located", "expected_area": "A区", "actual_area": "A区",
    }
    # 证据摘要：calendar=display 剂量（100 lux × 6 h = 600 lux·h）、无事件、无未结单
    dose = item["dose"]
    assert dose["exceeded"] is False
    assert dose["known_dose_luxh"] == pytest.approx(600.0)
    assert dose["closed_seconds"] == 0.0
    assert dose["reading_count"] == 13
    assert item["events"] == [] and item["open_ticket_ids"] == []

    # GET 按 request_key 取回同一份凭证
    got = client.get("/compliance-checks/chk-1")
    assert got.status_code == 200
    assert got.json() == out | {"already_committed": False}
    # 未固化过的键 → 404
    assert client.get("/compliance-checks/nope").status_code == 404


def test_replay_same_key_and_normalized_input(env):
    client, _ = env
    for eid, area in (("EX-1", "A区"), ("EX-2", "B区")):
        make_exhibit(client, eid)
        set_calendar_24h(client, eid)
        send(client, eid, steady_readings())
    # 轮次里两件都定位到预期区域
    reconcile(client, "R2", round_body(
        expected=[{"exhibit_id": "EX-1", "area": "A区"},
                  {"exhibit_id": "EX-2", "area": "B区"}],
        events=[ev("EX-1", "A区", BASE + 5, -50), ev("EX-2", "B区", BASE + 5, -50)],
    ))
    first = check(client, check_body("R2", ["EX-1", "EX-2"]))
    assert first["status"] == "compliant"

    # 展品清单顺序不同、时间写法不同（等价时区）→ 规范化输入相同 → 重放
    replay_body = check_body("R2", ["EX-2", "EX-1"])
    replay_body["start"] = "2023-11-14T22:13:20+00:00"
    replay = check(client, replay_body)
    assert replay["already_committed"] is True
    assert replay["credential_id"] == first["credential_id"]
    assert replay["digest"] == first["digest"]
    # 重放返回首次固化的原件（展品顺序保持首次请求的顺序）
    assert [x["canonical_id"] for x in replay["exhibits"]] == ["EX-1", "EX-2"]


def test_same_key_different_input_409(env):
    client, _ = env
    compliant_exhibit(client, "EX-1")
    check(client, check_body("R1", ["EX-1"]))
    # 换时间窗 / 换阈值 / 换轮次 / 换展品清单 → 规范化输入不同 → 409
    compliant_exhibit(client, "EX-2", round_id="R2")
    base = check_body("R1", ["EX-1"])
    variants = [
        {**base, "end": iso(BASE + 8 * 3600)},
        {**base, "max_rate_lux_per_second": 0.5},
        {**base, "round_id": "R2"},
        {**base, "exhibit_ids": ["EX-1", "EX-2"]},
    ]
    for body in variants:
        resp = client.post("/compliance-checks", json=body)
        assert resp.status_code == 409, (body, resp.text)


def test_replay_409_after_evidence_changed(env):
    client, _ = env
    eid = compliant_exhibit(client, "EX-1")
    first = check(client, check_body("R1", [eid]))
    assert first["status"] == "compliant"

    # 证据变化（新上报读数改变剂量摘要）→ 同键重放 409
    send(client, eid, [{"id": "r-extra", "ts": iso(BASE + 13500), "lux": 120}])
    resp = client.post("/compliance-checks", json=check_body("R1", [eid]))
    assert resp.status_code == 409

    # 处置单状态变化同样使摘要不同 → 409
    client2_eid = compliant_exhibit(client, "EX-2", round_id="R2")
    check(client, check_body("R2", [client2_eid], key="chk-2"))
    t = client.post(f"/exhibits/{client2_eid}/illumination-tickets", json={
        "start": iso(BASE), "end": iso(BASE + 3600),
        "max_rate_lux_per_second": 1.0, "stuck_tolerance_lux": 2.0,
        "min_stuck_seconds": 1800, "request_key": "t-open",
    })
    assert t.status_code == 200, t.text
    resp = client.post("/compliance-checks", json=check_body("R2", [client2_eid], key="chk-2"))
    assert resp.status_code == 409


# ---------- 逐件失败原因（不写入） ----------

def test_inventory_location_failures(env):
    client, _ = env
    for eid in ("ok", "gone", "away", "fuzzy", "intruder", "rogue"):
        make_exhibit(client, eid)
        set_calendar_24h(client, eid)
        send(client, eid, steady_readings())
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "ok", "area": "A"},
                  {"exhibit_id": "gone", "area": "A"},
                  {"exhibit_id": "away", "area": "A"},
                  {"exhibit_id": "fuzzy", "area": "A"}],
        events=[
            ev("ok", "A", BASE + 5, -50),
            ev("away", "B", BASE + 5, -50),
            ev("fuzzy", "A", BASE + 5, -50), ev("fuzzy", "B", BASE + 5, -51),
            ev("intruder", "C", BASE + 5, -50),
        ],
    ))
    out = check(client, check_body("R1", ["ok", "gone", "away", "fuzzy",
                                          "intruder", "rogue"]))
    assert out["status"] == "non_compliant"
    assert out["credential_id"] is None and out["digest"] is None
    assert out["created_at"] is None
    by_id = {x["canonical_id"]: x for x in out["exhibits"]}
    assert by_id["ok"]["compliant"] is True
    assert by_id["gone"]["location"]["status"] == "missing"
    assert [r["code"] for r in by_id["gone"]["reasons"]] == ["inventory_missing"]
    assert by_id["away"]["location"]["actual_area"] == "B"
    assert [r["code"] for r in by_id["away"]["reasons"]] == ["inventory_misplaced"]
    assert [r["code"] for r in by_id["fuzzy"]["reasons"]] == ["inventory_unknown_position"]
    assert [r["code"] for r in by_id["intruder"]["reasons"]] == ["inventory_unexpected"]
    assert [r["code"] for r in by_id["rogue"]["reasons"]] == ["inventory_not_covered"]

    # 核验失败不写入：GET 404，同一 request_key 修复后可重新核验
    assert client.get("/compliance-checks/chk-1").status_code == 404
    fixed = check(client, check_body("R1", ["ok"]))
    assert fixed["status"] == "compliant"
    assert client.get("/compliance-checks/chk-1").status_code == 200


def test_calendar_missing_failure(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    send(client, "EX-1", steady_readings())
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "EX-1", "area": "A"}],
        events=[ev("EX-1", "A", BASE + 5, -50)],
    ))
    out = check(client, check_body("R1", ["EX-1"]))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert [r["code"] for r in item["reasons"]] == ["calendar_missing"]
    assert item["dose"] is None


def test_dose_exceeded_failure(env):
    client, _ = env
    make_exhibit(client, "EX-1", limit=100.0)  # 100 lux × 6 h = 600 lux·h ≫ 100
    set_calendar_24h(client, "EX-1")
    send(client, "EX-1", steady_readings())
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "EX-1", "area": "A"}],
        events=[ev("EX-1", "A", BASE + 5, -50)],
    ))
    out = check(client, check_body("R1", ["EX-1"]))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert [r["code"] for r in item["reasons"]] == ["dose_exceeded"]
    assert item["dose"]["exceeded"] is True
    assert item["dose"]["first_exceeded_at"] is not None


@pytest.mark.parametrize("event_type", ["gap", "jump", "stuck"])
def test_diagnosis_event_failures(env, event_type):
    client, _ = env
    make_exhibit(client, "EX-1")
    set_calendar_24h(client, "EX-1")
    if event_type == "gap":
        readings = [{"id": "r1", "ts": iso(BASE), "lux": 100},
                    {"id": "r2", "ts": iso(BASE + 7200), "lux": 100}]  # 间隔 7200 > 3600
        kw = {}
    elif event_type == "jump":
        readings = [{"id": "r1", "ts": iso(BASE), "lux": 100},
                    {"id": "r2", "ts": iso(BASE + 1800), "lux": 500}]  # 速率 400/1800 > 0.1
        kw = {"max_rate_lux_per_second": 0.1}
    else:
        readings = steady_readings()  # 平稳链 + 低卡死阈值
        kw = {"min_stuck_seconds": 1800.0}
    send(client, "EX-1", readings)
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "EX-1", "area": "A"}],
        events=[ev("EX-1", "A", BASE + 5, -50)],
    ))
    out = check(client, check_body("R1", ["EX-1"], **kw))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert [r["code"] for r in item["reasons"]] == ["diagnosis_events"]
    assert event_type in {e["type"] for e in item["events"]}


def test_timestamp_conflict_does_not_block(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    set_calendar_24h(client, "EX-1")
    send(client, "EX-1", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 1800), "lux": 100},
        {"id": "r3", "ts": iso(BASE + 1800), "lux": 500},  # 同时刻异值 → timestamp_conflict
        {"id": "r4", "ts": iso(BASE + 3600), "lux": 100},
    ])
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "EX-1", "area": "A"}],
        events=[ev("EX-1", "A", BASE + 5, -50)],
    ))
    out = check(client, check_body("R1", ["EX-1"]))
    # timestamp_conflict 只记入证据，不阻断合规
    assert out["status"] == "compliant"
    item, = out["exhibits"]
    assert [e["type"] for e in item["events"]] == ["timestamp_conflict"]


def test_open_tickets_failure(env):
    client, _ = env
    eid = compliant_exhibit(client, "EX-1")
    t = client.post(f"/exhibits/{eid}/illumination-tickets", json={
        "start": iso(BASE), "end": iso(BASE + 3600),
        "max_rate_lux_per_second": 1.0, "stuck_tolerance_lux": 2.0,
        "min_stuck_seconds": 1800, "request_key": "t-open",
    })
    assert t.status_code == 200, t.text
    ticket_id = t.json()["ticket_id"]

    out = check(client, check_body("R1", [eid]))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert [r["code"] for r in item["reasons"]] == ["open_tickets"]
    assert item["open_ticket_ids"] == [ticket_id]

    # 结案（accepted）后不再是未结处置单 → 同键可重新核验并固化
    digest = client.get(f"/illumination-tickets/{ticket_id}").json()["current_digest"]
    close = client.post(f"/illumination-tickets/{ticket_id}/close", json={
        "expected_digest": digest, "decision": "accepted", "note": "正常工况",
    })
    assert close.status_code == 200, close.text
    fixed = check(client, check_body("R1", [eid]))
    assert fixed["status"] == "compliant"


def test_multiple_reasons_on_one_exhibit(env):
    client, _ = env
    make_exhibit(client, "EX-1", limit=100.0)  # 剂量必越限
    # 无日历 + 平稳读数成 stuck + 盘点缺失
    send(client, "EX-1", steady_readings())
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "EX-1", "area": "A"}], events=[],
    ))
    out = check(client, check_body("R1", ["EX-1"], min_stuck_seconds=1800.0))
    item, = out["exhibits"]
    codes = [r["code"] for r in item["reasons"]]
    assert codes == ["inventory_missing", "diagnosis_events", "calendar_missing"]


# ---------- 请求校验与别名 ----------

def test_validation_errors(env):
    client, _ = env
    compliant_exhibit(client, "EX-1")
    # 轮次未对账 → 404
    resp = client.post("/compliance-checks", json=check_body("R-nope", ["EX-1"]))
    assert resp.status_code == 404
    # 展品不存在 → 404
    resp = client.post("/compliance-checks", json=check_body("R1", ["EX-1", "nope"]))
    assert resp.status_code == 404
    # 解析后同一规范档案重复 → 422
    resp = client.post("/compliance-checks", json=check_body("R1", ["EX-1", "EX-1"]))
    assert resp.status_code == 422
    # end 不晚于 start → 422
    resp = client.post("/compliance-checks",
                       json=check_body("R1", ["EX-1"], end=BASE))
    assert resp.status_code == 422
    # 失败请求（含 404/422）不占用 request_key
    out = check(client, check_body("R1", ["EX-1"]))
    assert out["status"] == "compliant"


def test_alias_resolution_after_merge(env):
    client, _ = env
    compliant_exhibit(client, "EX-1")
    make_exhibit(client, "DUP-1")
    token = client.post("/merges/preview",
                        json={"source_id": "DUP-1", "target_id": "EX-1"}).json()["token"]
    r = client.post("/merges/commit", json={"token": token, "resolutions": []})
    assert r.status_code == 200, r.text

    out = check(client, check_body("R1", ["DUP-1"]))
    assert out["status"] == "compliant"
    item, = out["exhibits"]
    assert item["requested_id"] == "DUP-1" and item["canonical_id"] == "EX-1"

    # 归并后用规范 ID 重放同键：规范化输入相同 → 重放首次凭证
    replay = check(client, check_body("R1", ["EX-1"]))
    assert replay["already_committed"] is True
    assert replay["exhibits"][0]["requested_id"] == "DUP-1"  # 返回首次固化的原件


# ---------- 归并后盘点证据：按当前别名关系核对该轮固化的全部身份 ----------

def merge(client: TestClient, source: str, target: str) -> None:
    token = client.post("/merges/preview",
                        json={"source_id": source, "target_id": target}).json()["token"]
    r = client.post("/merges/commit", json={"token": token, "resolutions": []})
    assert r.status_code == 200, r.text


def test_merge_source_located_in_round_not_false_not_covered(env):
    """误判一：轮次中来源档案定位正常，归并后用目标 ID 或旧别名核验不得误报
    inventory_not_covered——来源身份的固化定位证据按当前别名关系归属目标。"""
    client, _ = env
    # 目标档案合规配置齐全；来源档案只有读数（归并后随档案搬到目标）
    make_exhibit(client, "TGT-1")
    set_calendar_24h(client, "TGT-1")
    make_exhibit(client, "SRC-1")
    send(client, "SRC-1", steady_readings())
    # 盘点轮次在归并前固化：只有来源身份 SRC-1 定位到 A区
    reconcile(client, "R1", round_body(
        expected=[{"exhibit_id": "SRC-1", "area": "A区"}],
        events=[ev("SRC-1", "A区", BASE + 5, -50)],
    ))
    merge(client, "SRC-1", "TGT-1")

    # 旧盘点结果不改写：轮次固化证据仍以来源身份 SRC-1 存储
    round_out = client.get("/inventory-rounds/R1").json()
    assert [e["canonical_id"] for e in round_out["expectations"]] == ["SRC-1"]

    for key, eid in (("chk-tgt", "TGT-1"), ("chk-alias", "SRC-1")):
        out = check(client, check_body("R1", [eid], key=key))
        assert out["status"] == "compliant", out
        item, = out["exhibits"]
        assert item["requested_id"] == eid and item["canonical_id"] == "TGT-1"
        assert item["location"] == {
            "status": "located", "expected_area": "A区", "actual_area": "A区",
        }
        assert item["reasons"] == []


def test_merge_source_missing_in_round_blocks_credential(env):
    """误判二：同轮目标定位正常而来源缺失，归并后不得签发合规凭证——
    来源身份在该轮的缺失证据同样归属目标，缺失阻断且失败不占用 request_key。"""
    client, _ = env
    make_exhibit(client, "TGT-2")
    set_calendar_24h(client, "TGT-2")
    send(client, "TGT-2", steady_readings())
    make_exhibit(client, "SRC-2")
    # 同一轮：TGT-2 定位正常，SRC-2 列入预期但全程无事件（缺失）
    reconcile(client, "R2", round_body(
        expected=[{"exhibit_id": "TGT-2", "area": "A区"},
                  {"exhibit_id": "SRC-2", "area": "A区"}],
        events=[ev("TGT-2", "A区", BASE + 5, -50)],
    ))
    merge(client, "SRC-2", "TGT-2")

    for eid in ("TGT-2", "SRC-2"):  # 目标 ID 与旧别名同口径
        out = check(client, check_body("R2", [eid]))
        assert out["status"] == "non_compliant"
        item, = out["exhibits"]
        assert item["canonical_id"] == "TGT-2"
        assert item["location"]["status"] == "missing"
        assert [r["code"] for r in item["reasons"]] == ["inventory_missing"]
        assert "SRC-2" in item["reasons"][0]["detail"]

    # 失败不占用 request_key：GET 404，同键重提仍是逐件原因而非 409
    assert client.get("/compliance-checks/chk-1").status_code == 404
    again = check(client, check_body("R2", ["TGT-2"]))
    assert again["status"] == "non_compliant"
    assert client.get("/compliance-checks/chk-1").status_code == 404


def test_merge_multiple_identities_same_area_compliant(env):
    """同区多身份：同一轮中来源与目标身份都明确定位于同一区域，
    归并后核验通过（跨归并证据全部合格且区域一致）。"""
    client, _ = env
    make_exhibit(client, "TGT-3")
    set_calendar_24h(client, "TGT-3")
    send(client, "TGT-3", steady_readings())
    make_exhibit(client, "SRC-3")
    reconcile(client, "R3", round_body(
        expected=[{"exhibit_id": "TGT-3", "area": "A区"},
                  {"exhibit_id": "SRC-3", "area": "A区"}],
        events=[ev("TGT-3", "A区", BASE + 5, -50),
                ev("SRC-3", "A区", BASE + 5, -50)],
    ))
    merge(client, "SRC-3", "TGT-3")

    out = check(client, check_body("R3", ["TGT-3"]))
    assert out["status"] == "compliant"
    item, = out["exhibits"]
    assert item["location"] == {
        "status": "located", "expected_area": "A区", "actual_area": "A区",
    }


def test_merge_multiple_identities_area_conflict_blocks(env):
    """区域冲突：同一轮中多个身份分别明确定位到不同区域，
    归并后核验阻断（inventory_area_conflict），失败不占用 request_key。"""
    client, _ = env
    make_exhibit(client, "TGT-4")
    set_calendar_24h(client, "TGT-4")
    send(client, "TGT-4", steady_readings())
    make_exhibit(client, "SRC-4")
    reconcile(client, "R4", round_body(
        expected=[{"exhibit_id": "TGT-4", "area": "A区"},
                  {"exhibit_id": "SRC-4", "area": "B区"}],
        events=[ev("TGT-4", "A区", BASE + 5, -50),
                ev("SRC-4", "B区", BASE + 5, -50)],
    ))
    merge(client, "SRC-4", "TGT-4")

    out = check(client, check_body("R4", ["TGT-4"]))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert item["location"]["status"] == "area_conflict"
    assert [r["code"] for r in item["reasons"]] == ["inventory_area_conflict"]
    assert "SRC-4" in item["reasons"][0]["detail"]
    assert client.get("/compliance-checks/chk-1").status_code == 404


def test_merge_other_discrepancy_identities_block(env):
    """归并身份在该轮的错位 / 意外出现证据同样阻断凭证（不择一放行）。"""
    client, _ = env
    make_exhibit(client, "TGT-5")
    set_calendar_24h(client, "TGT-5")
    send(client, "TGT-5", steady_readings())
    make_exhibit(client, "SRC-5")
    # TGT-5 定位正常；SRC-5 未列入预期但被读到（意外出现于 C区）
    reconcile(client, "R5", round_body(
        expected=[{"exhibit_id": "TGT-5", "area": "A区"}],
        events=[ev("TGT-5", "A区", BASE + 5, -50),
                ev("SRC-5", "C区", BASE + 5, -50)],
    ))
    merge(client, "SRC-5", "TGT-5")

    out = check(client, check_body("R5", ["TGT-5"]))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert item["location"]["status"] == "unexpected"
    assert [r["code"] for r in item["reasons"]] == ["inventory_unexpected"]


def test_replay_409_after_merge_changes_identity_evidence(env):
    """已签凭证不改写：归并使该轮身份证据变化后，同键重放核对证据摘要 → 409；
    GET 仍返回首次固化的凭证，新键按当前别名关系重新核验。"""
    client, _ = env
    make_exhibit(client, "TGT-6")
    set_calendar_24h(client, "TGT-6")
    send(client, "TGT-6", steady_readings())
    make_exhibit(client, "SRC-6")
    reconcile(client, "R6", round_body(
        expected=[{"exhibit_id": "TGT-6", "area": "A区"},
                  {"exhibit_id": "SRC-6", "area": "A区"}],
        events=[ev("TGT-6", "A区", BASE + 5, -50)],
    ))
    # 归并前核验通过并固化凭证（此时 SRC-6 还不是 TGT-6 的别名）
    first = check(client, check_body("R6", ["TGT-6"]))
    assert first["status"] == "compliant"

    # SRC-6（该轮缺失）归并进 TGT-6：同键重放按当前别名关系重算证据 → 409
    merge(client, "SRC-6", "TGT-6")
    resp = client.post("/compliance-checks", json=check_body("R6", ["TGT-6"]))
    assert resp.status_code == 409

    # 已签凭证不改写：GET 仍取回首次固化的原件
    got = client.get("/compliance-checks/chk-1")
    assert got.status_code == 200
    assert got.json()["digest"] == first["digest"]
    assert got.json()["status"] == "compliant"

    # 换新 request_key 按当前别名关系重新核验 → 缺失身份阻断
    out = check(client, check_body("R6", ["TGT-6"], key="chk-after-merge"))
    assert out["status"] == "non_compliant"
    item, = out["exhibits"]
    assert [r["code"] for r in item["reasons"]] == ["inventory_missing"]


# ---------- 并发 ----------

def test_concurrent_same_request_single_credential(env):
    client, _ = env
    compliant_exhibit(client, "EX-1")
    ids: list[str] = []
    replays: list[bool] = []

    def post(_: int) -> None:
        out = check(client, check_body("R1", ["EX-1"], key="conc"))
        assert out["status"] == "compliant"
        ids.append(out["credential_id"])
        replays.append(out["already_committed"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(8)))
    assert len(set(ids)) == 1
    assert replays.count(False) == 1 and replays.count(True) == 7


def test_concurrent_same_key_different_request_one_wins(env):
    client, _ = env
    compliant_exhibit(client, "EX-1")
    statuses: list[int] = []

    def post(i: int) -> None:
        body = check_body("R1", ["EX-1"], key="conc2",
                          end=BASE + (6 + i) * 3600)  # 每个请求时间窗不同
        statuses.append(client.post("/compliance-checks", json=body).status_code)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(8)))
    # 恰好一个请求固化成功，其余同键不同输入 → 409
    assert sorted(statuses) == [200] + [409] * 7
