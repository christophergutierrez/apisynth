#!/usr/bin/env python3
"""
Generate structured holdout evaluation sets: 30 examples per endpoint, 18 endpoints.

Holdouts are written to data/videoamp/{endpoint}/holdout.jsonl and are kept
completely out of training data. Each set covers:
  - By-ID endpoints: indirect / natural phrasings (Issue 2 failure mode)
  - me / no-param endpoints: short / ambiguous phrasings (Issue 1 failure mode)
  - List endpoints: bare unqualified prompts with params={} (Issue 4 failure mode),
    plus filtered variants and count-specified prompts
  - All: no pageToken values, no hardcoded pageSize for bare list prompts
"""

import json
import random
from pathlib import Path

OUT_BASE = Path("data/videoamp")
random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write(endpoint: str, records: list[dict]) -> None:
    out_dir = OUT_BASE / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "holdout.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  {endpoint}: {len(records)} records → {out_path}")


def rec(question: str, endpoint: str, params: dict) -> dict:
    return {"question": question, "api_call": {"endpoint": endpoint, "params": params}}


# ---------------------------------------------------------------------------
# No-param endpoints
# ---------------------------------------------------------------------------

def gen_me() -> list[dict]:
    endpoint = "GET /v1/me"
    return [
        rec("Get me", endpoint, {}),
        rec("Get me.", endpoint, {}),
        rec("Fetch me", endpoint, {}),
        rec("Show me me", endpoint, {}),
        rec("Who am I?", endpoint, {}),
        rec("my profile", endpoint, {}),
        rec("What's my user profile?", endpoint, {}),
        rec("pull up my user profile", endpoint, {}),
        rec("show my account", endpoint, {}),
        rec("display my account details", endpoint, {}),
        rec("user info", endpoint, {}),
        rec("get user info", endpoint, {}),
        rec("retrieve my user information", endpoint, {}),
        rec("what user am I logged in as?", endpoint, {}),
        rec("show me my current user", endpoint, {}),
        rec("my account details", endpoint, {}),
        rec("look up my profile", endpoint, {}),
        rec("identity", endpoint, {}),
        rec("current user", endpoint, {}),
        rec("who is logged in?", endpoint, {}),
        rec("me info", endpoint, {}),
        rec("I want to see my profile", endpoint, {}),
        rec("I need my user details", endpoint, {}),
        rec("get my info", endpoint, {}),
        rec("show me my info", endpoint, {}),
        rec("fetch my profile", endpoint, {}),
        rec("what is my user id?", endpoint, {}),
        rec("check my profile", endpoint, {}),
        rec("view my account", endpoint, {}),
        rec("load my user", endpoint, {}),
    ]


def gen_metric_types() -> list[dict]:
    endpoint = "GET /external/v1/content/metric-type-compatibility-matrix"
    return [
        rec("metric types", endpoint, {}),
        rec("get metric types", endpoint, {}),
        rec("show metric types", endpoint, {}),
        rec("fetch metric types", endpoint, {}),
        rec("list metric types", endpoint, {}),
        rec("what metric types are available?", endpoint, {}),
        rec("show me metric types", endpoint, {}),
        rec("metric type compatibility matrix", endpoint, {}),
        rec("get the metric type compatibility matrix", endpoint, {}),
        rec("show the metric compatibility matrix", endpoint, {}),
        rec("what metrics are compatible?", endpoint, {}),
        rec("content metric compatibility", endpoint, {}),
        rec("retrieve metric type compatibility", endpoint, {}),
        rec("metric compatibility", endpoint, {}),
        rec("what are the supported metric types?", endpoint, {}),
        rec("display metric type compatibility matrix", endpoint, {}),
        rec("check metric compatibility", endpoint, {}),
        rec("pull up metric types", endpoint, {}),
        rec("load the metric type matrix", endpoint, {}),
        rec("metric matrix", endpoint, {}),
        rec("I want to see metric types", endpoint, {}),
        rec("I need the metric type matrix", endpoint, {}),
        rec("content metrics compatibility", endpoint, {}),
        rec("give me the metric types", endpoint, {}),
        rec("fetch the compatibility matrix", endpoint, {}),
        rec("what metric combinations are valid?", endpoint, {}),
        rec("all metric types", endpoint, {}),
        rec("show all metric types", endpoint, {}),
        rec("metric type list", endpoint, {}),
        rec("get metric compatibility info", endpoint, {}),
    ]


# ---------------------------------------------------------------------------
# By-ID endpoints (path param as key in params dict)
# ---------------------------------------------------------------------------

EPISODE_IDS = [1, 2, 3, 4, 5, 10, 25, 50, 100, 500]
PROGRAM_IDS = [1, 2, 3, 4, 5, 10, 25, 50, 100, 500]
NETWORK_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
AUDIENCE_IDS = [232962]
MEASUREMENT_UUID = "cdd63ab5-2de5-47e4-97dc-7fb86f1600e0"
# content-metric has no confirmed valid UUIDs; use synthetic placeholder UUIDs
CONTENT_METRIC_UUIDS = [
    "00000000-0001-4000-a000-000000000001",
    "00000000-0001-4000-a000-000000000002",
    "00000000-0001-4000-a000-000000000003",
]


def by_id_phrasings(noun: str, id_label: str, id_values: list, endpoint_tpl: str, param_key: str) -> list[dict]:
    """
    Generate 30 by-ID holdout records. Rotates through id_values for variety.
    Covers direct, indirect, and natural phrasings — targeting Issue 2.
    """
    def pick(i: int):
        return id_values[i % len(id_values)]

    templates = [
        ("Get {noun} {id}", lambda i: rec(f"Get {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("Fetch {noun} {id}", lambda i: rec(f"Fetch {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("Show {noun} {id}", lambda i: rec(f"Show {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("I need to see {noun} {id}", lambda i: rec(f"I need to see {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("show me the details for {noun} {id}", lambda i: rec(f"show me the details for {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("look up {noun} {id}", lambda i: rec(f"look up {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("{noun} {id} info", lambda i: rec(f"{noun} {pick(i)} info", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("{noun} {id} details", lambda i: rec(f"{noun} {pick(i)} details", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("find {noun} {id}", lambda i: rec(f"find {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("find {noun} with id {id}", lambda i: rec(f"find {noun} with id {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("show me {noun} {id}", lambda i: rec(f"show me {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("retrieve {noun} {id}", lambda i: rec(f"retrieve {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("pull up {noun} {id}", lambda i: rec(f"pull up {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("display {noun} {id}", lambda i: rec(f"display {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("I want to see {noun} {id}", lambda i: rec(f"I want to see {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("get the {noun} with id {id}", lambda i: rec(f"get the {noun} with id {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("load {noun} {id}", lambda i: rec(f"load {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("check {noun} {id}", lambda i: rec(f"check {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("give me {noun} {id}", lambda i: rec(f"give me {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("{noun} number {id}", lambda i: rec(f"{noun} number {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("what is {noun} {id}?", lambda i: rec(f"what is {noun} {pick(i)}?", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("view {noun} {id}", lambda i: rec(f"view {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("open {noun} {id}", lambda i: rec(f"open {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("show {noun} with id {id}", lambda i: rec(f"show {noun} with id {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("get details for {noun} {id}", lambda i: rec(f"get details for {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("fetch details for {noun} {id}", lambda i: rec(f"fetch details for {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("I need {noun} {id}", lambda i: rec(f"I need {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("give me the details on {noun} {id}", lambda i: rec(f"give me the details on {noun} {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("look up {noun} with id {id}", lambda i: rec(f"look up {noun} with id {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
        ("fetch {noun} by id {id}", lambda i: rec(f"fetch {noun} by id {pick(i)}", endpoint_tpl.format(id=pick(i)), {param_key: pick(i)})),
    ]

    return [fn(i) for i, (_, fn) in enumerate(templates)]


def gen_episode() -> list[dict]:
    return by_id_phrasings(
        "episode", "episodeId", EPISODE_IDS,
        "GET /external/v1/content/episodes/{id}", "episodeId",
    )


def gen_program() -> list[dict]:
    return by_id_phrasings(
        "program", "programId", PROGRAM_IDS,
        "GET /external/v1/content/programs/{id}", "programId",
    )


def gen_network_by_id() -> list[dict]:
    return by_id_phrasings(
        "network", "id", NETWORK_IDS,
        "GET /external/v1/content/networks/{id}", "id",
    )


def gen_audience_by_id() -> list[dict]:
    return by_id_phrasings(
        "audience", "id", AUDIENCE_IDS,
        "GET /v1/audiences/{id}", "id",
    )


def gen_measurement_by_id() -> list[dict]:
    # measurement uses a UUID, not an integer
    uid = MEASUREMENT_UUID
    endpoint = f"GET /external/v1/measurements/{uid}"
    nouns = "measurement"
    return [
        rec(f"Get measurement {uid}", endpoint, {"id": uid}),
        rec(f"Fetch measurement {uid}", endpoint, {"id": uid}),
        rec(f"Show measurement {uid}", endpoint, {"id": uid}),
        rec(f"I need to see measurement {uid}", endpoint, {"id": uid}),
        rec(f"show me the details for measurement {uid}", endpoint, {"id": uid}),
        rec(f"look up measurement {uid}", endpoint, {"id": uid}),
        rec(f"measurement {uid} info", endpoint, {"id": uid}),
        rec(f"measurement {uid} details", endpoint, {"id": uid}),
        rec(f"find measurement {uid}", endpoint, {"id": uid}),
        rec(f"find measurement with id {uid}", endpoint, {"id": uid}),
        rec(f"show me measurement {uid}", endpoint, {"id": uid}),
        rec(f"retrieve measurement {uid}", endpoint, {"id": uid}),
        rec(f"pull up measurement {uid}", endpoint, {"id": uid}),
        rec(f"display measurement {uid}", endpoint, {"id": uid}),
        rec(f"I want to see measurement {uid}", endpoint, {"id": uid}),
        rec(f"get the measurement with id {uid}", endpoint, {"id": uid}),
        rec(f"load measurement {uid}", endpoint, {"id": uid}),
        rec(f"check measurement {uid}", endpoint, {"id": uid}),
        rec(f"give me measurement {uid}", endpoint, {"id": uid}),
        rec(f"what is measurement {uid}?", endpoint, {"id": uid}),
        rec(f"view measurement {uid}", endpoint, {"id": uid}),
        rec(f"open measurement {uid}", endpoint, {"id": uid}),
        rec(f"show measurement with id {uid}", endpoint, {"id": uid}),
        rec(f"get details for measurement {uid}", endpoint, {"id": uid}),
        rec(f"fetch details for measurement {uid}", endpoint, {"id": uid}),
        rec(f"I need measurement {uid}", endpoint, {"id": uid}),
        rec(f"give me the details on measurement {uid}", endpoint, {"id": uid}),
        rec(f"look up measurement with id {uid}", endpoint, {"id": uid}),
        rec(f"fetch measurement by id {uid}", endpoint, {"id": uid}),
        rec(f"get measurement report {uid}", endpoint, {"id": uid}),
    ]


def gen_content_metric() -> list[dict]:
    # No confirmed valid UUIDs — use synthetic placeholder UUIDs
    uids = CONTENT_METRIC_UUIDS
    def pick(i): return uids[i % len(uids)]
    endpoint_tpl = "GET /external/v1/content/metrics/{id}"
    return [
        rec(f"Get content metric {pick(i)}", endpoint_tpl.format(id=pick(i)), {"id": pick(i)})
        if i % 3 == 0 else
        rec(f"show me content metric {pick(i)}", endpoint_tpl.format(id=pick(i)), {"id": pick(i)})
        if i % 3 == 1 else
        rec(f"look up content metric {pick(i)}", endpoint_tpl.format(id=pick(i)), {"id": pick(i)})
        for i in range(30)
    ] if False else [
        rec(f"Get content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"Fetch content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"Show content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"I need to see content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"show me the details for content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"look up content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"content metric {pick(0)} info", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"content metric {pick(1)} details", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"find content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"find content metric with id {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"show me content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"retrieve content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"pull up content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"display content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"I want to see content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"get the content metric with id {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"load content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"check content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"give me content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"content metric number {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"what is content metric {pick(2)}?", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"view content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"open content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"show content metric with id {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"get details for content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"fetch details for content metric {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"I need content metric {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
        rec(f"give me the details on content metric {pick(0)}", endpoint_tpl.format(id=pick(0)), {"id": pick(0)}),
        rec(f"look up content metric with id {pick(1)}", endpoint_tpl.format(id=pick(1)), {"id": pick(1)}),
        rec(f"fetch content metric by id {pick(2)}", endpoint_tpl.format(id=pick(2)), {"id": pick(2)}),
    ]


def gen_audience_exports() -> list[dict]:
    """audience-exports: GET /v1/audiences/{audienceId}/exports — audienceId required, optional list params"""
    aid = 232962
    endpoint_tpl = f"GET /v1/audiences/{aid}/exports"
    return [
        # bare: no extra params
        rec(f"get exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"show exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"fetch exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"list exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"audience {aid} exports", endpoint_tpl, {"audienceId": aid}),
        rec(f"what exports does audience {aid} have?", endpoint_tpl, {"audienceId": aid}),
        rec(f"show me the exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"retrieve exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"I need exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"look up exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"display exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"pull up exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"view exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"I want to see exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"give me exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        # with pageSize
        rec(f"get 10 exports for audience {aid}", endpoint_tpl, {"audienceId": aid, "pageSize": 10}),
        rec(f"show 5 exports for audience {aid}", endpoint_tpl, {"audienceId": aid, "pageSize": 5}),
        rec(f"list 20 exports for audience {aid}", endpoint_tpl, {"audienceId": aid, "pageSize": 20}),
        # with sorts
        rec(f"get exports for audience {aid} sorted by created_at descending", endpoint_tpl, {"audienceId": aid, "sorts": ["-created_at"]}),
        rec(f"show exports for audience {aid} sorted by status", endpoint_tpl, {"audienceId": aid, "sorts": ["+status"]}),
        rec(f"list exports for audience {aid} ordered by created_at", endpoint_tpl, {"audienceId": aid, "sorts": ["-created_at"]}),
        # combined
        rec(f"get 10 exports for audience {aid} sorted by created_at", endpoint_tpl, {"audienceId": aid, "pageSize": 10, "sorts": ["-created_at"]}),
        rec(f"show 5 exports for audience {aid} sorted by status descending", endpoint_tpl, {"audienceId": aid, "pageSize": 5, "sorts": ["-status"]}),
        rec(f"fetch 15 exports for audience {aid} sorted ascending by status", endpoint_tpl, {"audienceId": aid, "pageSize": 15, "sorts": ["+status"]}),
        rec(f"get 20 exports for audience {aid} ordered by created_at ascending", endpoint_tpl, {"audienceId": aid, "pageSize": 20, "sorts": ["+created_at"]}),
        # indirect phrasings
        rec(f"I need to see exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"find all exports associated with audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"check exports on audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"load the export list for audience {aid}", endpoint_tpl, {"audienceId": aid}),
        rec(f"open exports for audience {aid}", endpoint_tpl, {"audienceId": aid}),
    ]


# ---------------------------------------------------------------------------
# List endpoints
# ---------------------------------------------------------------------------

def gen_episodes_list() -> list[dict]:
    endpoint = "GET /external/v1/content/episodes"
    net_ids = [1, 2, 4, 5, 6]
    prog_ids = [1, 2, 3, 4, 5]
    cors = [23, 25, 26, 27]
    return [
        # bare list — no params (Issue 4: model must NOT add pageSize)
        rec("list all episodes", endpoint, {}),
        rec("show all episodes", endpoint, {}),
        rec("get all episodes", endpoint, {}),
        rec("fetch all episodes", endpoint, {}),
        rec("episodes", endpoint, {}),
        rec("show me all episodes", endpoint, {}),
        rec("what episodes are there?", endpoint, {}),
        rec("display all episodes", endpoint, {}),
        # count-specified list — pageSize should be set
        rec("get 10 episodes", endpoint, {"pageSize": 10}),
        rec("show 5 episodes", endpoint, {"pageSize": 5}),
        rec("list 20 episodes", endpoint, {"pageSize": 20}),
        # filtered by networkId
        rec(f"get episodes from network {net_ids[0]}", endpoint, {"networkId": net_ids[0]}),
        rec(f"list episodes for network {net_ids[1]}", endpoint, {"networkId": net_ids[1]}),
        rec(f"show episodes on network {net_ids[2]}", endpoint, {"networkId": net_ids[2]}),
        # filtered by programId
        rec(f"get episodes for program {prog_ids[0]}", endpoint, {"programId": prog_ids[0]}),
        rec(f"list episodes in program {prog_ids[1]}", endpoint, {"programId": prog_ids[1]}),
        rec(f"show episodes belonging to program {prog_ids[2]}", endpoint, {"programId": prog_ids[2]}),
        # filtered by currencyOfRecord
        rec(f"get episodes with currency of record {cors[0]}", endpoint, {"currencyOfRecord": cors[0]}),
        rec(f"list episodes for CoR {cors[1]}", endpoint, {"currencyOfRecord": cors[1]}),
        # combined filters
        rec(f"get 10 episodes from network {net_ids[0]} for program {prog_ids[0]}", endpoint, {"pageSize": 10, "networkId": net_ids[0], "programId": prog_ids[0]}),
        rec(f"list 5 episodes on network {net_ids[1]} with CoR {cors[0]}", endpoint, {"pageSize": 5, "networkId": net_ids[1], "currencyOfRecord": cors[0]}),
        rec(f"show 20 episodes for program {prog_ids[2]} with currency {cors[1]}", endpoint, {"pageSize": 20, "programId": prog_ids[2], "currencyOfRecord": cors[1]}),
        rec(f"get episodes from network {net_ids[3]} program {prog_ids[3]} CoR {cors[2]}", endpoint, {"networkId": net_ids[3], "programId": prog_ids[3], "currencyOfRecord": cors[2]}),
        rec(f"10 episodes network {net_ids[4]} program {prog_ids[4]} CoR {cors[3]}", endpoint, {"pageSize": 10, "networkId": net_ids[4], "programId": prog_ids[4], "currencyOfRecord": cors[3]}),
        # natural indirect
        rec("I need to see all episodes", endpoint, {}),
        rec("load all episodes", endpoint, {}),
        rec(f"pull up episodes from network {net_ids[0]}", endpoint, {"networkId": net_ids[0]}),
        rec(f"I want episodes for program {prog_ids[1]}", endpoint, {"programId": prog_ids[1]}),
        rec(f"give me episodes with currency of record {cors[2]}", endpoint, {"currencyOfRecord": cors[2]}),
        rec(f"fetch 15 episodes from network {net_ids[2]} with CoR {cors[0]}", endpoint, {"pageSize": 15, "networkId": net_ids[2], "currencyOfRecord": cors[0]}),
    ]


def gen_programs_list() -> list[dict]:
    endpoint = "GET /external/v1/content/programs"
    net_ids = [1, 2, 4, 5, 6]
    cors = [23, 25, 26, 27]
    return [
        rec("list all programs", endpoint, {}),
        rec("show all programs", endpoint, {}),
        rec("get all programs", endpoint, {}),
        rec("fetch all programs", endpoint, {}),
        rec("programs", endpoint, {}),
        rec("show me all programs", endpoint, {}),
        rec("what programs are there?", endpoint, {}),
        rec("display all programs", endpoint, {}),
        rec("get 10 programs", endpoint, {"pageSize": 10}),
        rec("show 5 programs", endpoint, {"pageSize": 5}),
        rec("list 20 programs", endpoint, {"pageSize": 20}),
        rec("get 50 programs", endpoint, {"pageSize": 50}),
        rec(f"get programs from network {net_ids[0]}", endpoint, {"networkId": net_ids[0]}),
        rec(f"list programs for network {net_ids[1]}", endpoint, {"networkId": net_ids[1]}),
        rec(f"show programs on network {net_ids[2]}", endpoint, {"networkId": net_ids[2]}),
        rec(f"get programs with currency of record {cors[0]}", endpoint, {"currencyOfRecord": cors[0]}),
        rec(f"list programs for CoR {cors[1]}", endpoint, {"currencyOfRecord": cors[1]}),
        rec("find programs named Breaking Bad", endpoint, {"name": "Breaking Bad"}),
        rec("search programs by name The Crown", endpoint, {"name": "The Crown"}),
        rec("programs with name Survivor", endpoint, {"name": "Survivor"}),
        rec(f"get 10 programs from network {net_ids[0]} with CoR {cors[0]}", endpoint, {"pageSize": 10, "networkId": net_ids[0], "currencyOfRecord": cors[0]}),
        rec(f"list 5 programs on network {net_ids[1]} currency {cors[1]}", endpoint, {"pageSize": 5, "networkId": net_ids[1], "currencyOfRecord": cors[1]}),
        rec(f"programs from network {net_ids[2]} with CoR {cors[2]}", endpoint, {"networkId": net_ids[2], "currencyOfRecord": cors[2]}),
        rec(f"20 programs network {net_ids[3]}", endpoint, {"pageSize": 20, "networkId": net_ids[3]}),
        rec("I need to see all programs", endpoint, {}),
        rec("load all programs", endpoint, {}),
        rec(f"pull up programs from network {net_ids[0]}", endpoint, {"networkId": net_ids[0]}),
        rec(f"I want programs for network {net_ids[1]}", endpoint, {"networkId": net_ids[1]}),
        rec(f"give me programs with currency of record {cors[2]}", endpoint, {"currencyOfRecord": cors[2]}),
        rec(f"fetch 15 programs from network {net_ids[2]} with CoR {cors[0]}", endpoint, {"pageSize": 15, "networkId": net_ids[2], "currencyOfRecord": cors[0]}),
    ]


def gen_networks_list() -> list[dict]:
    endpoint = "GET /external/v1/content/networks"
    cors = [23, 25, 26, 27]
    return [
        rec("list all networks", endpoint, {}),
        rec("show all networks", endpoint, {}),
        rec("get all networks", endpoint, {}),
        rec("fetch all networks", endpoint, {}),
        rec("networks", endpoint, {}),
        rec("show me all networks", endpoint, {}),
        rec("what networks are available?", endpoint, {}),
        rec("display all networks", endpoint, {}),
        rec("get 10 networks", endpoint, {"pageSize": 10}),
        rec("show 5 networks", endpoint, {"pageSize": 5}),
        rec("list 20 networks", endpoint, {"pageSize": 20}),
        rec("get 50 networks", endpoint, {"pageSize": 50}),
        rec(f"get networks with currency of record {cors[0]}", endpoint, {"currencyOfRecord": cors[0]}),
        rec(f"list networks for CoR {cors[1]}", endpoint, {"currencyOfRecord": cors[1]}),
        rec(f"networks with CoR {cors[2]}", endpoint, {"currencyOfRecord": cors[2]}),
        rec("find networks named NBC", endpoint, {"name": "NBC"}),
        rec("search networks by name CBS", endpoint, {"name": "CBS"}),
        rec("networks named ESPN", endpoint, {"name": "ESPN"}),
        rec("networks with media group name Warner", endpoint, {"mediaGroupName": "Warner"}),
        rec("list networks in media group Comcast", endpoint, {"mediaGroupName": "Comcast"}),
        rec(f"get 10 networks with CoR {cors[0]}", endpoint, {"pageSize": 10, "currencyOfRecord": cors[0]}),
        rec(f"list 5 networks currency {cors[1]}", endpoint, {"pageSize": 5, "currencyOfRecord": cors[1]}),
        rec(f"20 networks CoR {cors[2]}", endpoint, {"pageSize": 20, "currencyOfRecord": cors[2]}),
        rec("networks named Fox with CoR 25", endpoint, {"name": "Fox", "currencyOfRecord": 25}),
        rec("I need to see all networks", endpoint, {}),
        rec("load all networks", endpoint, {}),
        rec("pull up all networks", endpoint, {}),
        rec(f"I want networks with currency of record {cors[0]}", endpoint, {"currencyOfRecord": cors[0]}),
        rec(f"give me networks with CoR {cors[2]}", endpoint, {"currencyOfRecord": cors[2]}),
        rec(f"fetch 15 networks with CoR {cors[0]}", endpoint, {"pageSize": 15, "currencyOfRecord": cors[0]}),
    ]


def gen_audiences_list() -> list[dict]:
    endpoint = "GET /v1/audiences"
    return [
        rec("list all audiences", endpoint, {}),
        rec("show all audiences", endpoint, {}),
        rec("get all audiences", endpoint, {}),
        rec("fetch all audiences", endpoint, {}),
        rec("audiences", endpoint, {}),
        rec("show me all audiences", endpoint, {}),
        rec("what audiences are available?", endpoint, {}),
        rec("display all audiences", endpoint, {}),
        rec("get 10 audiences", endpoint, {"pageSize": 10}),
        rec("show 5 audiences", endpoint, {"pageSize": 5}),
        rec("list 20 audiences", endpoint, {"pageSize": 20}),
        rec("get ready audiences", endpoint, {"status": "ready"}),
        rec("show failed audiences", endpoint, {"status": "failed"}),
        rec("list processing audiences", endpoint, {"status": "processing"}),
        rec("get advanced audiences", endpoint, {"type": "advanced"}),
        rec("show demographic audiences", endpoint, {"type": "demo"}),
        rec("list shared audiences", endpoint, {"type": "shared"}),
        rec("get owned audiences", endpoint, {"type": "owned"}),
        rec("list household-level audiences", endpoint, {"level": "HOUSEHOLD"}),
        rec("show person-level audiences", endpoint, {"level": "PERSON"}),
        rec("audiences for measurement", endpoint, {"useCases": "measurement"}),
        rec("audiences for activation", endpoint, {"useCases": "activation"}),
        rec("get 10 ready audiences", endpoint, {"pageSize": 10, "status": "ready"}),
        rec("list 5 advanced household audiences", endpoint, {"pageSize": 5, "type": "advanced", "level": "HOUSEHOLD"}),
        rec("find audiences named sports fans", endpoint, {"query": "sports fans"}),
        rec("search audiences for 18-34", endpoint, {"query": "18-34"}),
        rec("I need to see all audiences", endpoint, {}),
        rec("load all audiences", endpoint, {}),
        rec("pull up all audiences", endpoint, {}),
        rec("give me ready advanced audiences for measurement", endpoint, {"status": "ready", "type": "advanced", "useCases": "measurement"}),
    ]


def gen_measurements_list() -> list[dict]:
    endpoint = "GET /external/v1/measurements"
    cors = [23, 25, 26, 27]
    return [
        rec("list all measurements", endpoint, {}),
        rec("show all measurements", endpoint, {}),
        rec("get all measurements", endpoint, {}),
        rec("fetch all measurements", endpoint, {}),
        rec("measurements", endpoint, {}),
        rec("show me all measurements", endpoint, {}),
        rec("what measurements are available?", endpoint, {}),
        rec("display all measurements", endpoint, {}),
        rec("get 10 measurements", endpoint, {"pageSize": 10}),
        rec("show 5 measurements", endpoint, {"pageSize": 5}),
        rec("list 20 measurements", endpoint, {"pageSize": 20}),
        rec(f"get measurements with CoR {cors[0]}", endpoint, {"currencyOfRecord": cors[0]}),
        rec(f"list measurements for currency {cors[1]}", endpoint, {"currencyOfRecord": cors[1]}),
        rec("get ready measurements", endpoint, {"status": "ready"}),
        rec("show failed measurements", endpoint, {"status": "failed"}),
        rec("list processing measurements", endpoint, {"status": "processing"}),
        rec("find measurements named Q4 campaign", endpoint, {"name": "Q4 campaign"}),
        rec("search measurements by name brand study", endpoint, {"name": "brand study"}),
        rec("measurements created on 2024-01-15", endpoint, {"createdAt": "2024-01-15"}),
        rec("measurements created on 2025-03-01", endpoint, {"createdAt": "2025-03-01"}),
        rec("get measurements sorted by created date descending", endpoint, {"orderBy": "-createdAt"}),
        rec("list measurements ordered by name", endpoint, {"orderBy": "+name"}),
        rec(f"get 10 measurements with CoR {cors[0]}", endpoint, {"pageSize": 10, "currencyOfRecord": cors[0]}),
        rec(f"list 5 ready measurements for currency {cors[1]}", endpoint, {"pageSize": 5, "status": "ready", "currencyOfRecord": cors[1]}),
        rec("I need to see all measurements", endpoint, {}),
        rec("load all measurements", endpoint, {}),
        rec("pull up all measurements", endpoint, {}),
        rec("give me all ready measurements", endpoint, {"status": "ready"}),
        rec(f"fetch 20 measurements for CoR {cors[2]}", endpoint, {"pageSize": 20, "currencyOfRecord": cors[2]}),
        rec("show measurements sorted by status ascending", endpoint, {"orderBy": "+status"}),
    ]


def gen_media_groups() -> list[dict]:
    endpoint = "GET /external/v1/content/media-groups"
    return [
        rec("list all media groups", endpoint, {}),
        rec("show all media groups", endpoint, {}),
        rec("get all media groups", endpoint, {}),
        rec("fetch all media groups", endpoint, {}),
        rec("media groups", endpoint, {}),
        rec("show me all media groups", endpoint, {}),
        rec("what media groups are there?", endpoint, {}),
        rec("display all media groups", endpoint, {}),
        rec("get 10 media groups", endpoint, {"pageSize": 10}),
        rec("show 5 media groups", endpoint, {"pageSize": 5}),
        rec("list 20 media groups", endpoint, {"pageSize": 20}),
        rec("find media groups named Warner", endpoint, {"name": "Warner"}),
        rec("search media groups by name Comcast", endpoint, {"name": "Comcast"}),
        rec("media groups named Disney", endpoint, {"name": "Disney"}),
        rec("media group Fox", endpoint, {"name": "Fox"}),
        rec("I need to see all media groups", endpoint, {}),
        rec("load all media groups", endpoint, {}),
        rec("pull up all media groups", endpoint, {}),
        rec("I want to see media groups", endpoint, {}),
        rec("give me all media groups", endpoint, {}),
        rec("media groups list", endpoint, {}),
        rec("get 15 media groups", endpoint, {"pageSize": 15}),
        rec("show 25 media groups", endpoint, {"pageSize": 25}),
        rec("list 30 media groups", endpoint, {"pageSize": 30}),
        rec("fetch 10 media groups named CBS", endpoint, {"pageSize": 10, "name": "CBS"}),
        rec("find 5 media groups named NBC", endpoint, {"pageSize": 5, "name": "NBC"}),
        rec("show media groups named ABC", endpoint, {"name": "ABC"}),
        rec("I need media groups", endpoint, {}),
        rec("view all media groups", endpoint, {}),
        rec("open media groups", endpoint, {}),
    ]


def gen_currency_of_record() -> list[dict]:
    # currency-of-record has NO pageSize param — critical test of Issue 4
    endpoint = "GET /external/v1/currency-of-record"
    return [
        rec("get currency of record", endpoint, {}),
        rec("show currency of record", endpoint, {}),
        rec("fetch currency of record", endpoint, {}),
        rec("list currency of record", endpoint, {}),
        rec("currency of record", endpoint, {}),
        rec("show me currency of record", endpoint, {}),
        rec("what currencies of record are available?", endpoint, {}),
        rec("display currency of record", endpoint, {}),
        rec("CoR", endpoint, {}),
        rec("list all CoR values", endpoint, {}),
        rec("get CoR", endpoint, {}),
        rec("show CoR", endpoint, {}),
        rec("fetch CoR", endpoint, {}),
        rec("currency of record options", endpoint, {}),
        rec("what CoR values exist?", endpoint, {}),
        rec("available currency of record values", endpoint, {}),
        rec("I need currency of record", endpoint, {}),
        rec("I want to see currency of record", endpoint, {}),
        rec("load currency of record", endpoint, {}),
        rec("pull up currency of record", endpoint, {}),
        rec("find currency of record for AD_MEASUREMENT", endpoint, {"reportingScope": "AD_MEASUREMENT"}),
        rec("currency of record for content measurement", endpoint, {"reportingScope": "CONTENT_MEASUREMENT"}),
        rec("get CoR for ad measurement", endpoint, {"reportingScope": "AD_MEASUREMENT"}),
        rec("get CoR for content measurement", endpoint, {"reportingScope": "CONTENT_MEASUREMENT"}),
        rec("list CoR for AD_MEASUREMENT", endpoint, {"reportingScope": "AD_MEASUREMENT"}),
        rec("show CoR values named 2024-25", endpoint, {"name": "2024-25"}),
        rec("find CoR named 2023-24", endpoint, {"name": "2023-24"}),
        rec("currency of record named 2025-26", endpoint, {"name": "2025-26"}),
        rec("all currency of record for reporting scope CONTENT_MEASUREMENT", endpoint, {"reportingScope": "CONTENT_MEASUREMENT"}),
        rec("CoR for AD_MEASUREMENT named 2024-25", endpoint, {"reportingScope": "AD_MEASUREMENT", "name": "2024-25"}),
    ]


def gen_audience_statuses() -> list[dict]:
    endpoint = "GET /v1/audiences/status"
    return [
        rec("show audience statuses", endpoint, {}),
        rec("get audience statuses", endpoint, {}),
        rec("list audience statuses", endpoint, {}),
        rec("fetch audience statuses", endpoint, {}),
        rec("audience statuses", endpoint, {}),
        rec("show me audience statuses", endpoint, {}),
        rec("what are the audience statuses?", endpoint, {}),
        rec("display audience statuses", endpoint, {}),
        rec("get 10 audience statuses", endpoint, {"pageSize": 10}),
        rec("show 5 audience statuses", endpoint, {"pageSize": 5}),
        rec("list 20 audience statuses", endpoint, {"pageSize": 20}),
        rec("get ready audience statuses", endpoint, {"status": "ready"}),
        rec("show failed audience statuses", endpoint, {"status": "failed"}),
        rec("list processing audience statuses", endpoint, {"status": "processing"}),
        rec("get draft audience statuses", endpoint, {"status": "draft"}),
        rec("audience statuses sorted by created date descending", endpoint, {"orderBy": "desc(createdAt)"}),
        rec("show audience statuses ordered by name ascending", endpoint, {"orderBy": "asc(name)"}),
        rec("audience statuses ordered by created at ascending", endpoint, {"orderBy": "asc(createdAt)"}),
        rec("list audience statuses sorted by name descending", endpoint, {"orderBy": "desc(name)"}),
        rec("get 10 ready audience statuses", endpoint, {"pageSize": 10, "status": "ready"}),
        rec("list 5 failed audience statuses", endpoint, {"pageSize": 5, "status": "failed"}),
        rec("20 audience statuses sorted by created date", endpoint, {"pageSize": 20, "orderBy": "desc(createdAt)"}),
        rec("I need to see audience statuses", endpoint, {}),
        rec("load all audience statuses", endpoint, {}),
        rec("pull up audience statuses", endpoint, {}),
        rec("I want ready audience statuses", endpoint, {"status": "ready"}),
        rec("give me audience statuses", endpoint, {}),
        rec("10 ready audience statuses sorted by name", endpoint, {"pageSize": 10, "status": "ready", "orderBy": "asc(name)"}),
        rec("fetch 15 processing audience statuses", endpoint, {"pageSize": 15, "status": "processing"}),
        rec("all failed audience statuses sorted by created at descending", endpoint, {"status": "failed", "orderBy": "desc(createdAt)"}),
    ]


def gen_consents() -> list[dict]:
    endpoint = "GET /v1/consents"
    return [
        rec("list all consents", endpoint, {}),
        rec("show all consents", endpoint, {}),
        rec("get all consents", endpoint, {}),
        rec("fetch all consents", endpoint, {}),
        rec("consents", endpoint, {}),
        rec("show me all consents", endpoint, {}),
        rec("what consents are available?", endpoint, {}),
        rec("display all consents", endpoint, {}),
        rec("get 10 consents", endpoint, {"pageSize": 10}),
        rec("show 5 consents", endpoint, {"pageSize": 5}),
        rec("list 20 consents", endpoint, {"pageSize": 20}),
        rec("get consents for advertisers", endpoint, {"q": "recipient_kind eq ADVERTISER"}),
        rec("show consents for ad agencies", endpoint, {"q": "recipient_kind eq AD_AGENCY"}),
        rec("list consents for organizations", endpoint, {"q": "recipient_kind eq ORGANIZATION"}),
        rec("get consents for brands", endpoint, {"q": "recipient_kind eq BRAND"}),
        rec("consents where recipient name starts with Acme", endpoint, {"q": "recipient_name startswith Acme"}),
        rec("consents where recipient name contains Media", endpoint, {"q": "recipient_name contains Media"}),
        rec("consents where recipient name ends with Corp", endpoint, {"q": "recipient_name endswith Corp"}),
        rec("10 consents for advertisers", endpoint, {"pageSize": 10, "q": "recipient_kind eq ADVERTISER"}),
        rec("list 5 consents for ad agencies", endpoint, {"pageSize": 5, "q": "recipient_kind eq AD_AGENCY"}),
        rec("consents for advertiser or ad agency", endpoint, {"q": "recipient_kind in ADVERTISER,AD_AGENCY"}),
        rec("consents with recipient ancestor path", endpoint, {"fetchRecipientAncestorPath": True}),
        rec("get consents with ancestor path info", endpoint, {"fetchRecipientAncestorPath": True}),
        rec("consents for advertiser named Acme with ancestor path", endpoint, {"q": "recipient_kind eq ADVERTISER", "fetchRecipientAncestorPath": True}),
        rec("I need to see all consents", endpoint, {}),
        rec("load all consents", endpoint, {}),
        rec("pull up all consents", endpoint, {}),
        rec("I want consents for advertisers", endpoint, {"q": "recipient_kind eq ADVERTISER"}),
        rec("give me consents", endpoint, {}),
        rec("fetch 10 consents for brands", endpoint, {"pageSize": 10, "q": "recipient_kind eq BRAND"}),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    ("me", gen_me),
    ("metric-types", gen_metric_types),
    ("episode", gen_episode),
    ("program", gen_program),
    ("network", gen_network_by_id),
    ("audience", gen_audience_by_id),
    ("measurement", gen_measurement_by_id),
    ("content-metric", gen_content_metric),
    ("audience-exports", gen_audience_exports),
    ("episodes", gen_episodes_list),
    ("programs", gen_programs_list),
    ("networks", gen_networks_list),
    ("audiences", gen_audiences_list),
    ("measurements", gen_measurements_list),
    ("media-groups", gen_media_groups),
    ("currency-of-record", gen_currency_of_record),
    ("audience-statuses", gen_audience_statuses),
    ("consents", gen_consents),
]


def main():
    total = 0
    for name, fn in GENERATORS:
        records = fn()
        assert len(records) == 30, f"{name}: expected 30 records, got {len(records)}"
        write(name, records)
        total += len(records)
    print(f"\nTotal: {total} holdout records across {len(GENERATORS)} endpoints")


if __name__ == "__main__":
    main()
