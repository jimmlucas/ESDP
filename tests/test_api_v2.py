from fastapi.testclient import TestClient

from api_service import app


client = TestClient(app)


def valid_payload():
    return {
        "sample_id": "test_sample",
        "round": 1,
        "coverage_est": 40.0,
        "expected_genome_size": 5_000_000.0,
        "raw_read_n50": 15_000.0,
        "ai_cov_cv": 0.20,
        "current_n50": 4_800_000.0,
        "current_num_contigs": 2,
        "current_assembly_frac": 0.99,
        "delta_n50_last": None,
        "n50_from_R1": None,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "esdp-api"
    assert data["api_version"] == "2.0.0"
    assert data["model_version"] == "esdp-sequential-rf-v2"


def test_model_info():
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == "esdp-sequential-rf-v2"
    assert data["model_type"] == "RandomForestClassifier"
    assert data["decision_threshold"] == 0.45
    assert data["legal_decision_rounds"] == [1, 2, 3, 4]
    assert data["maximum_racon_round"] == 5
    assert data["feature_count"] == 10

    assert data["model_sha256"] == (
        "9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf"
    )

    assert data["feature_schema_sha256"] == (
        "2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced"
    )


def test_predict_r1():
    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sample_id"] == "test_sample"
    assert data["round"] == 1
    assert data["threshold"] == 0.45

    assert data["decision"] in {
        "STOP",
        "CONTINUE",
    }

    assert 0.0 <= data["p_continue"] <= 1.0

    if data["decision"] == "CONTINUE":
        assert data["next_action"] == "RACON_R2"
    else:
        assert data["next_action"] == "MEDAKA"


def test_predict_matches_frozen_reference_case():
    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert abs(
        data["p_continue"]
        - 0.5939402938271252
    ) < 1e-12

    assert data["decision"] == "CONTINUE"
    assert data["next_action"] == "RACON_R2"


def test_r5_is_rejected():
    payload = valid_payload()

    payload["round"] = 5

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_busco_is_rejected():
    payload = valid_payload()

    payload["busco_complete"] = 98.5

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_qv_is_rejected():
    payload = valid_payload()

    payload["qv"] = 40.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_genus_is_rejected():
    payload = valid_payload()

    payload["genus"] = "Escherichia"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_r4_continue_action_is_valid():
    payload = valid_payload()

    payload.update(
        {
            "sample_id": "test_r4",
            "round": 4,
            "delta_n50_last": 10000.0,
            "n50_from_R1": 50000.0,
        }
    )

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    if data["decision"] == "CONTINUE":
        assert data["next_action"] == "RACON_R5_THEN_MEDAKA"
    else:
        assert data["next_action"] == "MEDAKA"
