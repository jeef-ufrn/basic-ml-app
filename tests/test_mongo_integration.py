import os
import time
import pytest
import importlib

@pytest.mark.integration
def test_mongo_basic_ops(monkeypatch):

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    if not mongo_uri or not mongo_db:
        pytest.skip("As variáveis MONGO_URI ou MONGO_DB não foram configuradas - Ignorando testes de integração com o MongoDB")

    import db.engine as engine
    importlib.reload(engine)

    coll_name = f"integration_test_{int(time.time())}"
    coll = engine.get_mongo_collection(coll_name)

    try:
        coll.drop()
    except Exception:
        pass

    doc = {"text": "hello integration", "owner": "pytest_mongo_integration"}
    insert_res = coll.insert_one(doc)
    assert insert_res.inserted_id is not None

    fetched = coll.find_one({"_id": insert_res.inserted_id})
    assert fetched is not None
    assert fetched["text"] == "hello integration"
    assert fetched["owner"] == "pytest_mongo_integration"

    coll.update_one({"_id": insert_res.inserted_id}, {"$set": {"text": "updated"}})
    fetched2 = coll.find_one({"_id": insert_res.inserted_id})
    assert fetched2["text"] == "updated"

    coll.delete_one({"_id": insert_res.inserted_id})
    assert coll.find_one({"_id": insert_res.inserted_id}) is None

    coll.drop()

    assert True

