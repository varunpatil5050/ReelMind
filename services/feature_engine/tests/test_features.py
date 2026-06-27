import pytest
from services.feature_engine.feature_store import FeatureStore, UserFeatures, VideoFeatures

@pytest.mark.asyncio
async def test_synthetic_user_features():
    store = FeatureStore()
    uf = store._synthetic_user_features("u_1")
    assert isinstance(uf, UserFeatures)
    assert uf.user_id == "u_1"
    assert len(uf.to_vector()) > 0
    assert isinstance(uf.to_dict(), dict)

@pytest.mark.asyncio
async def test_synthetic_video_features():
    store = FeatureStore()
    vf = store._synthetic_video_features("v_1")
    assert isinstance(vf, VideoFeatures)
    assert vf.video_id == "v_1"
    assert len(vf.to_vector()) > 0
    assert isinstance(vf.to_dict(), dict)

@pytest.mark.asyncio
async def test_cross_features():
    store = FeatureStore()
    uf = store._synthetic_user_features("u_1")
    vf = store._synthetic_video_features("v_1")
    
    cf = await store.compute_cross_features(uf, vf)
    assert cf.user_id == "u_1"
    assert cf.video_id == "v_1"
    
    vec = cf.to_vector()
    assert len(vec) == 8  # 8 cross features
