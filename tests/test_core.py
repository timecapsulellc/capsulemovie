import pytest
from capsule_movie_core.pipelines import text2video_pipeline

def test_pipeline_initialization():
    """Test that the video generation pipeline can be initialized"""
    pipeline = text2video_pipeline.Text2VideoPipeline()
    assert pipeline is not None

def test_version():
    """Test that version information is available"""
    assert hasattr(text2video_pipeline, "__version__")