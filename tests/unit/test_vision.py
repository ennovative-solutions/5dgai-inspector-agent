import os
from unittest.mock import MagicMock, patch, mock_open
import pytest

from inspector_agent_app.app_utils.vision import analyze_damage_image

@patch("inspector_agent_app.app_utils.vision.vertexai.init")
@patch("inspector_agent_app.app_utils.vision.Part.from_data")
@patch("inspector_agent_app.app_utils.vision.Part.from_uri")
@patch("inspector_agent_app.app_utils.vision.GenerativeModel")
def test_analyze_damage_image_local(
    mock_model_class, mock_from_uri, mock_from_data, mock_init
) -> None:
    # Arrange
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Mocked damage description: stain in carpet."
    mock_model_instance.generate_content.return_value = mock_response

    mock_from_data.side_effect = lambda data, mime_type: f"Part(local, {mime_type})"

    # Act
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data=b"fake_image_bytes")
    ):
        result = analyze_damage_image(
            image_path="damage.png",
            prompt="Find damages",
            project_id="test-project",
            location="europe-west1",
        )

    # Assert
    mock_init.assert_called_once_with(project="test-project", location="europe-west1")
    mock_from_data.assert_called_once_with(data=b"fake_image_bytes", mime_type="image/png")
    mock_model_class.assert_called_once_with("gemini-1.5-flash")
    mock_model_instance.generate_content.assert_called_once_with(
        ["Part(local, image/png)", "Find damages"]
    )
    assert result == "Mocked damage description: stain in carpet."


@patch("inspector_agent_app.app_utils.vision.vertexai.init")
@patch("inspector_agent_app.app_utils.vision.Part.from_data")
@patch("inspector_agent_app.app_utils.vision.Part.from_uri")
@patch("inspector_agent_app.app_utils.vision.GenerativeModel")
def test_analyze_damage_image_gcs(
    mock_model_class, mock_from_uri, mock_from_data, mock_init
) -> None:
    # Arrange
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Mocked GCS damage description."
    mock_model_instance.generate_content.return_value = mock_response

    mock_from_uri.side_effect = lambda uri, mime_type: f"Part(gcs, {uri}, {mime_type})"

    # Act
    result = analyze_damage_image(
        image_path="gs://my-bucket/damage.webp",
        prompt=None,
        project_id="test-project",
        location="europe-west1",
    )

    # Assert
    mock_init.assert_called_once_with(project="test-project", location="europe-west1")
    mock_from_uri.assert_called_once_with(uri="gs://my-bucket/damage.webp", mime_type="image/webp")
    
    called_args = mock_model_instance.generate_content.call_args[0][0]
    assert called_args[0] == "Part(gcs, gs://my-bucket/damage.webp, image/webp)"
    assert "foto van een beschadigd" in called_args[1]
    assert result == "Mocked GCS damage description."
