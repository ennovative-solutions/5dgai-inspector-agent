import os
from unittest.mock import MagicMock, patch, mock_open
import pytest

from inspector_agent_app.app_utils.vision import compare_images

@patch("inspector_agent_app.app_utils.vision.vertexai.init")
@patch("inspector_agent_app.app_utils.vision.Part.from_data")
@patch("inspector_agent_app.app_utils.vision.Part.from_uri")
@patch("inspector_agent_app.app_utils.vision.GenerativeModel")
def test_compare_images_local(
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
        result = compare_images(
            before_image_path="before.png",
            after_image_path="after.jpg",
            prompt="Find damages",
            project_id="test-project",
            location="europe-west1",
        )

    # Assert
    mock_init.assert_called_once_with(project="test-project", location="europe-west1")
    mock_from_data.assert_any_call(data=b"fake_image_bytes", mime_type="image/png")
    mock_from_data.assert_any_call(data=b"fake_image_bytes", mime_type="image/jpeg")
    mock_model_class.assert_called_once_with("gemini-1.5-flash")
    mock_model_instance.generate_content.assert_called_once_with(
        ["Part(local, image/png)", "Part(local, image/jpeg)", "Find damages"]
    )
    assert result == "Mocked damage description: stain in carpet."


@patch("inspector_agent_app.app_utils.vision.vertexai.init")
@patch("inspector_agent_app.app_utils.vision.Part.from_data")
@patch("inspector_agent_app.app_utils.vision.Part.from_uri")
@patch("inspector_agent_app.app_utils.vision.GenerativeModel")
def test_compare_images_gcs(
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
    result = compare_images(
        before_image_path="gs://my-bucket/before.png",
        after_image_path="gs://my-bucket/after.webp",
        prompt=None,
        project_id="test-project",
        location="europe-west1",
    )

    # Assert
    mock_init.assert_called_once_with(project="test-project", location="europe-west1")
    mock_from_uri.assert_any_call(uri="gs://my-bucket/before.png", mime_type="image/png")
    mock_from_uri.assert_any_call(uri="gs://my-bucket/after.webp", mime_type="image/webp")
    
    called_args = mock_model_instance.generate_content.call_args[0][0]
    assert called_args[0] == "Part(gcs, gs://my-bucket/before.png, image/png)"
    assert called_args[1] == "Part(gcs, gs://my-bucket/after.webp, image/webp)"
    assert "voor'-situatie" in called_args[2]
    assert result == "Mocked GCS damage description."
