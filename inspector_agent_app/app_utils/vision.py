import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from typing import Optional

def compare_images(
    before_image_path: str,
    after_image_path: str,
    prompt: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """
    Compares two images ('before' and 'after') using the Vertex AI SDK and Gemini model
    to identify differences indicating potential damages.

    Args:
        before_image_path: Local file path or GCS URI (gs://...) to the before image.
        after_image_path: Local file path or GCS URI (gs://...) to the after image.
        prompt: Optional custom text prompt to instruct Gemini.
        model_name: The model to use, defaults to 'gemini-1.5-flash'.
        project_id: GCP Project ID. Fallback to GOOGLE_CLOUD_PROJECT env var.
        location: GCP Location. Fallback to GOOGLE_CLOUD_LOCATION or us-central1.

    Returns:
        The text response from the model detailing the differences.
    """
    # 1. Initialize Vertex AI
    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    resolved_location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if resolved_project:
        vertexai.init(project=resolved_project, location=resolved_location)
    else:
        vertexai.init(location=resolved_location)

    # Helper to get MIME type based on file extension
    def get_mime_type(path: str) -> str:
        lower_path = path.lower()
        if lower_path.endswith(".png"):
            return "image/png"
        elif lower_path.endswith(".webp"):
            return "image/webp"
        elif lower_path.endswith(".gif"):
            return "image/gif"
        return "image/jpeg"

    # Helper to build Part object from path or GCS URI
    def create_image_part(path: str) -> Part:
        mime_type = get_mime_type(path)
        if path.startswith("gs://"):
            return Part.from_uri(uri=path, mime_type=mime_type)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found at local path: {path}")
            with open(path, "rb") as f:
                data = f.read()
            return Part.from_data(data=data, mime_type=mime_type)

    before_part = create_image_part(before_image_path)
    after_part = create_image_part(after_image_path)

    # 3. Formulate prompt
    default_prompt = (
        "Hier zijn twee foto's van een huurpand: de eerste is de 'voor'-situatie (aanvang huur) "
        "en de tweede is de 'na'-situatie (einde huur). Vergelijk de twee foto's in detail en "
        "geef alle verschillen aan die duiden op mogelijke schade of veranderingen die niet "
        "onder normale slijtage vallen. Wees specifiek over wat er veranderd is en waar."
    )
    final_prompt = prompt or default_prompt

    # 4. Invoke model
    model = GenerativeModel(model_name)
    response = model.generate_content([before_part, after_part, final_prompt])

    return response.text
