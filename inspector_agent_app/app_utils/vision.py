import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from typing import Optional

def analyze_damage_image(
    image_path: str,
    prompt: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """
    Analyzes an image of a broken/damaged part of a rental property using the Vertex AI SDK and Gemini model
    to identify visible damage and define the recommended reparation.

    Args:
        image_path: Local file path or GCS URI (gs://...) to the image of the damage.
        prompt: Optional custom text prompt to instruct Gemini.
        model_name: The model to use, defaults to 'gemini-1.5-flash'.
        project_id: GCP Project ID. Fallback to GOOGLE_CLOUD_PROJECT env var.
        location: GCP Location. Fallback to GOOGLE_CLOUD_LOCATION or us-central1.

    Returns:
        The text response from the model detailing the damage and reparation.
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

    image_part = create_image_part(image_path)

    # 3. Formulate prompt
    default_prompt = (
        "Onderzoek deze foto van een beschadigd onderdeel van een huurpand.\n"
        "1. Identificeer en beschrijf in detail welke schade er zichtbaar is op de foto.\n"
        "2. Definieer de reparatie of herstelwerkzaamheden die uitgevoerd moeten worden.\n"
        "Geef een duidelijk en gestructureerd overzicht van de schade en de aanbevolen oplossing."
    )
    final_prompt = prompt or default_prompt

    # 4. Invoke model
    model = GenerativeModel(model_name)
    response = model.generate_content([image_part, final_prompt])

    return response.text
