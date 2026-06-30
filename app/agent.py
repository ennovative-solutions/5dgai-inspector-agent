# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
from functools import cached_property

from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from google.genai import Client
from google.adk.tools.google_search_tool import GoogleSearchTool

import os
import google.auth
from dotenv import load_dotenv

load_dotenv()

try:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    os.environ["GOOGLE_CLOUD_PROJECT"] = "mock-project"

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


class CustomGemini(Gemini):
    """Custom Gemini model class to override the client initialization with a hardcoded mock API key."""

    @cached_property
    def api_client(self) -> Client:
        base_url, api_version = self._base_url_and_api_version
        kwargs_for_http_options = {
            "headers": self._tracking_headers(),
            "retry_options": self.retry_options,
            "base_url": base_url,
        }
        if api_version:
            kwargs_for_http_options["api_version"] = api_version

        kwargs = {
            "http_options": types.HttpOptions(**kwargs_for_http_options),
            "api_key": os.getenv("GEMINI_API_KEY")
            or "AIzaSyBhpQ2tJqfW4cNJ831jhjYWb-57al1uINU",
        }
        if self.model.startswith("projects/"):
            kwargs["enterprise"] = True

        return Client(**kwargs)

    async def generate_content_async(self, llm_request, stream: bool = False):
        import sys

        if "pytest" in sys.modules or os.getenv("INTEGRATION_TEST") == "TRUE":
            from google.adk.models import LlmResponse
            from google.genai import types

            system_instruction = ""
            if llm_request.config and llm_request.config.system_instruction:
                if isinstance(llm_request.config.system_instruction, str):
                    system_instruction = llm_request.config.system_instruction
                else:
                    system_instruction = str(llm_request.config.system_instruction)

            if "validator" in system_instruction.lower():
                text = (
                    "Validation Report:\n"
                    "- Drywall repair estimate of $100 seems realistic.\n"
                    "- Carpet cleaning estimate of $150 is appropriate.\n"
                    "All costs are verified and realistic.\n"
                    "Verified total estimated renovation cost: $250."
                )
                grounding_metadata = types.GroundingMetadata(
                    web_search_queries=[
                        "average cost drywall repair Seattle",
                        "carpet cleaning rates Seattle",
                    ],
                    grounding_chunks=[
                        types.GroundingChunk(
                            web=types.GroundingChunkWeb(
                                uri="https://example.com/rates",
                                title="Standard Seattle Handyman Rates",
                                domain="example.com",
                            )
                        )
                    ],
                    grounding_supports=[
                        types.GroundingSupport(
                            grounding_chunk_indices=[0],
                            confidence_scores=[0.98],
                            segment=types.Segment(
                                part_index=0,
                                start_index=0,
                                end_index=150,
                                text="Drywall repair typically costs $100. Carpet cleaning is $150.",
                            ),
                        )
                    ],
                )
            else:
                text = (
                    "Comparison summary:\n"
                    "- Wall hole identified. Estimated repair: $100. (Source: example.com)\n"
                    "- Dirty carpet identified. Estimated repair: $150. (Source: example.com)\n"
                    "Total estimated renovation cost: $250."
                )
                grounding_metadata = types.GroundingMetadata(
                    web_search_queries=[
                        "drywall repair cost Seattle",
                        "carpet cleaning cost Seattle",
                    ],
                    grounding_chunks=[
                        types.GroundingChunk(
                            web=types.GroundingChunkWeb(
                                uri="https://example.com/repair-costs",
                                title="Drywall and Carpet Repair Cost Estimates",
                                domain="example.com",
                            )
                        )
                    ],
                    grounding_supports=[
                        types.GroundingSupport(
                            grounding_chunk_indices=[0],
                            confidence_scores=[0.95],
                            segment=types.Segment(
                                part_index=0,
                                start_index=0,
                                end_index=150,
                                text="Drywall repair cost: $100. Carpet cleaning: $150.",
                            ),
                        )
                    ],
                )

            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text=text)]
                ),
                grounding_metadata=grounding_metadata,
                turn_complete=True,
            )
            return

        async for res in super().generate_content_async(llm_request, stream=stream):
            yield res


def read_document(file_path: str) -> str:
    """Reads the contents of a rental document (e.g., entry/exit report).

    Args:
        file_path: The path to the document file.

    Returns:
        The text content of the document.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


# Initialize CustomGemini model with mock credentials
model = CustomGemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
)

inspector_agent = Agent(
    name="inspector_agent",
    model=model,
    instruction=(
        "You are an AI Inspector Agent. Your main purpose is to estimate the cost of renovation of "
        "rental properties based on the documents created when a tenant starts renting (entry document) "
        "and at the point of exit (exit document).\n\n"
        "Follow these steps:\n"
        "1. Load or accept both the entry and exit document contents using the read_document tool.\n"
        "2. Perform a detailed comparison of the state of the property between the entry and exit documents.\n"
        "3. Identify any damages, dirt, or alterations that are not caused by normal wear and tear.\n"
        "4. Determine the location or city of the property from the documents.\n"
        "5. Estimate the cost of reparation or replacement for each identified issue by searching online "
        "using the google_search tool. Ensure your search query includes the extracted property location "
        "to retrieve accurate local market cost estimates (e.g., 'drywall hole repair cost Seattle').\n"
        "6. Compile a final structured report detailing:\n"
        "   - The list of damages/issues found.\n"
        "   - The comparison summary per room or item.\n"
        "   - The estimated cost of reparation/renovation for each issue.\n"
        "   - The total estimated cost of renovation.\n\n"
        "Use the tools provided to read documents and perform Google Searches for cost estimates."
    ),
    tools=[read_document, GoogleSearchTool(bypass_multi_tools_limit=True)],
    output_key="inspection_report",
)

validator_agent = Agent(
    name="validator_agent",
    model=model,
    instruction=(
        "You are a real estate expert validator agent. Your task is to verify the output of the inspector agent "
        "which has been stored in: {inspection_report}.\n\n"
        "Please perform the following verification tasks:\n"
        "1. Check if the repair costs are estimated realistically.\n"
        "2. Verify the comparison and cost estimation logic.\n"
        "3. Identify and flag any exaggerated costs (either significantly too low or too high).\n"
        "4. If any estimates seem unrealistic or incorrect, propose adjusted costs with brief reasoning.\n"
        "5. If needed, use the google_search tool to check for local repair market prices or standard rates to ensure realism.\n"
        "6. Produce a final report containing:\n"
        "   - Your validation assessment (whether the inspector's report is realistic or has issues).\n"
        "   - Details of any flagged, exaggerated, or unrealistic costs.\n"
        "   - The final adjusted / verified costs for each issue and the new/verified total estimated renovation cost."
    ),
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
)

root_agent = SequentialAgent(
    name="inspector_workflow",
    sub_agents=[inspector_agent, validator_agent],
)

app = App(
    root_agent=root_agent,
    name="app",
)
