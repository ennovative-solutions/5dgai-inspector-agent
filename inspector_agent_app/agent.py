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
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

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
            "api_key": os.getenv("GEMINI_API_KEY"),
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
                    "  Handyman details: John Doe (Drywall Expert) - contact: john@example.com (Source: find_handymen)\n"
                    "- Carpet cleaning estimate of $150 is appropriate.\n"
                    "  Handyman details: Jane Smith (Carpet Care) - contact: jane@example.com (Source: find_handymen)\n"
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
                    "  Handyman details: John Doe (Drywall Expert) - contact: john@example.com (Source: find_handymen)\n"
                    "- Dirty carpet identified. Estimated repair: $150. (Source: example.com)\n"
                    "  Handyman details: Jane Smith (Carpet Care) - contact: jane@example.com (Source: find_handymen)\n"
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


def duckduckgo_search(query: str) -> dict:
    """Searches the web using DuckDuckGo to find information, cost estimates, or rates.

    Args:
        query: The search query string.

    Returns:
        A dict with 'status' and 'results' or 'error' keys.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = [
                {
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", ""),
                }
                for r in ddgs.text(query, max_results=3)
            ]
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def analyze_property_damage_image(image_path: str) -> dict:
    """Analyzes an image of a broken/damaged part of a property to identify the damage and recommended reparation.

    Args:
        image_path: The local path or GCS URI (gs://...) to the image of the damage.

    Returns:
        A dict with keys 'status' and 'result' (detailing the damage and reparation) or 'error' (detailing the error).
    """
    try:
        from inspector_agent_app.app_utils.vision import analyze_damage_image
        result = analyze_damage_image(image_path=image_path)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Initialize CustomGemini model with mock credentials
model = CustomGemini(
    model="gemini-3.5-flash",
    retry_options=types.HttpRetryOptions(
        attempts=6,
        initial_delay=2.0,
        http_status_codes=[429, 500, 502, 503, 504],
    ),
)

mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=[
                "--directory",
                "/Users/dennisbaerten/Development/repairman-mcp-server",
                "run",
                "repairman-mcp-server",
            ],
        )
    )
)

inspector_agent = Agent(
    name="inspector_agent",
    model=model,
    instruction=(
        "You are an AI property inspector agent. Your main purpose is to estimate the cost of property renovation "
        "by comparing the entry state and exit state of a rental property.\n\n"
        "The conversation will progress in multiple turns:\n"
        "1. In the first turn, the user will provide the Entry Inspection Report. You must simply acknowledge receipt of the Entry report, summarize the initial state of the property, and ask the user to provide the Exit Inspection Report. Do not call any search tools or find handymen at this stage.\n"
        "2. Once the user provides the Exit Inspection Report, compare the state of the property between the entry and exit reports, identify any damages/alterations not caused by normal wear and tear, determine the location, and proceed with estimation and handyman lookup.\n\n"
        "Follow these steps when exit report is provided:\n"
        "- Step A: Perform a detailed comparison of the state of the property between the entry and exit documents. If the user provides a picture of a damaged or broken part of the property, use the `analyze_property_damage_image` tool to investigate the damage, check what damage is visible in the picture, and define the reparation that can be done.\n"
        "- Step B: Identify any damages, dirt, or alterations that are not caused by normal wear and tear.\n"
        "- Step C: Determine the location or city of the property from the documents. CRITICAL: If the location/city is not specified in either the Entry or Exit Inspection Report, you MUST stop and ask the user to provide the location of the property. Do not call any search tools or find handymen, and do not perform Step D, E, or F until the location is provided by the user.\n"
        "- Step D: Estimate the cost of reparation or replacement for each identified issue by searching online "
        "using the duckduckgo_search tool. Ensure your search query includes the extracted property location "
        "to retrieve accurate local market cost estimates (e.g., 'drywall hole repair cost Seattle').\n"
        "- Step E: For each identified damage or repair issue, use the find_handymen tool (specifying the damage "
        "description as 'problem' and the property location/city as 'location') to retrieve details of a "
        "repairman who can make the fix.\n"
        "- Step F: Compile a final structured report detailing the comparison, cost estimates, matching handyman info, and total estimated cost of renovation.\n\n"
        "CRITICAL BEHAVIOR: Never search the web or filesystem for the entry or exit reports themselves. Only use duckduckgo_search to find repair and cleaning costs for the specific damages identified."
    ),
    tools=[duckduckgo_search, mcp_toolset, analyze_property_damage_image],
    output_key="inspection_report",
)

validator_agent = Agent(
    name="validator_agent",
    model=model,
    instruction=(
        "You are a real estate expert validator agent. Your task is to verify the output of the inspector agent "
        "which has been stored in: {inspection_report}.\n\n"
        "CRITICAL: If the inspector agent is still waiting for the Exit Inspection Report (i.e. the exit report has not been provided yet) or is asking the user for the location of the property (because it was not specified in the reports), simply confirm this status / support the request and do not call any search tools or perform validation.\n\n"
        "If the exit report has been compared and verified, perform the following verification tasks:\n"
        "1. Check if the repair costs are estimated realistically.\n"
        "2. Verify the comparison and cost estimation logic.\n"
        "3. Identify and flag any exaggerated costs (either significantly too low or too high).\n"
        "4. If any estimates seem unrealistic or incorrect, propose adjusted costs with brief reasoning.\n"
        "5. If needed, use the duckduckgo_search tool to check for local repair market prices or standard rates to ensure realism.\n"
        "6. Verify or look up handymen details using the find_handymen tool if adjustments are made or if handymen details are missing/unrealistic.\n"
        "7. Produce a final report containing:\n"
        "   - Your validation assessment (whether the inspector's report is realistic or has issues).\n"
        "   - Details of any flagged, exaggerated, or unrealistic costs.\n"
        "   - The final adjusted / verified costs for each issue and the final list of verified handymen details for each issue.\n"
        "   - The new/verified total estimated renovation cost."
    ),
    tools=[duckduckgo_search, mcp_toolset],
)

root_agent = SequentialAgent(
    name="inspector_agent",
    sub_agents=[inspector_agent, validator_agent],
)

app = App(
    root_agent=root_agent,
    name="inspector_agent_app",
)
