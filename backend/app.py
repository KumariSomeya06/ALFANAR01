from langchain_openai import AzureChatOpenAI  # type: ignore
from langgraph.graph import StateGraph, END  # type: ignore
from typing import TypedDict, List
import json, os
import requests
from dotenv import load_dotenv
load_dotenv()


# -----------------------------
# 1. Define Orchestrator State
# -----------------------------
class OrchestratorState(TypedDict):
    user_input: str
    intents: List[str]
    routes: List[str]
    result: dict
    container: str
    file_path: str
    final_response: str
    raw_text: str   # store extracted OCR text
    task_breakdown: str


# -----------------------------
# 2. Initialize Models
# -----------------------------
orchestrator_llm = AzureChatOpenAI(
    deployment_name="gpt-4o-realtime-preview",
    temperature=0
)

scm_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini", temperature=0)
ocr_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini", temperature=0)
sap_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini", temperature=0)
project_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini", temperature=0)


# -----------------------------
# 3. Orchestrator Node
# -----------------------------
def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    """Classify intents and decide task routing."""
    system_prompt = """
    You are an AI Orchestrator for a multi-agent workflow system.
    Your job is to read the user's input and decide which agents should be triggered.

    Rules:
    1. Identify ALL possible user intents from the input (not just one).
    2. Each intent maps to a route:
       - 'ocr' → If user mentions uploading, reading, or extracting data from a document/PDF/quote/invoice.
       - 'scm' → ONLY if user asks to compare, evaluate, analyze, or score vendors or risk analysis on qoutations , or any thing about the vendor informations.
       - 'sap' → ONLY SAP project creation/updates
       - 'project' → ONLY project planning/health/summary or any related things and data loading to table
       - 'end' → irrelevant requests
    3. Multi-intent:
       - Example: "Upload vendor PDF and compare quotations" → ["ocr","scm"]
       - Example : "If SAP project creation is mentioned with vendors" → ["ocr","scm","sap"]
       - Example : "If Only project health or data loading is given then →["ocr","scm", "project" ]
    4. Always return valid JSON with keys: intents, task_breakdown, routes.
    """

    response = orchestrator_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_input"]}
    ])

    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {
            "intents": ["unknown"],
            "task_breakdown": response.content,
            "routes": ["end"]
        }

    return {
        **state,
        "intents": parsed.get("intents", []),
        "task_breakdown": parsed.get("task_breakdown", ""),
        "routes": parsed.get("routes", ["end"]),
    }


# -----------------------------
# 4. Sub-Agent Nodes
# -----------------------------
def ocr_agent(state: OrchestratorState) -> OrchestratorState:
    """Fetch OCR text from Azure Function and normalize into schema."""
    try:
        url = os.getenv("AZURE_OCR_FUNCTION_URL")
        r = requests.get(url)
        raw_ocr = r.json().get("data", "")
        print("OCR RAW TEXT:", raw_ocr)
    except Exception as e:
        return {**state, "result": {"error": f"[OCR Agent ERROR] {str(e)}"}}

    # Store raw OCR text
    state["raw_text"] = raw_ocr

    system_prompt = f"""
    You are the OCR Normalization Agent.
    You must only extract fields directly from the provided OCR JSON.
    If a field is missing, set it to null and add its name in "missing_fields".
    Never invent or assume values.
    Schema:
    {{
        "raw_id": "RAW_xxxx",
        "doc_id": "DOC_2025_xxx",
        "rfq_id": "<string or null>",
        "vendor_name": "<string or null>",
        "warranty_months": <int or null>,
        "delivery_weeks": <int or null>,
        "price": "<string or null>",
        "payment_terms": "<string or null>",
        "page_number": <int or null>,
        "submitted_on": "<timestamp YYYY-MM-DD HH:MM>",
        "source_file_path": "{state['file_path']}",
        "missing_fields": [<list of field names>]
    }}
    Only return valid JSON. Never summarize or rephrase values.
    """

    llm_response = ocr_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"OCR RAW JSON: {raw_ocr}"}
    ])

    try:
        normalized = json.loads(llm_response.content.strip())
    except Exception:
        normalized = {"error": "Failed to parse JSON", "raw_response": llm_response.content}

    return {**state, "result": normalized}


def scm_agent(state: OrchestratorState) -> OrchestratorState:
    """Handle all vendor-related queries: comparison, evaluation, risk analysis, scoring, or details."""

    vendor_data = state.get("result", {})
    raw_text = state.get("raw_text", "")
    user_query = state.get("user_input", "")

    system_prompt = """
    You are the SCM (Supply Chain Management) Agent.
    Your job is to answer ANY query related to vendor quotations.

    Rules:
    1. Always rely on the structured vendor data provided (JSON fields like rfq_id, vendor_name, warranty_months, delivery_weeks, price, payment_terms).
    2. Do NOT hallucinate. If a field is missing, explicitly state it as "not available".
    3. Handle multiple use cases:
       - Comparison: If multiple vendors are present, compare them and highlight strengths/weaknesses.
       - Evaluation: If a single vendor, summarize their offer in detail.
       - Risk Analysis: Point out risks like high cost, delayed delivery, missing warranty, vague payment terms.
       - Scoring: Assign scores (0–100) with justification if the query asks for ranking/scoring.
       - Fact lookup: If the user asks for a specific field (e.g., RFQ ID, delivery time, payment terms), return the exact field value without rephrasing.
    4. Response format:
       - If a general query (comparison, evaluation, risk analysis) → return JSON:
         {
           "summary": "executive summary",
           "vendor_scores": [
             {"vendor": "Vendor A", "score": 85, "strengths": [...], "weaknesses": [...]},
             {"vendor": "Vendor B", "score": 78, "strengths": [...], "weaknesses": [...]}
           ],
           "recommendation": "..."
         }
       - If a fact lookup (like 'What is the RFQ ID?') → return JSON:
         { "answer": "<exact value or 'Not available'>" }
    5. Always return valid JSON only, never plain text.
    """

    response = scm_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {user_query}\nVendor data: {json.dumps(vendor_data, indent=2)}\nRaw text: {raw_text}"}
    ])

    try:
        parsed = json.loads(response.content.strip())
    except Exception:
        parsed = {"error": "Failed to parse SCM JSON", "raw_response": response.content}

    return {**state, "result": parsed}


def sap_agent(state: OrchestratorState) -> OrchestratorState:
    """Simulate SAP project creation."""
    raw_text = state.get("raw_text", "")
    result = sap_llm.invoke(
        f"Simulate SAP Project creation. Input: {state['user_input']}\nContext from document:\n{raw_text}"
    ).content
    return {**state, "result": {"sap_response": result}}


def project_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Provide project planning/health summary AND load project to Databricks via Azure Function.
    
    Expected input example:
    "Create SAP Project for Transmission Line TL-2025 with start 2025-09-01, budget ₹45Cr."
    
    Response:
    {
        "project_summary": "...",
        "databricks_status": "CREATED (Pending Activation)",
        "value": "Enables touchless project setup with instant alignment to procurement and finance."
    }
    """
    raw_text = state.get("raw_text", "")
    user_input = state.get("user_input", "")

    # Step 1: Generate project summary using LLM
    project_summary = project_llm.invoke(
        f"Provide detailed schedule, cost, and risk summary for project: {user_input}\nContext:\n{raw_text}"
    ).content

    # Step 2: Call Azure Function to insert project record into table
    databricks_status = None
    try:
        azure_func_url = os.getenv("AZURE_PROJECT_LOAD_FUNCTION_URL")
        payload = {"user_input": user_input}  # Can add more structured info if needed
        r = requests.post(azure_func_url, json=payload, timeout=15)
        r.raise_for_status()
        resp_json = r.json()
        databricks_status = resp_json.get("status", "UNKNOWN")
        value_description = resp_json.get("value", "Project setup executed.")
    except Exception as e:
        databricks_status = f"ERROR: {str(e)}"
        value_description = "Failed to trigger Databricks project load."

    # Combine results
    result = {
        "project_summary": project_summary,
        "databricks_status": databricks_status,
        "value": value_description
    }

    return {**state, "result": result}



# -----------------------------
# 5. Synthesizer Node
# -----------------------------
def response_synthesizer(state: OrchestratorState) -> OrchestratorState:
    """Convert structured results into a friendly chat response."""
    system_prompt = """
    You are a friendly business assistant chatbot.
    Summarize the structured results into a clear, conversational response.
    Avoid JSON or technical formatting.
    """

    structured_result = json.dumps(state.get("result", {}), indent=2)
    raw_text = state.get("raw_text", "")

    response = orchestrator_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize these results:\n{structured_result}\n\nContext from document:\n{raw_text}"}
    ])

    return {**state, "final_response": response.content}


# -----------------------------
# 6. Workflow Definition
# -----------------------------
workflow = StateGraph(OrchestratorState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("ocr", ocr_agent)
workflow.add_node("scm", scm_agent)
workflow.add_node("sap", sap_agent)
workflow.add_node("project", project_agent)
workflow.add_node("synthesizer", response_synthesizer)
workflow.set_entry_point("orchestrator")


def multi_intent_router(state: OrchestratorState):
    """Route execution to next required agent."""
    routes = state.get("routes", [])
    if not routes:
        return "synthesizer"
    next_route = routes.pop(0)
    state["routes"] = routes
    return next_route if next_route in ["ocr", "scm", "sap", "project"] else "synthesizer"


# Routing
for node in ["orchestrator", "ocr", "scm", "sap", "project"]:
    workflow.add_conditional_edges(
        node, multi_intent_router,
        {"ocr": "ocr", "scm": "scm", "sap": "sap", "project": "project", "synthesizer": "synthesizer"}
    )
workflow.add_edge("synthesizer", END)


# -----------------------------
# 7. Compile Workflow
# -----------------------------
ocr_app = workflow.compile()
