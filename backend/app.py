from langchain_openai import AzureChatOpenAI  # type: ignore
from langgraph.graph import StateGraph, END  # type: ignore
from typing import TypedDict, List, Dict
import json, os
import requests
from datetime import datetime, timezone
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
    # file_path: str
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
    try:
        url = os.getenv("AZURE_OCR_FUNCTION_URL")
        r = requests.get(url)
        raw_ocr = r.json()
        # print("OCR RAW RESPONSE:", raw_ocr)
    except Exception as e:
        return {**state, "result": {"error": f"[OCR Agent ERROR] {str(e)}"}}

    # Build a simplified list: [ {blob, extracted_on, text}, ... ]
    extracted_docs = []
    for item in raw_ocr.get("data", []):
        extracted_docs.append({
            "blob": item.get("blob"),
            "extracted_on_utc": item.get("extracted_on_utc"),
            "text": item.get("text")
        })

    # Save this in raw_text (stringified JSON so it survives downstream LLM input)
    state["raw_text"] = json.dumps(extracted_docs, indent=2)

    # --- Keep normalization pipeline the same ---
    system_prompt = f"""
    You are the OCR Normalization Agent.
    You will receive OCR-extracted blocks of text (one per vendor proposal).
    Your task is to parse and normalize them into structured JSON.

    Rules:
    - Extract only fields explicitly present.

    - Never hallucinate or invent values.
    - Each vendor entry must be a separate JSON object.
    - Do not include markdown formatting, code fences, or extra text. 

    Schema per vendor:
    {{
        "doc_id": "<string or null>",
        "rfq_id": "<string or null>",
        "vendor_name": "<string or null>",
        "submission_date": "<YYYY-MM-DD or null>",
        "unit_price": <float or null>,
        "delivery_weeks": <int or null>,
        "source_file_path": "<string or null>",  # blob path
        "extracted_on_utc": "<string or null>",  # extraction timestamp

    }}
    

    Return an array of such objects, one for each vendor.
    Only return valid JSON.
    """
    # "raw_id": "<string or null>",
    # "missing_fields": [<list of field names>]
    # - If a field is missing, set it to null and include it in "missing_fields".

    llm_response = ocr_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"OCR RAW DATA:\n{json.dumps(extracted_docs, indent=2)}"}
    ])

    try:
        normalized = json.loads(llm_response.content.strip())
    except Exception:
        normalized = {"error": "Failed to parse JSON", "raw_response": llm_response.content}

    return {**state, "result": normalized}



def scm_agent(state: OrchestratorState) -> OrchestratorState:
    """Handle all vendor-related queries: comparison, evaluation, risk analysis, scoring, or details."""
    # print("160", state.get("result"))

    # Normalize vendor_data
    result_data = state.get("result", [])

    if isinstance(result_data, list):
        vendor_data = result_data
    elif isinstance(result_data, dict):
        vendor_data = result_data.get("vendors", [])
        if isinstance(vendor_data, dict):
            vendor_data = [vendor_data]
    else:
        vendor_data = []

    raw_text = state.get("raw_text", "")
    user_query = state.get("user_input", "")

    # Step 1: Extract RFQ ID and vendor names from user query
    extractor_prompt = """
    You are an assistant that extracts RFQ ID and vendor names from user queries.

    Task:
    - User query contains exactly one RFQ ID and may mention multiple vendor names.
    - Return RFQ ID and list of vendor names explicitly mentioned.
    - Only return valid JSON.
    - Do not include markdown formatting, code fences, or extra text. 

    JSON Schema:
    {
      "rfq_id": "<string or null>",
      "vendor_names": ["<vendor name 1>", "<vendor name 2>", ...]
    }
    """
    try:
        extraction_resp = scm_llm.invoke([
            {"role": "system", "content": extractor_prompt},
            {"role": "user", "content": f"User query: {user_query}"}
        ])
        parsed_extract = json.loads(extraction_resp.content.strip())
        rfq_id = parsed_extract.get("rfq_id")
        vendor_names = parsed_extract.get("vendor_names", [])
    except Exception:
        rfq_id, vendor_names = None, []

    # Step 2: Identify missing vendors
    existing_names = {v.get("vendor_name") for v in vendor_data}
    missing_names = [vn for vn in vendor_names if vn not in existing_names]

    # Step 3: Call Azure function if there are missing vendors
    if rfq_id and missing_names:
        try:
            url = os.getenv("AZURE_VENDOR_FETCH_FUNCTION_URL")
            payload = {
                "rfq_id": rfq_id,
                "vendor_names": missing_names
            }
            # print("209", payload)
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            fetched_vendors = r.json()  # Expect list of vendor objects
            # print("213", fetched_vendors)

            if isinstance(fetched_vendors, dict):
                fetched_vendors = [fetched_vendors]

            vendor_data.extend(fetched_vendors)  # merge into single vendors list
        except Exception as e:
            return {**state, "result": {"error": f"[SCM Agent ERROR] Failed to fetch vendor data: {str(e)}"}}

    # Step 4: SCM reasoning / scoring / comparison
    system_prompt = """
    You are the SCM (Supply Chain Management) Agent.
    Your job is to answer ANY query related to vendor quotations.

    Rules:
    1. You will receive a list of vendor objects (not just one).
       Each vendor object may include:
       - rfq_id, vendor_name, warranty, delivery_weeks, proposal_amount, currency,
        payment_terms, source_file_path, extracted_on_utc.
    2. Do NOT hallucinate. If a field is missing, state "not available".
    3. Use weightage when comparing or scoring:
       - proposal_amount (Cost): 40%
       - delivery_weeks (Delivery Time): 25%
       - warranty: 15%
       - payment_terms: 10%
       - vendor reputation/name: 10%
    4. Handle multiple use cases:
       - Comparison, evaluation, risk analysis, scoring
       - Fact lookup if asked for a specific field
    5. Do not include markdown formatting, code fences, or extra text and Response must be valid JSON with structure:
    {
      "summary": "...",
      "vendor_scores": [
        {
          "vendor": "<name>",
          "score": 85,
          "strengths": [...],
          "weaknesses": [...],
          "source_file_path": "<blob>",
          "extracted_on_utc": "<timestamp>"
        }
      ],
      "vendors": [
        {
            "doc_id": "<string or null>",
            "rfq_id": "<string or null>",
            "vendor_name": "<string or null>",
            "submission_date": "<YYYY-MM-DD or null>",
            "unit_price": <float or null>,
            "delivery_weeks": <int or null>,
            "source_file_path": "<string or null>",
            "extracted_on_utc": "<string or null>"
        }
      ],
      "recommendation": "..."
    }
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

    Supports multi-vendor input from OCR or Azure Function.
    """

    raw_text = state.get("raw_text", "")
    user_input = state.get("user_input", "")

    result_data = state.get("result", {})

    # Normalize vendors safely
    if isinstance(result_data, list):
        vendors = result_data
    elif isinstance(result_data, dict):
        if "vendors" in result_data:
            vendors = result_data["vendors"]
            if not isinstance(vendors, list):
                vendors = [vendors]
        else:
            vendors = [result_data]
    else:
        vendors = []

    # print("line 311 vendors:", vendors)

    # Step 1: Generate project summary using LLM
    context_text = f"Project input: {user_input}\n"
    if vendors:
        context_text += f"Vendor proposals:\n{json.dumps(vendors, indent=2)}\n"
    context_text += f"Additional context:\n{raw_text}"

    project_summary = project_llm.invoke(
        """
        You are the Project Planning & Reporting Agent.
        Generate a **formal, well-structured project summary** based on the following context.

        Requirements:
        1. Response must always be in a clear and professional tone (formal).
        2. Organize output into sections with headings:
        - Executive Summary
        - Project Schedule (table format: Task | Start Date | End Date | Duration | Responsible Party)
        - Cost Breakdown (table format: Item | Estimated Cost | Currency | Notes)
        - Risk Assessment (table format: Risk | Probability | Impact | Mitigation Strategy)
        - Vendor Alignment (table format: Vendor | RFQ ID | Proposal Amount | Delivery Weeks | Strengths | Weaknesses)
        3. Use proper tabular formatting (no JSON, no markdown fences).
        4. If some data is unavailable, mark as “Not provided”.
        5. Ensure the response is concise but professional.

        Context:
        Project input: {user_input}

        Vendor proposals:
        {json.dumps({vendors}, indent=2)}

        Additional context:
        {raw_text}
        """
    ).content

    # Step 2: Call new Azure ingestion function
    try:
        azure_func_url = "https://alfanarapp.azurewebsites.net/api/ingest-keys"
        code = os.getenv("AZURE_PROJECT_LOAD_FUNCTION_CODE")

        # Create unique file name using first vendor’s rfq_id
        rfq_id = vendors[0].get("rfq_id", "UNKNOWN") if vendors else "UNKNOWN"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        file_name = f"{rfq_id}__{timestamp}.json"

        params = {
            "code": code,
            "file_name": file_name,
            "overwrite": "false",
            "mode": "ndjson",
        }

        # Azure expects {"data": [...]}
        payload = {"data": vendors}
        # print("payload:", payload)

        r = requests.post(
            azure_func_url,
            params=params,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        r.raise_for_status()
        resp_json = r.json()
        # print("393",resp_json)

        databricks_status = resp_json.get("status", "SUCCESS")
        value_description = resp_json.get("value", "Project vendors ingested.")
    except Exception as e:
        databricks_status = f"ERROR: {str(e)}"
        value_description = "Failed to trigger ingestion."

    result = {
        "project_summary": project_summary,
        "databricks_status": databricks_status,
        "value": value_description,
        "vendors": vendors,
    }

    return {**state, "result": result}



# -----------------------------
# 5. Synthesizer Node
# -----------------------------
def response_synthesizer(state: OrchestratorState) -> OrchestratorState:
    structured_result = json.dumps(state.get("result", {}), indent=2)
    user_query = state.get("user_input", "").strip()

    prompt = f"""
    You are a friendly AI assistant for multi-agent workflow.

    Instructions:
    1. Always provide the response in plain text (no JSON, no markdown fences).
    2. If the user input is **only about uploading / taking / sharing a file** (without asking a question or analysis),
       respond with something like:
       "Your file(s) have been processed successfully! You can now ask me questions about the documents."
    3. If the user input contains **both upload and analysis requests**, skip the upload confirmation
       and only provide the analysis/answer.
    4. If the result includes vendor comparison, scores, recommendations, or project summary, summarize them clearly.
    5. If the result contains **databricks_status** or **value**, always mention whether data loading succeeded or failed.
    6. If the query is chit-chat or not related to files/vendors/projects, just reply naturally.

    User query:
    "{user_query}"

    Structured data:
    {structured_result}
    """

    llm_response = orchestrator_llm.invoke([
        {"role": "system", "content": "You are a friendly and precise assistant."},
        {"role": "user", "content": prompt}
    ])

    return {**state, "final_response": llm_response.content}



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
    """Route execution to next required agent with OCR precedence."""
    routes = state.get("routes", [])

    if not routes:
        return "synthesizer"

    next_route = routes.pop(0)

    # Ensure OCR always runs first if required
    if next_route in ["scm", "sap", "project"]:
        # If OCR hasn't run yet, force it first
        if not state.get("raw_text"):  # no OCR results yet
            # Push the current route back so it executes after OCR
            state["routes"] = [next_route] + routes
            return "ocr"

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
