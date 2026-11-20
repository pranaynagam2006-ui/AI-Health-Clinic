# AI DataAssist — HealthPulse Clinical Operations

A local-first clinical-supply inventory and vendor-purchasing workflow built with FastAPI, LangGraph, SQLite, and Pydantic. FastAPI serves a responsive single-page dashboard at `/` and `/embed`; its vanilla HTML/JavaScript frontend can also be embedded in another website. The dashboard includes live agent traces, the blocked-action simulator, schema inspection, clinic inventory, charts, and approval controls. A companion Streamlit UI remains available for local use. Gemini Flash is the default planner model; Ollama can replace it with one environment setting. Inventory and order tools run deterministically in Python, and model-produced tool arguments are checked against strict Pydantic schemas before execution.

The app creates purchase-order **drafts only**. Orders above the configured approval threshold pause at a durable LangGraph checkpoint, are recorded in SQLite, and continue only after an explicit YES or NO through the API or dashboard.

## Requirements

- Python 3.12 (tested and recommended; other Python versions are not currently validated)
- A Gemini API key from Google AI Studio for the default hosted free tier, or a local Ollama installation/model
- No paid service is required. The local database and vector index use files on disk.

The exact pinned versions are in [`requirements.txt`](requirements.txt). They target LangGraph 0.2.74, Pydantic 2.10.6, and the matching LangGraph SQLite checkpointer API. `sqlite3` is Python's standard-library module; no ORM or external database server is used.

## Setup

Run commands from the project directory.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Configure Gemini or Ollama

For Gemini, put your AI Studio key in `.env`:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.8-flash
GOOGLE_API_KEY=your-key
```

The Gemini integration reports a typed `missing_api_key` error rather than silently switching providers. Google controls its free-tier eligibility and rate limits; check the current AI Studio terms for your region.

To use Ollama instead, install Ollama and pull a supported local model, for example:

```bash
ollama pull qwen2.5:7b
```

Then change only the provider/model settings in `.env` (remove the API-key requirement):

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

`llama3.1:8b` can be used by setting `LLM_MODEL=llama3.1:8b`. Ollama must be running locally. Connection failures are surfaced as `llm_unavailable`.

Other useful settings:

```dotenv
DATABASE_PATH=./data/inventory.db
APPROVAL_THRESHOLD=0
POLICY_STORE_PATH=./data/policy_index
POLICY_FOLDER=./data/policies
API_BASE_URL=http://127.0.0.1:8001
```

With the default threshold of `$0`, every positive-cost purchase order waits for a decision. A request at or below the threshold is automatically approved; approval does not send an order to a supplier.

## Seed inventory from CSV

The bundled [`data/sample_inventory.csv`](data/sample_inventory.csv) has six PPE, surgical, diagnostic, and infection-control items distributed across three clinics and three vendors with varying lead times and minimum order quantities:

```bash
python -m scripts.seed
```

For a larger reproducible demo set, generate and import 2,000 products across 10 clinics, 12 suppliers, and six supply categories:

```bash
python -m scripts.generate_sample_data
python -m scripts.seed --csv data/large_sample_inventory.csv
```

The generator is deterministic by default. Change its size and seed when needed:

```bash
python -m scripts.generate_sample_data --count 5000 --seed 123 --output data/my_inventory.csv
python -m scripts.seed --csv data/my_inventory.csv
```

The bundled CSV import adds or updates its SKUs; it does not clear unrelated inventory. The API dashboard's inventory table and query chips use the imported rows immediately.

Import a larger/user-maintained CSV:

```bash
python -m scripts.seed --csv /path/to/products.csv
python -m scripts.seed --csv /path/to/products.csv --database ./data/custom.db
```

CSV columns:

| Column | Meaning |
| --- | --- |
| `sku` | Unique product identifier |
| `product_name` | Product label |
| `stock_level` | Current on-hand units (non-negative integer) |
| `daily_usage` | Average units used per day (non-negative number) |
| `lead_time_days` | Supplier lead time (positive integer) |
| `unit_cost` | Cost per unit in the CSV's currency units |
| `supplier_name`, `supplier_email` | Supplier contact data |
| `minimum_order_quantity` | Minimum quantity for a supplier order line |
| `reorder_point` | Optional integer; if omitted/blank, seeded as `ceil(daily_usage * lead_time_days)` |
| `category` | Optional clinical supply category |
| `clinic_id`, `clinic_name`, `city`, `region`, `bed_count` | Optional destination clinic fields |
| `vendor_rating`, `vendor_support_contact` | Optional vendor support and rating fields |

Vendor records are deduplicated by name. Re-importing a SKU updates its inventory, clinic assignment, and vendor terms. Invalid rows stop the import with the row number and validation details.

## Index supplier and policy documents

Place `.md` or `.txt` supplier/purchasing documents in `data/policies/` or another folder, then run:

```bash
python -m scripts.ingest_policies
# Or:
python -m scripts.ingest_policies --folder /path/to/policies --store ./data/policy_index
```

The first ingestion downloads the `all-MiniLM-L6-v2` sentence-transformer model for local CPU use. Chroma persists the embeddings locally. The planner retrieves up to three matching excerpts before making a decision. An absent index is permitted; retrieval starts when an index is built.

## Run the application

1. Seed inventory as above.
2. Start the API:

   ```bash
   uvicorn app.api:app --reload --host 127.0.0.1 --port 8001
   ```

3. Open the dashboard at <http://127.0.0.1:8001/> (or <http://127.0.0.1:8001/embed>), and interactive API docs at <http://127.0.0.1:8001/docs>. Choose another port with `--port` if 8001 is already in use.
4. Optionally, in a second terminal, start the Streamlit companion dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

The FastAPI dashboard offers a natural-language query bar with inventory and purchase-order example chips, a tool-execution/safety status, sortable inventory and purchase-order tables with clinic/vendor charts, a live structured trace with validation and latency, a schema inspector, and a blocked DELETE simulator. The pending-approval panel shows item, quantity, clinic, supplier, amount, and request time; its YES/NO buttons show progress and a friendly empty state, and approvals only record a decision (they do not send orders). The three agent cards show each agent's latest actual action, selected tool, validation status, elapsed time, and whether approval is pending. Built-in read-query chips go through the Planner Agent, Inventory Agent, and validated tools using deterministic intent routing, so they also work when the LLM is unavailable or rate-limited; free-form planning uses the configured Gemini/Ollama model. The page displays an iframe snippet for embedding; replace the localhost origin with the deployed HTTPS origin when hosting. You can also embed it directly:

```html
<iframe src="https://your-host.example/embed" title="Clinic Supply Desk"
  width="100%" height="900" loading="lazy" style="border:0;border-radius:16px"></iframe>
```

An example API request:

```bash
curl -X POST http://127.0.0.1:8001/api/query \
  -H 'Content-Type: application/json' \
  -d '{"message":"Check stock for GLV-NIT-M"}'
```

The API also exposes `GET /api/clinics`, `GET /api/vendors`, `GET /api/inventory`, `GET /api/approvals`, `POST /api/approvals/{order_id}`, `GET /api/orders/{order_id}`, `GET /api/schema`, `GET /api/logs`, `POST /api/security/simulate`, and `GET /health`. The original `/chat`, `/inventory`, and `/approvals` routes remain available as compatibility aliases. The simulator returns a typed blocked-operation response and writes an audit event; it never issues SQL or changes supply/order data. A pending approval is resolved with:

```json
{"decision": "YES", "thread_id": "the-thread-id-returned-by-chat"}
```

Pass the same `thread_id` returned from `/api/query` to `/api/approvals/{order_id}`. `NO` rejects the order. The SQLite database includes clinics, vendors, inventory, orders and line items, approval data, logs, plus the LangGraph checkpoint tables used for durable interrupt/resume.

## Architecture and safety

- **Planner Agent** interprets the user's request and chooses only explicitly listed inventory, purchase-order, reorder, and draft-order tools or responds directly.
- **Inventory Agent** validates and executes the tool using Python/SQLite parameterized statements. There is no model-facing SQL tool or arbitrary SQL execution path.
- **Approval Guardrail Agent** stores the pending order and pauses using LangGraph `interrupt`; resume runs through the persisted SQLite checkpointer only after a validated YES/NO.
- The dashboard's built-in low-stock, PPE, supplier-terms, pending-order, and clinic queries use deterministic Planner routing and then execute through the Inventory Agent. This exposes real per-agent logs without consuming LLM quota; other natural-language requests still use the configured LLM.
- Query responses are shown as conversational text above the results table. Built-in deterministic queries summarize their actual records (counts, reorder status, and supplier lead-time/MOQ ranges) without adding LLM quota usage; free-form LLM plans can return a contextual natural-language answer.
- Destructive operations are not registered as executable tools. The dashboard simulator submits a typed DELETE intent to the tool-layer guard, which logs and rejects it without performing a database write.
- Pydantic schemas use `extra="forbid"`. Tool arguments are validated once in the planner-output schema and again immediately before deterministic execution.
- `agent_logs` stores structured planner decisions, schema-validation results, tool calls/results and duration, blocked destructive attempts, approval requests, and decisions. `GET /api/logs` is the live trace source. `GET /api/schema` exposes columns for the inventory, clinic, vendor, purchase-order view, and audit tables.
- Tool calls for inventory lookup, reorder calculation, and draft creation are deterministic. Reorder threshold is lead-time demand (`ceil(daily usage × lead time)`); `needs_reorder` compares current stock against it. Draft line quantity is `max(requested quantity, supplier MOQ)`.
- Model initialization, provider errors, invalid arguments, missing Gemini keys, and invalid approvals return explicit typed API errors.

## Tests

```bash
pytest -q
```

Tests cover strict tool arguments/SQL rejection, destructive-operation blocking and audit logging, order queries, schema/dashboard routes, reorder calculations, MOQ handling, approval interrupt and resume (including rejection), and model configuration errors. The test graph uses an injected deterministic planner, so it does not require a Gemini key, network access, or Ollama.

## Free hosting notes

The least costly/reliable arrangement is to run the API and Streamlit locally and use Ollama. A hosted Gemini free tier may be used within Google's current limits. This app's database and LangGraph checkpoints are local SQLite files, and its Chroma index is local storage.

For a demo, deploy the FastAPI service on a free Render web-service tier, or use the optional Streamlit companion on Hugging Face Spaces. Configure environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `GOOGLE_API_KEY`, `DATABASE_PATH`, `APPROVAL_THRESHOLD`, and the policy paths) in host settings; never commit `.env`. Free instances may sleep, have ephemeral filesystems, or have changing quotas. SQLite/checkpoints and Chroma data can be lost when an instance is recycled unless the host explicitly provides persistent storage; do not treat a free ephemeral instance as a production durable order system. For hosted persistence, attach a persistent disk where available and ensure only one API process writes the SQLite file.

## Example request and trace

See [`sample_run.txt`](sample_run.txt) for a representative user request, approval pause, resume, and structured reasoning log.
