import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS clinics (
                    clinic_id TEXT PRIMARY KEY,
                    clinic_name TEXT NOT NULL,
                    city TEXT NOT NULL,
                    region TEXT NOT NULL,
                    bed_count INTEGER NOT NULL CHECK(bed_count >= 0)
                );
                CREATE TABLE IF NOT EXISTS vendors (
                    vendor_id INTEGER PRIMARY KEY,
                    vendor_name TEXT NOT NULL UNIQUE,
                    rating REAL NOT NULL DEFAULT 5.0 CHECK(rating >= 0 AND rating <= 5),
                    support_contact TEXT NOT NULL,
                    lead_time_days INTEGER NOT NULL DEFAULT 1 CHECK(lead_time_days > 0),
                    minimum_order_quantity INTEGER NOT NULL DEFAULT 1 CHECK(minimum_order_quantity > 0)
                );
                CREATE TABLE IF NOT EXISTS suppliers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inventory (
                    sku TEXT PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    stock_level INTEGER NOT NULL CHECK(stock_level >= 0),
                    daily_usage REAL NOT NULL CHECK(daily_usage >= 0),
                    lead_time_days INTEGER NOT NULL CHECK(lead_time_days > 0),
                    reorder_point INTEGER NOT NULL CHECK(reorder_point >= 0),
                    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
                    vendor_id INTEGER REFERENCES vendors(vendor_id),
                    minimum_order_quantity INTEGER NOT NULL CHECK(minimum_order_quantity > 0),
                    unit_cost_cents INTEGER NOT NULL CHECK(unit_cost_cents > 0),
                    category TEXT NOT NULL DEFAULT 'Other',
                    clinic_id TEXT REFERENCES clinics(clinic_id)
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
                    vendor_id INTEGER REFERENCES vendors(vendor_id),
                    clinic_id TEXT REFERENCES clinics(clinic_id),
                    status TEXT NOT NULL CHECK(status IN ('draft', 'pending', 'approved', 'rejected')),
                    total_cost_cents INTEGER NOT NULL CHECK(total_cost_cents >= 0),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
                    sku TEXT NOT NULL REFERENCES inventory(sku),
                    quantity INTEGER NOT NULL CHECK(quantity > 0),
                    unit_cost_cents INTEGER NOT NULL CHECK(unit_cost_cents > 0)
                );
                CREATE VIEW IF NOT EXISTS purchase_orders AS
                    SELECT o.id AS po_id, o.clinic_id, oi.sku AS item_id, o.vendor_id,
                           oi.quantity, o.total_cost_cents, o.created_at AS order_date, o.status
                    FROM orders o JOIN order_items oi ON oi.order_id=o.id;
                CREATE TABLE IF NOT EXISTS agent_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS agent_logs_thread_idx ON agent_logs(thread_id, id);
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    order_id TEXT PRIMARY KEY REFERENCES orders(id),
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                    decision TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS pending_approvals_status_idx ON pending_approvals(status);
                """
            )
            self._ensure_column(conn, "inventory", "category", "TEXT NOT NULL DEFAULT 'Other'")
            self._ensure_column(conn, "inventory", "clinic_id", "TEXT")
            self._ensure_column(conn, "inventory", "vendor_id", "INTEGER")
            self._ensure_column(conn, "orders", "vendor_id", "INTEGER")
            self._ensure_column(conn, "orders", "clinic_id", "TEXT")
            conn.executemany(
                """INSERT INTO clinics(clinic_id, clinic_name, city, region, bed_count)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(clinic_id) DO NOTHING""",
                [
                    ("CLINIC-001", "Northside Community Clinic", "Seattle", "North", 50),
                    ("CLINIC-002", "Downtown Family Clinic", "Seattle", "Central", 32),
                    ("CLINIC-003", "Lakeshore Diagnostic Clinic", "Bellevue", "East", 24),
                ],
            )
            conn.execute(
                """INSERT INTO vendors(vendor_name, rating, support_contact, lead_time_days, minimum_order_quantity)
                   SELECT s.name, 5.0, s.email, COALESCE(MIN(i.lead_time_days), 1),
                          COALESCE(MIN(i.minimum_order_quantity), 1)
                   FROM suppliers s LEFT JOIN inventory i ON i.supplier_id=s.id
                   GROUP BY s.id
                   ON CONFLICT(vendor_name) DO NOTHING"""
            )
            conn.execute(
                """UPDATE inventory SET vendor_id=(
                       SELECT v.vendor_id FROM vendors v JOIN suppliers s ON s.name=v.vendor_name
                       WHERE s.id=inventory.supplier_id
                   ) WHERE vendor_id IS NULL"""
            )
            conn.execute(
                """UPDATE inventory SET clinic_id='CLINIC-001'
                   WHERE clinic_id IS NULL OR clinic_id=''"""
            )
            conn.execute(
                """UPDATE orders SET vendor_id=(
                       SELECT v.vendor_id FROM vendors v JOIN suppliers s ON s.name=v.vendor_name
                       WHERE s.id=orders.supplier_id
                   ) WHERE vendor_id IS NULL"""
            )
            conn.execute(
                """UPDATE orders SET clinic_id=(
                       SELECT i.clinic_id FROM order_items oi JOIN inventory i ON i.sku=oi.sku
                       WHERE oi.order_id=orders.id LIMIT 1
                   ) WHERE clinic_id IS NULL"""
            )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def log_event(self, thread_id: str, agent: str, event: str, details: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO agent_logs(thread_id, agent, event, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, agent, event, json.dumps(details, default=str), _now()),
            )

    def list_logs(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if thread_id:
                rows = conn.execute(
                    "SELECT * FROM agent_logs WHERE thread_id = ? ORDER BY id", (thread_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM agent_logs ORDER BY id DESC LIMIT 200").fetchall()
        return [
            {
                "id": row["id"],
                "thread_id": row["thread_id"],
                "agent": row["agent"],
                "event": row["event"],
                "details": json.loads(row["details_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def upsert_inventory(
        self,
        *,
        sku: str,
        product_name: str,
        stock_level: int,
        daily_usage: float,
        lead_time_days: int,
        reorder_point: int,
        supplier_name: str,
        supplier_email: str,
        minimum_order_quantity: int,
        unit_cost_cents: int,
        category: str = "Other",
        clinic_id: str = "CLINIC-001",
        clinic_name: str = "Northside Community Clinic",
        city: str = "Seattle",
        region: str = "North",
        bed_count: int = 50,
        vendor_rating: float = 5.0,
        vendor_support_contact: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO clinics(clinic_id, clinic_name, city, region, bed_count)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(clinic_id) DO UPDATE SET clinic_name=excluded.clinic_name,
                     city=excluded.city, region=excluded.region, bed_count=excluded.bed_count""",
                (clinic_id, clinic_name, city, region, bed_count),
            )
            conn.execute(
                """INSERT INTO suppliers(name, email) VALUES (?, ?)
                   ON CONFLICT(name) DO UPDATE SET email = excluded.email""",
                (supplier_name, supplier_email),
            )
            supplier_id = conn.execute("SELECT id FROM suppliers WHERE name = ?", (supplier_name,)).fetchone()["id"]
            conn.execute(
                """INSERT INTO vendors(vendor_name, rating, support_contact, lead_time_days, minimum_order_quantity)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(vendor_name) DO UPDATE SET rating=excluded.rating,
                     support_contact=excluded.support_contact, lead_time_days=excluded.lead_time_days,
                     minimum_order_quantity=excluded.minimum_order_quantity""",
                (
                    supplier_name,
                    vendor_rating,
                    vendor_support_contact or supplier_email,
                    lead_time_days,
                    minimum_order_quantity,
                ),
            )
            vendor_id = conn.execute(
                "SELECT vendor_id FROM vendors WHERE vendor_name = ?", (supplier_name,)
            ).fetchone()["vendor_id"]
            conn.execute(
                """INSERT INTO inventory(sku, product_name, stock_level, daily_usage, lead_time_days,
                     reorder_point, supplier_id, vendor_id, minimum_order_quantity, unit_cost_cents, category, clinic_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(sku) DO UPDATE SET product_name=excluded.product_name,
                     stock_level=excluded.stock_level, daily_usage=excluded.daily_usage,
                     lead_time_days=excluded.lead_time_days, reorder_point=excluded.reorder_point,
                     supplier_id=excluded.supplier_id, vendor_id=excluded.vendor_id,
                     minimum_order_quantity=excluded.minimum_order_quantity,
                     unit_cost_cents=excluded.unit_cost_cents, category=excluded.category,
                     clinic_id=excluded.clinic_id""",
                (
                    sku,
                    product_name,
                    stock_level,
                    daily_usage,
                    lead_time_days,
                    reorder_point,
                    supplier_id,
                    vendor_id,
                    minimum_order_quantity,
                    unit_cost_cents,
                    category,
                    clinic_id,
                ),
            )

    def query_inventory(
        self,
        sku: str | None = None,
        search: str | None = None,
        category: str | None = None,
        clinic_id: str | None = None,
        low_stock_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = """SELECT i.sku, i.product_name, i.stock_level, i.daily_usage, i.lead_time_days,
                   i.reorder_point, i.minimum_order_quantity, i.unit_cost_cents, i.category,
                   i.clinic_id, c.clinic_name, c.city, c.region, c.bed_count,
                   s.name AS supplier_name, s.email AS supplier_email,
                   v.vendor_id, v.rating AS vendor_rating, v.support_contact
                   FROM inventory i JOIN suppliers s ON s.id=i.supplier_id
                   JOIN vendors v ON v.vendor_id=i.vendor_id
                   LEFT JOIN clinics c ON c.clinic_id=i.clinic_id"""
        filters: list[str] = []
        params: list[Any] = []
        if sku:
            filters.append("i.sku = ?")
            params.append(sku)
        if search:
            needle = f"%{search}%"
            filters.append("(i.product_name LIKE ? OR i.sku LIKE ? OR s.name LIKE ? OR c.clinic_name LIKE ?)")
            params.extend([needle, needle, needle, needle])
        if category:
            filters.append("i.category LIKE ?")
            params.append(category)
        if clinic_id:
            filters.append("i.clinic_id = ?")
            params.append(clinic_id)
        if low_stock_only:
            filters.append("i.stock_level <= i.reorder_point")
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY i.sku"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def create_draft_order(
        self,
        sku: str,
        requested_quantity: int,
        clinic_id: str | None = None,
    ) -> dict[str, Any]:
        order_id = f"PO-{uuid.uuid4().hex[:10].upper()}"
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            item = conn.execute(
                """SELECT i.*, s.name AS supplier_name, s.email AS supplier_email,
                          v.vendor_id, v.rating AS vendor_rating, v.support_contact,
                          c.clinic_name
                   FROM inventory i JOIN suppliers s ON s.id=i.supplier_id
                   JOIN vendors v ON v.vendor_id=i.vendor_id
                   JOIN clinics c ON c.clinic_id=i.clinic_id
                   WHERE i.sku=?""",
                (sku,),
            ).fetchone()
            if item is None:
                raise ValueError(f"Unknown SKU: {sku}")
            destination_clinic_id = clinic_id or item["clinic_id"]
            clinic = conn.execute(
                "SELECT clinic_name FROM clinics WHERE clinic_id=?", (destination_clinic_id,)
            ).fetchone()
            if clinic is None:
                raise ValueError(f"Unknown clinic: {destination_clinic_id}")
            quantity = max(requested_quantity, item["minimum_order_quantity"])
            total_cost_cents = quantity * item["unit_cost_cents"]
            conn.execute(
                """INSERT INTO orders(id, supplier_id, vendor_id, clinic_id, status, total_cost_cents, created_at)
                   VALUES (?, ?, ?, ?, 'draft', ?, ?)""",
                (
                    order_id,
                    item["supplier_id"],
                    item["vendor_id"],
                    destination_clinic_id,
                    total_cost_cents,
                    _now(),
                ),
            )
            conn.execute(
                "INSERT INTO order_items(order_id, sku, quantity, unit_cost_cents) VALUES (?, ?, ?, ?)",
                (order_id, sku, quantity, item["unit_cost_cents"]),
            )
        return {
            "order_id": order_id,
            "sku": sku,
            "product_name": item["product_name"],
            "supplier_name": item["supplier_name"],
            "supplier_email": item["supplier_email"],
            "vendor_id": item["vendor_id"],
            "vendor_rating": item["vendor_rating"],
            "clinic_id": destination_clinic_id,
            "clinic_name": clinic["clinic_name"],
            "requested_quantity": requested_quantity,
            "quantity": quantity,
            "minimum_order_quantity": item["minimum_order_quantity"],
            "unit_cost_cents": item["unit_cost_cents"],
            "total_cost_cents": total_cost_cents,
            "total_cost": total_cost_cents / 100,
            "status": "draft",
        }

    def set_order_status(self, order_id: str, status: str) -> None:
        with self.connect() as conn:
            result = conn.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))
            if result.rowcount != 1:
                raise ValueError(f"Unknown order: {order_id}")

    def record_pending_approval(self, order_id: str, thread_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("UPDATE orders SET status='pending' WHERE id=?", (order_id,))
            inserted = conn.execute(
                """INSERT INTO pending_approvals(order_id, thread_id, status, created_at)
                   VALUES (?, ?, 'pending', ?)
                   ON CONFLICT(order_id) DO NOTHING""",
                (order_id, thread_id, _now()),
            ).rowcount
        return inserted == 1

    def resolve_approval(self, order_id: str, thread_id: str, decision: str) -> None:
        status = "approved" if decision == "YES" else "rejected"
        with self.connect() as conn:
            result = conn.execute(
                """UPDATE pending_approvals SET status=?, decision=?, resolved_at=?
                   WHERE order_id=? AND thread_id=? AND status='pending'""",
                (status, decision, _now(), order_id, thread_id),
            )
            if result.rowcount != 1:
                raise ValueError(f"Pending approval not found for order {order_id}")
            conn.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))

    def get_pending_approval(self, order_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT p.order_id, p.thread_id, p.status, p.created_at, o.total_cost_cents
                   FROM pending_approvals p JOIN orders o ON o.id=p.order_id WHERE p.order_id=?""",
                (order_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.order_id, p.thread_id, p.status, p.created_at, o.total_cost_cents,
                          s.name AS supplier_name, o.clinic_id, c.clinic_name,
                          (SELECT group_concat(i.product_name || ' (' || oi.sku || ') × ' ||
                                               oi.quantity, '; ')
                           FROM order_items oi JOIN inventory i ON i.sku=oi.sku
                           WHERE oi.order_id=o.id) AS items_summary
                   FROM pending_approvals p JOIN orders o ON o.id=p.order_id
                   JOIN suppliers s ON s.id=o.supplier_id
                   LEFT JOIN clinics c ON c.clinic_id=o.clinic_id
                   WHERE p.status='pending'
                   ORDER BY p.created_at"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            order = conn.execute(
                """SELECT o.id, o.status, o.total_cost_cents, o.created_at, o.clinic_id,
                          s.name AS supplier_name, s.email AS supplier_email,
                          c.clinic_name, v.vendor_id, v.rating AS vendor_rating
                   FROM orders o JOIN suppliers s ON s.id=o.supplier_id
                   LEFT JOIN vendors v ON v.vendor_id=o.vendor_id
                   LEFT JOIN clinics c ON c.clinic_id=o.clinic_id WHERE o.id=?""",
                (order_id,),
            ).fetchone()
            if order is None:
                return None
            items = conn.execute(
                """SELECT oi.sku, i.product_name, oi.quantity, oi.unit_cost_cents
                   FROM order_items oi JOIN inventory i ON i.sku=oi.sku WHERE oi.order_id=?""",
                (order_id,),
            ).fetchall()
        result = dict(order)
        result["items"] = [dict(item) for item in items]
        result["total_cost"] = result["total_cost_cents"] / 100
        return result

    def list_clinics(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT clinic_id, clinic_name, city, region, bed_count FROM clinics ORDER BY clinic_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_vendors(self, vendor_name: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            query = """SELECT vendor_id, vendor_name, rating, support_contact,
                              lead_time_days, minimum_order_quantity
                       FROM vendors"""
            params: tuple[Any, ...] = ()
            if vendor_name:
                query += " WHERE vendor_name LIKE ?"
                params = (f"%{vendor_name}%",)
            rows = conn.execute(query + " ORDER BY vendor_name", params).fetchall()
        return [dict(row) for row in rows]

    def query_orders(
        self,
        status: str | None = None,
        vendor_name: str | None = None,
        clinic_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """SELECT o.id, o.status, o.total_cost_cents, o.created_at, o.clinic_id,
                          c.clinic_name, v.vendor_name, oi.sku, i.product_name,
                          oi.quantity, oi.unit_cost_cents
                   FROM orders o
                   JOIN vendors v ON v.vendor_id=o.vendor_id
                   LEFT JOIN clinics c ON c.clinic_id=o.clinic_id
                   JOIN order_items oi ON oi.order_id=o.id
                   JOIN inventory i ON i.sku=oi.sku"""
        filters: list[str] = []
        params: list[Any] = []
        if status:
            filters.append("o.status=?")
            params.append(status)
        if vendor_name:
            filters.append("v.vendor_name LIKE ?")
            params.append(f"%{vendor_name}%")
        if clinic_id:
            filters.append("o.clinic_id=?")
            params.append(clinic_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY o.created_at DESC, o.id"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        orders: dict[str, dict[str, Any]] = {}
        for row in rows:
            order_id = row["id"]
            order = orders.setdefault(
                order_id,
                {
                    "order_id": order_id,
                    "status": row["status"],
                    "total_cost_cents": row["total_cost_cents"],
                    "created_at": row["created_at"],
                    "clinic_id": row["clinic_id"],
                    "clinic_name": row["clinic_name"],
                    "vendor_name": row["vendor_name"],
                    "items": [],
                },
            )
            order["items"].append(
                {
                    "sku": row["sku"],
                    "product_name": row["product_name"],
                    "quantity": row["quantity"],
                    "unit_cost_cents": row["unit_cost_cents"],
                }
            )
        return list(orders.values())

    def inspect_schema(self) -> list[dict[str, Any]]:
        objects = (
            "inventory",
            "clinics",
            "vendors",
            "purchase_orders",
            "agent_logs",
            "orders",
            "order_items",
            "pending_approvals",
        )
        with self.connect() as conn:
            result = []
            for name in objects:
                exists = conn.execute(
                    "SELECT type FROM sqlite_master WHERE name=?", (name,)
                ).fetchone()
                if exists is None:
                    continue
                columns = conn.execute(f"PRAGMA table_info('{name}')").fetchall()
                result.append(
                    {
                        "name": name,
                        "type": exists["type"],
                        "columns": [
                            {
                                "name": column["name"],
                                "type": column["type"],
                                "not_null": bool(column["notnull"]),
                                "primary_key": bool(column["pk"]),
                            }
                            for column in columns
                        ],
                    }
                )
        return result
