import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.db import Database
from app.errors import DataImportError
from app.schemas import SeedProductRow
from app.tools import calculate_reorder_point

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "sample_inventory.csv"


def import_csv(database: Database, csv_path: str | Path) -> int:
    path = Path(csv_path)
    if not path.is_file():
        raise DataImportError(f"CSV file not found: {path}")
    database.initialize()
    count = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {
                "sku",
                "product_name",
                "stock_level",
                "daily_usage",
                "lead_time_days",
                "unit_cost",
                "supplier_name",
                "supplier_email",
                "minimum_order_quantity",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise DataImportError(
                    "CSV is missing required columns: " + ", ".join(sorted(required - set(reader.fieldnames or [])))
                )
            for line_number, raw in enumerate(reader, start=2):
                try:
                    if not (raw.get("reorder_point") or "").strip():
                        raw["reorder_point"] = None
                    row = SeedProductRow.model_validate(raw)
                    cents = int(
                        (Decimal(str(row.unit_cost)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    )
                    reorder_point = row.reorder_point
                    if reorder_point is None:
                        reorder_point = calculate_reorder_point(
                            row.stock_level, row.daily_usage, row.lead_time_days
                        )
                    database.upsert_inventory(
                        sku=row.sku,
                        product_name=row.product_name,
                        stock_level=row.stock_level,
                        daily_usage=row.daily_usage,
                        lead_time_days=row.lead_time_days,
                        reorder_point=reorder_point,
                        supplier_name=row.supplier_name,
                        supplier_email=row.supplier_email,
                        minimum_order_quantity=row.minimum_order_quantity,
                        unit_cost_cents=cents,
                        category=row.category,
                        clinic_id=row.clinic_id,
                        clinic_name=row.clinic_name,
                        city=row.city,
                        region=row.region,
                        bed_count=row.bed_count,
                        vendor_rating=row.vendor_rating,
                        vendor_support_contact=row.vendor_support_contact,
                    )
                except (ValidationError, InvalidOperation, ValueError) as exc:
                    raise DataImportError(f"Invalid CSV row {line_number}: {exc}") from exc
                count += 1
    except UnicodeDecodeError as exc:
        raise DataImportError(f"CSV must be UTF-8 encoded: {exc}") from exc
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import products and supplier terms from CSV into SQLite.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to input CSV (default: bundled sample).")
    parser.add_argument("--database", default=str(Settings().database_path), help="SQLite database path.")
    args = parser.parse_args()
    try:
        count = import_csv(Database(args.database), args.csv)
    except DataImportError as exc:
        parser.error(exc.message)
    print(f"Imported {count} product rows from {args.csv}.")


if __name__ == "__main__":
    main()
