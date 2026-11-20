import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "large_sample_inventory.csv"
FIELDNAMES = [
    "sku",
    "product_name",
    "category",
    "stock_level",
    "daily_usage",
    "lead_time_days",
    "unit_cost",
    "supplier_name",
    "supplier_email",
    "minimum_order_quantity",
    "reorder_point",
    "clinic_id",
    "clinic_name",
    "city",
    "region",
    "bed_count",
    "vendor_rating",
    "vendor_support_contact",
]
CATEGORIES = {
    "PPE": ("nitrile gloves", "surgical masks", "face shields", "isolation gowns"),
    "Surgical": ("sterile gauze", "suture kits", "medical tape", "scalpel blades"),
    "Diagnostics": ("syringes", "test strips", "specimen cups", "thermometer covers"),
    "Infection Control": ("hand sanitizer", "disinfectant wipes", "surface cleaner", "sharps containers"),
    "Wound Care": ("adhesive bandages", "wound dressings", "elastic bandages", "saline wash"),
    "IV Therapy": ("IV tubing", "catheters", "saline bags", "alcohol swabs"),
}
CLINICS = [
    ("CLINIC-001", "Northside Community Clinic", "Seattle", "North", 50),
    ("CLINIC-002", "Downtown Family Clinic", "Seattle", "Central", 32),
    ("CLINIC-003", "Lakeshore Diagnostic Clinic", "Bellevue", "East", 24),
    ("CLINIC-004", "Harborview Specialty Center", "Tacoma", "South", 74),
    ("CLINIC-005", "Cedar Valley Health Clinic", "Everett", "North", 41),
    ("CLINIC-006", "Rainier Community Health", "Renton", "South", 36),
    ("CLINIC-007", "Cascade Regional Clinic", "Redmond", "East", 58),
    ("CLINIC-008", "Soundview Medical Center", "Bellingham", "North", 88),
    ("CLINIC-009", "Olympic Peninsula Clinic", "Port Angeles", "West", 29),
    ("CLINIC-010", "Columbia Basin Health Center", "Yakima", "Central", 63),
]
VENDORS = [
    ("Northstar Medical Supply", "orders@northstarmed.example", 4, 100, 4.7),
    ("Precision Health Goods", "sales@precisionhealth.example", 8, 25, 4.4),
    ("Evergreen Clinic Products", "service@evergreen.example", 12, 12, 4.9),
    ("Summit Surgical Partners", "orders@summitsurgical.example", 5, 40, 4.6),
    ("ClearPath Diagnostics", "support@clearpathdx.example", 9, 20, 4.5),
    ("Blue Cedar Medical", "supply@bluecedar.example", 3, 60, 4.8),
    ("Harborline Health Supply", "orders@harborline.example", 15, 10, 4.2),
    ("Pioneer Clinical Goods", "sales@pioneerclinical.example", 7, 30, 4.3),
    ("Redwood Care Distribution", "service@redwoodcare.example", 6, 50, 4.6),
    ("Meridian Medical Cooperative", "orders@meridianmed.example", 11, 15, 4.1),
    ("Valley Diagnostic Supply", "help@valleydx.example", 2, 24, 4.9),
    ("Soundside PPE Group", "team@soundsideppe.example", 10, 80, 4.4),
]


def generate_csv(output: str | Path = DEFAULT_OUTPUT, count: int = 2000, seed: int = 42) -> int:
    if count < 1:
        raise ValueError("count must be a positive integer")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    category_names = tuple(CATEGORIES)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index in range(1, count + 1):
            category = category_names[(index - 1) % len(category_names)]
            product = CATEGORIES[category][(index - 1) % len(CATEGORIES[category])]
            vendor_name, vendor_email, lead_days, minimum_quantity, rating = VENDORS[(index - 1) % len(VENDORS)]
            clinic_id, clinic_name, city, region, bed_count = CLINICS[randomizer.randrange(len(CLINICS))]
            writer.writerow(
                {
                    "sku": f"HP-SUP-{index:06d}",
                    "product_name": f"{product} - item {index:04d}",
                    "category": category,
                    "stock_level": randomizer.randint(0, 500),
                    "daily_usage": f"{randomizer.uniform(0.5, 45):.1f}",
                    "lead_time_days": lead_days,
                    "unit_cost": f"{randomizer.uniform(0.04, 85):.2f}",
                    "supplier_name": vendor_name,
                    "supplier_email": vendor_email,
                    "minimum_order_quantity": minimum_quantity,
                    "reorder_point": "",
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "city": city,
                    "region": region,
                    "bed_count": bed_count,
                    "vendor_rating": rating,
                    "vendor_support_contact": vendor_email,
                }
            )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic multi-clinic sample inventory CSV.")
    parser.add_argument("--count", type=int, default=2000, help="Number of product rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible stock and clinic assignment.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination CSV path.")
    args = parser.parse_args()
    try:
        count = generate_csv(args.output, args.count, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Generated {count} product rows at {args.output}.")


if __name__ == "__main__":
    main()
