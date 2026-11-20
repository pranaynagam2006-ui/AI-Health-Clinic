import argparse

from app.config import Settings
from app.retrieval import ingest_policy_folder


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Index local supplier and purchasing policy documents.")
    parser.add_argument("--folder", default=str(settings.policy_folder), help="Folder containing .md and .txt files.")
    parser.add_argument("--store", default=str(settings.policy_store_path), help="Chroma persistence directory.")
    args = parser.parse_args()
    count = ingest_policy_folder(args.folder, args.store)
    print(f"Indexed {count} policy document(s) in {args.store}.")


if __name__ == "__main__":
    main()
