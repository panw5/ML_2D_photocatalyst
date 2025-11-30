from ase.db import connect

def inspect_db(db_path, max_rows=5):
    """
    Read a .db file and display key information.

    Parameters:
        db_path (str): Path to the .db file
        max_rows (int): Number of records to preview
    """
    try:
        db = connect(db_path)
        print(f"The database contains {len(db)} records.")
        print("Available fields:")

        # Print all fields (from the first record only)
        for row in db.select():
            print("Field list:", list(row.key_value_pairs.keys()))
            break

        print("\nPreview of the first few records:")
        for i, row in enumerate(db.select()):
            print(f"\n--- Record {i+1} ---")
            print("ID:", row.id)
            print("Formula:", row.formula)
            print("Key-Value Pairs:", row.key_value_pairs)
            if i + 1 >= max_rows:
                break

    except Exception as e:
        print(f"Failed to read database: {e}")

# Example call
inspect_db("../data/c2db-2.db", max_rows=10)
