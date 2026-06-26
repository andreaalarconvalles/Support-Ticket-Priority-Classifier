"""Text cleaning + tokenisation pipeline for the Support Ticket Priority Classifier.

Phase 3 (Person 2 support / Juan Jose ML): clean -> tokenise -> pad -> encode labels.
"""

import re

import pandas as pd

PLACEHOLDER_PATTERN = re.compile(r"\{[a-z_]+\}")


def fill_product_placeholder(description: str, product: str) -> str:
    """Replace the literal '{product_purchased}' token with the real product name.

    The raw dataset is template-generated and every row has at least one unfilled
    placeholder (see backend/eda.ipynb, section 4b) - the most common by far is
    '{product_purchased}', which has the real value sitting in the Product Purchased
    column of the same row.
    """
    return description.replace("{product_purchased}", product)


def strip_unfilled_placeholders(description: str) -> str:
    """Remove any remaining '{placeholder}' tokens that have no real value to fill in.

    Unlike {product_purchased}, things like {error_message} and {product_id} have no
    backing column in the dataset - there's no ground truth to substitute, so the only
    honest option is to drop the token. Covers the full long tail of ~150 placeholder
    variants found in eda.ipynb, not just these two.
    """
    description = PLACEHOLDER_PATTERN.sub("", description)
    return re.sub(r"\s{2,}", " ", description).strip()


def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the 'Ticket Description' column, returning a copy."""
    df = df.copy()
    df["Ticket Description"] = df.apply(
        lambda row: fill_product_placeholder(row["Ticket Description"], row["Product Purchased"]),
        axis=1,
    )
    df["Ticket Description"] = df["Ticket Description"].apply(strip_unfilled_placeholders)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/customer_support_tickets.csv")

    before = df["Ticket Description"].str.contains(PLACEHOLDER_PATTERN).sum()
    df = clean_descriptions(df)
    after = df["Ticket Description"].str.contains(PLACEHOLDER_PATTERN).sum()

    print(f"Rows with any literal '{{placeholder}}' before cleaning: {before}")
    print(f"Rows with any literal '{{placeholder}}' after cleaning:  {after}")
    print()
    print("Example:")
    print(df.loc[0, "Ticket Description"])
