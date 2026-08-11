import pandas as pd

from src.preprocessing import CarFeatureEngineer, FEATURE_COLUMNS


def test_feature_engineering_counts_distinct_items_and_adds_ratios():
    record = {column: 1 for column in FEATURE_COLUMNS}
    for column in [
        "make_model", "body_type", "vat", "Type", "Fuel", "Paint_Type",
        "Upholstery_type", "Gearing_Type", "Drive_chain",
    ]:
        record[column] = "example"
    record["Comfort_Convenience"] = "Air conditioning,Air conditioning,Armrest"
    record["Entertainment_Media"] = "Bluetooth,Radio"
    record["Extras"] = ""
    record["Safety_Security"] = None
    record["km"] = 30_000
    record["age"] = 2
    record["hp_kW"] = 90
    record["Weight_kg"] = 1_200

    transformed = CarFeatureEngineer().fit_transform(pd.DataFrame([record]))

    assert transformed.loc[0, "num_comfort_convenience"] == 2
    assert transformed.loc[0, "num_entertainment_media"] == 2
    assert transformed.loc[0, "num_extras"] == 0
    assert transformed.loc[0, "num_safety_security"] == 0
    assert transformed.loc[0, "km_per_year"] == 10_000
    assert transformed.loc[0, "power_to_weight"] == 75
