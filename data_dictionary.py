import csv
import os
from datetime import datetime

def generate_data_dictionary(output_folder, tool_version="2.0"):
    dictionary_path = os.path.join(output_folder, "Data_Dictionary.csv")
    
    headers = [
        "Field Name", "Description", "Data Type", "Units", "Example"
    ]

    rows = [
        ("Point ID", "Unique identifier for GCP or verification point", "String", "-", "GCP-001"),
        ("Type", "Type of point (GCP or VER)", "String", "-", "GCP"),
        ("Easting", "UTM Easting coordinate", "Float", "meters", "421593.12"),
        ("Northing", "UTM Northing coordinate", "Float", "meters", "3074579.45"),
        ("Latitude", "Latitude in decimal degrees", "Float", "degrees", "27.941234"),
        ("Longitude", "Longitude in decimal degrees", "Float", "degrees", "-82.456789"),
        ("Zone", "UTM Zone Number", "Integer", "-", "17"),
        ("Hemisphere", "UTM Hemisphere (N/S)", "String", "-", "N"),
        ("Elevation", "Elevation from DEM/DSM (if available)", "Float", "meters", "12.34"),
        ("Spacing (m)", "Nominal spacing from layout engine", "Integer", "meters", "400"),
        ("Layout Pattern", "Pattern used for GCP distribution", "String", "-", "Fractal"),
        ("Pattern Score", "Complexity score from sequence detection", "Float", "-", "0.75"),
        ("Flight Timings", "Cyclic timing for multi-pass surveys", "String", "minutes", "1.2,2.5,3.8"),
        ("Date Generated", "Date of export", "Date", "-", datetime.now().strftime("%Y-%m-%d"))
    ]

    with open(dictionary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["UAS Survey Tool Data Dictionary (v{})".format(tool_version)])
        writer.writerow([])
        writer.writerow(headers)
        writer.writerows(rows)

    return dictionary_path