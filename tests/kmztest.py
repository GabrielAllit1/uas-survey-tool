import zipfile
import xml.etree.ElementTree as ET

kmz_path = r"C:\Users\gabri\Documents\MasTec\KMZ\kmzpack\06_19_2025 Union Ridge Solar AOI.kmz"

try:
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        # List files in the KMZ
        print("Files in KMZ:", kmz.namelist())

        # Extract the main KML (usually doc.kml)
        with kmz.open('doc.kml') as kml_file:
            kml_data = kml_file.read()
            print(f"Read {len(kml_data)} bytes from doc.kml")

            # Parse KML XML
            root = ET.fromstring(kml_data)
            print("Root tag:", root.tag)

            # Optionally print first-level children tags
            for child in root:
                print("Child tag:", child.tag)

except Exception as e:
    print("Error reading KMZ:", e)
