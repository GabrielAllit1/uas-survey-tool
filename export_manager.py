import os
import logging
import csv
import zipfile
from simplekml import Kml
import shutil
import tempfile

def export_to_csv(data, filename, fields):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        logging.debug(f"Exported CSV: {filename}")
    except Exception as e:
        logging.error(f"CSV export failed for {filename}: {e}")
        raise

def export_to_combined_csv(control_points, verification_shots, filename):
    try:
        fields = ['Type', 'Name', 'Latitude', 'Longitude', 'Elevation', 'Pattern_Score', 'Global_Offset']
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for cp in control_points:
                writer.writerow({
                    'Type': 'Control Point',
                    'Name': cp['name'],
                    'Latitude': cp['lat'],
                    'Longitude': cp['lon'],
                    'Elevation': cp.get('elevation', 0.0),
                    'Pattern_Score': cp.get('pattern_score', 0.0),
                    'Global_Offset': cp.get('global_offset', 0.0)
                })
            for vs in verification_shots:
                writer.writerow({
                    'Type': 'Verification Shot',
                    'Name': vs['name'],
                    'Latitude': vs['lat'],
                    'Longitude': vs['lon'],
                    'Elevation': vs.get('elevation', 0.0),
                    'Pattern_Score': vs.get('pattern_score', 0.0),
                    'Global_Offset': vs.get('global_offset', 0.0)
                })
        logging.debug(f"Exported combined CSV: {filename}")
    except Exception as e:
        logging.error(f"Combined CSV export failed for {filename}: {e}")
        raise

def export_to_kmz(control_points, verification_shots, filename):
    try:
        kml = Kml()
        control_folder = kml.newfolder(name="Control Points")
        verification_folder = kml.newfolder(name="Verification Shots")

        for cp in control_points:
            point = control_folder.newpoint(name=cp['name'])
            point.coords = [(cp['lon'], cp['lat'], cp.get('elevation', 0.0))]
            point.description = f"Pattern Score: {cp.get('pattern_score', 0.0):.2f}"

        for vs in verification_shots:
            point = verification_folder.newpoint(name=vs['name'])
            point.coords = [(vs['lon'], vs['lat'], vs.get('elevation', 0.0))]
            point.description = f"Pattern Score: {vs.get('pattern_score', 0.0):.2f}"

        temp_dir = tempfile.mkdtemp()
        kml_file = os.path.join(temp_dir, 'doc.kml')
        kml.save(kml_file)

        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as kmz:
            kmz.write(kml_file, 'doc.kml')

        shutil.rmtree(temp_dir)
        logging.debug(f"Exported KMZ: {filename}")
    except Exception as e:
        logging.error(f"KMZ export failed for {filename}: {e}")
        raise