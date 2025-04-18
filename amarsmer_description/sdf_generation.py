import json
from lxml import etree
import numpy as np
import pathlib

'''
This code reads a json file and converts it to sdf.
'''

# --- Load poses from JSON ---
current_path = str(pathlib.Path(__file__).parent.resolve())

with open(current_path + "/poses.json", "r") as f:
    poses = json.load(f)

# --- Create SDF ---
def create_sdf_for_path(poses, model_radius=0.05):
    sdf = etree.Element("sdf", version="1.6")
    world = etree.SubElement(sdf, "world", name="default")

    for i, pose in enumerate(poses[::5]):  # Sample every n-th
        model = etree.SubElement(world, "model", name=f"path_point_{i}")
        etree.SubElement(model, "static").text = "true"

        p = pose["position"]
        o = pose["orientation"]
        yaw = np.arctan2(2 * (o['w'] * o['z']), 1 - 2 * (o['z'] ** 2))
        pose_str = f"{p[0]} {p[1]} {p[2]} 0 0 {yaw}"
        etree.SubElement(model, "pose").text = pose_str

        link = etree.SubElement(model, "link", name="link")
        visual = etree.SubElement(link, "visual", name="visual")
        geometry = etree.SubElement(visual, "geometry")
        sphere = etree.SubElement(geometry, "sphere")
        etree.SubElement(sphere, "radius").text = str(model_radius)

        material = etree.SubElement(visual, "material")
        etree.SubElement(material, "ambient").text = "0 0 1 1"
        etree.SubElement(material, "diffuse").text = "0 0 1 1"

    return etree.tostring(sdf, pretty_print=True, encoding="unicode")

# --- Save SDF to file ---
sdf_content = create_sdf_for_path(poses)
with open(current_path + "/urdf/path_markers.sdf", "w") as f:
    f.write(sdf_content)
