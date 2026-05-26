import os
import xml.etree.ElementTree as ET

# Paths
annotations_path = "dataset/annotations"
images_path = "dataset/images"

labels_path = "dataset/labels"

os.makedirs(labels_path, exist_ok=True)

# Class mapping
class_map = {
    "helmet": 0,
    "head": 1,
    "person": 2
}

# Convert XML to YOLO
for xml_file in os.listdir(annotations_path):

    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotations_path, xml_file))
    root = tree.getroot()

    size = root.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    label_file = os.path.join(
        labels_path,
        xml_file.replace(".xml", ".txt")
    )

    with open(label_file, "w") as f:

        for obj in root.findall("object"):

            class_name = obj.find("name").text

            if class_name not in class_map:
                continue

            class_id = class_map[class_name]

            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            f.write(
                f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
            )

print("Conversion completed successfully.")