import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from pathlib import Path


def parse_pts(file_path):
    points = []
    with open(file_path, "r") as f:
        inside_braces = False
        for line in f:
            line = line.strip()
            if line == "{":
                inside_braces = True
                continue
            elif line == "}":
                break
            elif inside_braces:
                x, y = map(float, line.split())
                points.append((x, y))
    return points

def pretty_print_xml(element):
    """Returns a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    # The replace is to remove the XML declaration that minidom adds
    return reparsed.toprettyxml(indent="  ")


def merge_data(menpo_dirpath, base_xml_filepath, output_xml_filepath):
    """
    Parses a menpo2d.txt file, converts the data to XML format,
    and appends it to a base XML file.
    """
    # --- 1. Parse the existing XML file ---
    try:
        tree = ET.parse(base_xml_filepath)
        root = tree.getroot()
        print(f"Loaded '{base_xml_filepath}' successfully.")
    except FileNotFoundError:
        print(f"Base file '{base_xml_filepath}' not found. A new file will be created.")
        root = ET.Element('images')

    menpo_images = Path(menpo_dirpath)
    for item in menpo_images.iterdir(): 
        print(f"Processing directory: {item}")
        if item.is_dir():
            for subitem in item.iterdir():
                print(f"Processing subdirectory: {subitem}")
                if subitem.is_dir():
                    for pic in subitem.iterdir():
                        if pic.is_file() and pic.suffix == ".pts":
                            print(f"Processing file: {pic}")
                            landmarks = parse_pts(pic)
                            if len(landmarks) != 68:
                                print(f"Skipping {pic.name} because it has {len(landmarks)} landmarks.")
                                continue
                            img_path = subitem.with_suffix('.jpg')
                            image_elem = ET.Element('image', {
                            'file': img_path + '.jpg',
                            'width': '0',
                            'height': '0'
                            })
                            box_elem = ET.SubElement(image_elem, 'box', {
                            'top': str(int(0)),
                            'left': str(int(0)),
                            'width': str(int(0)),
                            'height': str(int(0))
                            })
                            for i, (x, y) in enumerate(landmarks):
                                 # Format the landmark name with a leading zero, e.g., '01', '02'
                                part_name = f"{i + 1:02d}"
                                ET.SubElement(box_elem, 'part', {
                                    'name': part_name,
                                    'x': str(int(x)),
                                    'y': str(int(y))
                                })      
                            root.append(image_elem) 
                            print(pretty_print_xml(root))         
                            print(f"Appended '{pic.name}' to the XML file.")
                            break
                if subitem.is_file() and subitem.suffix == ".pts":
                    print(f"Processing file: {subitem}")
                    img_path = subitem.with_suffix('.jpg')
                    landmarks = parse_pts(subitem)
                    if len(landmarks) != 68:
                        print(f"Skipping {pic.name} because it has {len(landmarks)} landmarks.")
                        
                        continue
                    image_elem = ET.Element('image', {
                    'file': img_path,
                    'width': '0',
                    'height': '0'
                    })
                    box_elem = ET.SubElement(image_elem, 'box', {
                    'top': str(int(0)),
                    'left': str(int(0)),
                    'width': str(int(0)),
                    'height': str(int(0))
                    })
                    for i, (x, y) in enumerate(landmarks):
                        part_name = f"{i + 1:02d}"
                        ET.SubElement(box_elem, 'part', {
                            'name': part_name,
                            'x': str(int(x)),
                            'y': str(int(y))
                        })      
                        root.append(image_elem) 
                        print(pretty_print_xml(root))         
                        print(f"Appended '{clean_name}' to the XML file.")
                        
                    
    
    # with open(menpo_filepath, 'r') as f:
    #     for line in f:
    #         parts = line.strip().split()
    #         if not parts:
    #             continue

    #         # Extract filepath, bounding box, and landmark coordinates
    #         filepath = parts[0]
            
    #         coords = [float(p) for p in parts[2:]]
    #         landmarks = list(zip(coords[0::2], coords[1::2]))

    #         # --- 3. Build the new XML elements ---
    #         # NOTE: Image width and height are not in menpo2d.txt, so we use '0'.
    #         image_elem = ET.Element('image', {
    #             'file': filepath,
    #             'width': '0',
    #             'height': '0'
    #         })

            # box_elem = ET.SubElement(image_elem, 'box', {
            #     'top': str(int(0)),
            #     'left': str(int(0)),
            #     'width': str(int(0)),
            #     'height': str(int(0))
            # })

    #         # Create a <part> element for each landmark
    #         for i, (x, y) in enumerate(landmarks):
    #             # Format the landmark name with a leading zero, e.g., '01', '02'
    #             part_name = f"{i + 1:02d}"
    #             ET.SubElement(box_elem, 'part', {
    #                 'name': part_name,
    #                 'x': str(int(x)),
    #                 'y': str(int(y))
    #             })
            
    #         # --- 4. Append the newly created <image> element to the root ---
    #         root.append(image_elem)

    # # --- 5. Write the combined and formatted XML to the output file ---
    # with open(output_xml_filepath, "w") as f:
    #     f.write(pretty_print_xml(root))

    # print(f"\nSuccessfully merged data into '{output_xml_filepath}'.")


if __name__ == "__main__":
    # Define file paths
    base_xml = "eyes_only.xml"
    menpo_data = "image"
    output_xml = "eyes_only_updated1.xml"
    
    # Run the merging process
    merge_data(menpo_data, base_xml, output_xml)

    # # Optional: Print the content of the new file to verify
    # print("\n--- Contents of the new 'eyes_only_updated.xml': ---")

    # eyes = {
    #     "01",
    #     "09",
    #     "17",
    #     "31",
    #     "32",
    #     "36",
    #     "37",
    #     "38",
    #     "39",
    #     "40",
    #     "41",
    #     "42",
    #     "43",
    #     "44",
    #     "45",
    #     "46",
    #     "47",
    #     "48"
    # }

    # tree = ET.parse("eyes_only_updated.xml")
    # root = tree.getroot()

    # for box in root.findall("./images/image/box"):
    #     parts = box.findall(".//part[@name]")
    #     for part in parts:
    #         if part.get("name") not in eyes:
    #             print("Removing:", ET.tostring(part))
    #             box.remove(part)  # ✅ Remove from parent, not from tree
    #         else:
    #             continue

    # tree.write("converted.xml")



