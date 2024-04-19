import matplotlib.pyplot as plt

from PIL import Image
import xmltodict

from pyproj import Proj, transform


# Function to parse KML and extract coordinates


def parse_kml(kml_path):
    with open(kml_path, "r") as file:
        doc = xmltodict.parse(file.read())

    coordinates = doc["kml"]["Document"]["Placemark"]["LineString"]["coordinates"]

    coord_list = coordinates.strip().split(" ")

    coord_pairs = [tuple(map(float, coord.split(","))) for coord in coord_list]

    return coord_pairs


# Function to transform coordinates


def transform_coords(coord_pairs, src_proj, dest_proj):
    transformer = Proj(projparams=dest_proj)  # Destination projection

    projected_coords = [
        transform(Proj(projparams=src_proj), transformer, lon, lat)
        for lon, lat, _height in coord_pairs
    ]

    return projected_coords


# Load the orthophoto

image_path = "Screenshot 2024-04-19 003054.png"

img = Image.open(image_path)
print(img.size[0])

# Parse the KML file

kml_path = "For python.kml"

coord_pairs = parse_kml(kml_path)
# print(coord_pairs)


# Project coordinates (Example: WGS84 to a projected system)

# This needs to match your orthophoto's projection

src_proj = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

dest_proj = "+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"  # Change this to match your needs
projected_coords = transform_coords(coord_pairs, src_proj, dest_proj)


# Plotting

fig, ax = plt.subplots()

ax.imshow(img)
print(
    *[
        img.size[0]
        * (x - min(px for px, py in projected_coords))
        / (
            max(px for px, py in projected_coords)
            - min(px for px, py in projected_coords)
        )
        for x, y in projected_coords
    ]
)

min_x = min(px for px, py in projected_coords)
max_x = max(px for px, py in projected_coords)
min_y = min(py for px, py in projected_coords)
max_y = max(py for px, py in projected_coords)

# Normalize and scale coordinates to the image dimensions
x_pixels, y_pixels = zip(
    *[
        (
            img.size[0] * (x - min_x) / (max_x - min_x) if max_x > min_x else 0,
            img.size[1] * (y - min_y) / (max_y - min_y) if max_y > min_y else 0,
        )
        for x, y in projected_coords
    ]
)


ax.plot(x_pixels, y_pixels, "r-", linewidth=2)  # Draw in red

plt.show()
