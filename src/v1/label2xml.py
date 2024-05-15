import os
import xml.etree.ElementTree as ET

__all__ = ['label2xml']

import cv2
import subprocess

import numpy as np

from ai_pytorch_object_detection.config import TRAIN_DIR, VALID_DIR


def _remove_xml_files(directory_path: str | list[str]):
    try:
        if isinstance(directory_path, str):
            # Construct the shell command
            command = f'find {directory_path} -name "*.xml" -type f -print0 | xargs -0 rm -f'

            # Execute the command using subprocess
            subprocess.run(command, shell=True, check=True)
            print(f'XML files in {directory_path} removed successfully.')
        if isinstance(directory_path, list):
            for path in directory_path:
                _remove_xml_files(path)
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')


def _convert_to_xml(**kwargs) -> bool:
    """
    Converts label to xml format and returns it.
    :param kwargs: dictionary of arguments to be converted. It should contain the following keys
    :return:
    """
    # Create the root element
    annotation = ET.Element("annotation")

    # Add sub-elements to the root
    folder = ET.SubElement(annotation, "folder")
    folder.text = kwargs["folder"]
    filename = ET.SubElement(annotation, "filename")
    filename.text = kwargs["filename"]
    path = ET.SubElement(annotation, "path")
    path.text = kwargs["path"]

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = kwargs["database"]

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = kwargs["width"]
    height = ET.SubElement(size, "height")
    height.text = kwargs["height"]
    depth = ET.SubElement(size, "depth")
    depth.text = kwargs["depth"]

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = kwargs["segmented"]
    object_elem = ET.SubElement(annotation, "object")
    name = ET.SubElement(object_elem, "name")
    name.text = kwargs["name"]
    pose = ET.SubElement(object_elem, "pose")
    pose.text = kwargs["pose"]
    truncated = ET.SubElement(object_elem, "truncated")
    truncated.text = str(kwargs["truncated"])
    difficult = ET.SubElement(object_elem, "difficult")
    difficult.text = str(kwargs["difficult"])

    bndbox = ET.SubElement(object_elem, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = kwargs["xmin"]
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = kwargs["ymin"]
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = kwargs["xmax"]
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = kwargs["ymax"]

    # Create the XML tree
    tree = ET.ElementTree(annotation)

    # Save the XML to a file
    tree.write(f"{kwargs['save_folder']}{kwargs['filename']}.xml")


def _reverse_normalize(data, resolution):
    x_center_normalized, y_center_normalized, width_normalized, height_normalized = np.float32(data)
    # Получаем целочисленные координаты
    xmin = int(np.round(x_center_normalized * resolution[1]))
    ymin = int(np.round(y_center_normalized * resolution[0]))
    xmax = int(np.round((x_center_normalized + width_normalized) * resolution[1]))
    ymax = int(np.round((y_center_normalized + height_normalized) * resolution[0]))

    return [xmin, ymin, xmax, ymax]


def label2xml(path: list[str] = [TRAIN_DIR, VALID_DIR], path_to_save: str = "../save"):
    """
    Converts label to xml format and returns it.
    """

    for j in range(len(path)):
        img_path = f'{path[j]}/images/'
        label_path = f'{path[j]}/labels/'
        _remove_xml_files([img_path, label_path])
        img_files = os.listdir(img_path)
        num_img_files = len(img_files)

        label_files = os.listdir(label_path)
        num_label_files = len(label_files)

        print(f'Количество файлов в папке с метками: {num_label_files}')
        print(f'Количество файлов в папке с изображениями: {num_img_files}')
        for i, (img_file, label_file) in enumerate(zip(img_files, label_files)):
            img = cv2.imread(os.path.join(img_path, img_file))

            with open(os.path.join(label_path, label_file)) as f:
                data = f.readlines()
                if len(data) > 0 and img is not None:
                    data = data[0].split()
                    # xmin,ymin,xmax,ymax = data[1:5]
                    xmin, ymin, xmax, ymax = _reverse_normalize(data[1:5], img.shape)
                    params = {
                        "count": i,
                        "folder": "images",
                        "filename": img_file[:-4],
                        "path": path[j],
                        "database": "Unknown",
                        "width": str(img.shape[1]),
                        "height": str(img.shape[0]),
                        "depth": str(img.shape[2]),
                        "segmented": "0",
                        "name": "FPV_DRONE",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "difficult": "0",
                        "xmin": str(xmin),
                        "ymin": str(ymin),
                        "xmax": str(xmax),
                        "ymax": str(ymax),
                        "save_folder": os.path.join(img_path)
                    }
                    _convert_to_xml(**params)
                else:
                    import shutil

                    # Create a new directory
                    new_dir = f'./_{path[j]}_bad_jpg/'
                    os.makedirs(new_dir, exist_ok=True)

                    # Move a file into the new directory
                    old_file_path = os.path.join(img_path, img_file)
                    new_file_path = os.path.join(new_dir, img_file)
                    shutil.move(old_file_path, new_file_path)


label2xml()
