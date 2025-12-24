# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MaskGenerator.py
# 2025/12/24

import os
import numpy as np
import shutil
import cv2
import glob
import shutil
import json
import traceback

class MaskGenerator:
  def __init__(self):
    self.RGB_COLORS = { "Missing part": (19, 164, 201),
                        "Broken part": (166, 255, 71),
                        "Scratch": (180, 45, 56),
                        "Cracked": (225, 150, 96),
                        "Dent": (144, 60, 89),
                        "Flaking": (167, 116, 27),
                        "Paint chip":(255,0, 255),
                        "Corrosion": (115, 194, 206)}

  def generate(self, annotations_dir, output_dir):
    json_files = glob.glob(annotations_dir + "/*.json")
    for json_file in json_files:
        print(json_file)

        basename = os.path.basename(json_file)
        mask_pngfilename = basename.replace(".json", "")
        with open(json_file, 'r') as f:
           
           json_data = json.load(f)
           objects = json_data["objects"]
           size = json_data["size"]
           h = size["height"]
           w = size["width"]
           mask = np.zeros((h, w, 3), dtype=np.uint8)
           for object in objects:
               title = object["classTitle"]
               color = None
               try:
                 color = self.RGB_COLORS[title]
               except:
                print("No found color to {}".format(title) ) 
                input("Error")
                continue
                 
               (r, g, b) = color
               bgr_color = (b, g, r)
               points = object["points"]["exterior"]
               pt = []
               for point in points:
                  [x, y] = point
                  x = int(x)
                  y = int(y)
                  pt.append([x, y])
               pts = np.array(pt)

               cv2.fillConvexPoly(mask, points =pts, color=(bgr_color))
        output_mask_filepath = os.path.join(output_dir, mask_pngfilename)
        cv2.imwrite(output_mask_filepath, mask)
        print("Save {}".format(output_mask_filepath))

if __name__ == "__main__":
  try:
    annotations_dir = "./Car parts dataset/File1/ann/"
    output_dir      = "./Car-Damages/masks/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    generator = MaskGenerator()
    generator.generate(annotations_dir, output_dir)

  except:
    traceback.print_exc()
