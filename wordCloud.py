"""
Script to grab wikipedia content from pages and create wordclouds
Jonathan Hagen
"""

import pprint
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from PIL import Image
import wikipedia as wiki
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude
from wordcloud import WordCloud, ImageColorGenerator

result = wiki.search("amazon", results=10)
print(*result, sep=', ')
title = result[1]
page = wiki.page(title)
content = page.content

d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

image_color = np.array(Image.open(os.path.join(d, "amazon.png")))
image_color = image_color[::1, ::1]

# create mask  white is "masked out"
image_mask = image_color.copy()
image_mask[image_mask.sum(axis=2) == 0] = 255

# some finesse: we enforce boundaries between colors so they get less washed out.
# For that we do some edge detection in the image
edges = np.mean([gaussian_gradient_magnitude(image_color[:, :, i] / 255., 2) for i in range(3)], axis=0)
image_mask[edges > .08] = 255

font_path = d + '/BarlowCondensed-SemiBold.ttf'
wc = WordCloud(max_words=100000, font_path=font_path, 
               mask=image_mask, max_font_size=35, random_state=1, relative_scaling=0, background_color="white")

wc.generate(content)
image_colors = ImageColorGenerator(image_color)
wc.recolor(color_func=image_colors)
plt.figure(figsize=(20, 10))
plt.tight_layout(pad=0) 
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
wc.to_file("cloud.png")
plt.show()