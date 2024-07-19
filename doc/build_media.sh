#!/bin/bash

python media/geometry_graph.py

find media/ -type f -name "*.py" -exec manim -a --media_dir /tmp {} \;
mkdir -p build/media/
find /tmp/videos -type f -name "ANIM*.mp4" -exec sh -c \
    'ffmpeg -y -i {} -pix_fmt yuv420p10le -filter:v scale="trunc(ih/4)*2:trunc(iw/4)*2" \
    -c:v libsvtav1 -crf 48 -preset 6 -svtav1-params tune=2:scd=0:film-grain=0:keyint=5s:fast-decode=1:irefresh-type=2 \
    build/media/$(basename {} .mp4).webm' \;
find /tmp/images -type f -name "IMG*.png" -exec sh -c \
    'ffmpeg -y -i {} -pix_fmt yuv444p12le \
    -c:v libaom-av1 -tune ssim -crf 35 -cpu-used 1 -denoise-noise-level 0 -still-picture 1 \
    build/media/$(basename {} .png).avif' \;
