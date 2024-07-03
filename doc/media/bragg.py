# clear && manim -p -a --media_dir /tmp bragg.py
from manim import *

config.frame_width = 2.5
config.frame_height = 2.5
config.pixel_height, config.pixel_width = 1024, 1024
config.frame_rate = 25

config.output_file = "_"  # solve strage name with -a option


def get_base():
    return Group(
        Arrow3D([0, 0, 0], [1, 0, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([0, 0, 0], [0, 1, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([0, 0, 0], [0, 0, 1], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        MathTex("L_1", color=GREEN, font_size=18).move_to([1.2, 0, 0]).rotate(3*PI/4),
        MathTex("L_2", color=GREEN, font_size=18).move_to([0, 1.2, 0]).rotate(3*PI/4),
        MathTex("L_3", color=GREEN, font_size=18).move_to([0, 0, 1.2]).rotate(3*PI/4),
    )


class IMGBragg(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/4, focal_distance=10, frame_center=[0.0, 0.0, 0.1])
        base = get_base()
        u_i = Group(
            Arrow3D([0, 0, -1], [0, 0, 0], color=YELLOW, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
            MathTex("u_i", color=YELLOW, font_size=18).move_to([0, 0, -1.2]).rotate(3*PI/4),
        )
        self.add(base, u_i)
