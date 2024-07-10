# clear && manim -p -a --media_dir /tmp conv_geometry.py
from manim import *

config.frame_width = 2.5
config.frame_height = 2.5
config.pixel_height, config.pixel_width = 1024, 1024
config.frame_rate = 25

config.output_file = "_"  # solve strange name with -a option


def get_base_pyfai(center):
    return Group(
        Arrow3D([center, center, 0], [center+1, center, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([center, center, 0], [center, center-1, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([center, center, 0], [center, center, 1], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        MathTex("L_3", color=GREEN, font_size=18).move_to([center+1.2, center, 0]).rotate(5*PI/6),
        MathTex("L_2", color=GREEN, font_size=18).move_to([center+0, center-1.2, 0]).rotate(5*PI/6),
        MathTex("L_1", color=GREEN, font_size=18).move_to([center+0, center, 1.2]).rotate(5*PI/6),
        Text("PyFai", color=GREEN, font_size=18).move_to([center+0.6, center-0.6, 0]).rotate(5*PI/6),
    )


def get_base_lauetools(center):
    return Group(
        Arrow3D([center, center, 0], [center+1, center, 0], color=BLUE, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([center, center, 0], [center, center+1, 0], color=BLUE, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([center, center, 0], [center, center, 1], color=BLUE, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        MathTex("X", color=BLUE, font_size=18).move_to([center+1.2, center, 0]).rotate(5*PI/6),
        MathTex("Y", color=BLUE, font_size=18).move_to([center+0, center+1.2, 0]).rotate(5*PI/6),
        MathTex("Z", color=BLUE, font_size=18).move_to([center+0, center, 1.2]).rotate(5*PI/6),
        Text("LaueTools", color=BLUE, font_size=18).move_to([center+0.6, center+0.6, 0]).rotate(5*PI/6),
    )


def get_base_or(center):
    return Group(
        Arrow3D([center, center, 0], [center+1, center, 0], color=RED, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([center, center, 0], [center, center-1, 0], color=RED, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([center, center, 0], [center, center, 1], color=RED, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        MathTex("Y", color=RED, font_size=18).move_to([center+1.2, center, 0]).rotate(5*PI/6),
        MathTex("X", color=RED, font_size=18).move_to([center+0, center-1.2, 0]).rotate(5*PI/6),
        MathTex("Z", color=RED, font_size=18).move_to([center+0, center, 1.2]).rotate(5*PI/6),
        Text("OR", color=RED, font_size=18).move_to([center+0.6, center-0.6, 0]).rotate(5*PI/6),
    )


class IMGPyfaiLauetools(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/3, focal_distance=10, frame_center=[0.0, -0.2, 0.1])
        base_pyfai = get_base_pyfai(0)
        base_lauetools = get_base_lauetools(0.2)
        self.add(base_pyfai, base_lauetools)


class IMGLauetoolsOr(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/3, focal_distance=10, frame_center=[0.0, -0.2, 0.1])
        base_lauetools = get_base_lauetools(0.2)
        base_or = get_base_or(0)
        self.add(base_lauetools, base_or)
