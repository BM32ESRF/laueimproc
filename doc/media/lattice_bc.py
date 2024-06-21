# clear && manim -p -a --media_dir /tmp lattice_bc.py
from manim import *
import torch

from laueimproc.diffraction.lattice import lattice_to_primitive

config.frame_width = 2.5
config.frame_height = 2.5
config.pixel_height, config.pixel_width = 1024, 1024
config.frame_rate = 25

config.output_file = "_"  # solve strage name with -a option


def get_lattice(a, b, c, alpha, beta, gamma):
    lattice = torch.tensor([a, b, c, alpha, beta, gamma])
    prim = lattice_to_primitive(lattice)
    ((e1x, e2x, e3x), (e1y, e2y, e3y), (e1z, e2z, e3z)) = prim.tolist()
    vertex = [
        [0, 0, 0],
        [e1x, e1y, e1z],
        [e1x+e2x, e1y+e2y, e1z+e2z],
        [e2x, e2y, e2z],
        [e3x, e3y, e3z],
        [e3x+e1x, e3y+e1y, e3z+e1z],
        [e3x+e1x+e2x, e3y+e1y+e2y, e3z+e1z+e2z],
        [e3x+e2x, e3y+e2y, e3z+e2z],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],
        [1, 0, 4], [1, 4, 5],
        [1, 2, 6], [1, 5, 6],
        [2, 3, 7], [2, 6, 7],
        [0, 3, 7], [0, 4, 7],
        [4, 5, 6], [4, 7, 6],
    ]
    surface = Polyhedron(vertex, faces,
        faces_config={"fill_opacity": 0.25, "stroke_opacity": 1, "stroke_width": 0, "color": BLUE},
        graph_config={
            "edge_config": {"stroke_opacity": 0},
            "vertex_config": {"radius": 0},
        },
    )

    e1 = Arrow3D([0, 0, 0], [e1x, e1y, e1z], color=RED, resolution=16, thickness=0.01, height=0.1, base_radius=0.04)
    e2 = Arrow3D([0, 0, 0], [e2x, e2y, e2z], color=RED, resolution=16, thickness=0.01, height=0.1, base_radius=0.04)
    e3 = Arrow3D([0, 0, 0], [e3x, e3y, e3z], color=RED, resolution=16, thickness=0.01, height=0.1, base_radius=0.04)
    e1_text = MathTex("e_1", color=RED, font_size=18).move_to([1.1*e1x, 1.1*e1y, 1.1*e1z + 0.05]).rotate(3*PI/4)
    e2_text = MathTex("e_2", color=RED, font_size=18).move_to([1.1*e2x, 1.1*e2y, 1.1*e2z + 0.05]).rotate(3*PI/4)
    e3_text = MathTex("e_3", color=RED, font_size=18).move_to([1.1*e3x, 1.1*e3y, 1.1*e3z + 0.05]).rotate(3*PI/4)

    a_text = MathTex("a", color=YELLOW, font_size=18).move_to([0.6*e1x + 0.05, 0.6*e1y + 0.05, 0.6*e1z + 0.05]).rotate(3*PI/4)
    b_text = MathTex("b", color=YELLOW, font_size=18).move_to([0.6*e2x + 0.05, 0.6*e2y + 0.05, 0.6*e2z + 0.05]).rotate(3*PI/4)
    c_text = MathTex("c", color=YELLOW, font_size=18).move_to([0.6*e3x + 0.05, 0.6*e3y + 0.05, 0.6*e3z + 0.05]).rotate(3*PI/4)
    alpha_text = MathTex(r"\alpha", color=YELLOW, font_size=18).move_to([0.3*(e2x + e3x), 0.3*(e2y + e3y), 0.3*(e2z + e3z)]).rotate(3*PI/4)
    beta_text = MathTex(r"\beta", color=YELLOW, font_size=18).move_to([0.3*(e1x + e3x), 0.3*(e1y + e3y), 0.3*(e1z + e3z)]).rotate(3*PI/4)
    gamma_text = MathTex(r"\gamma", color=YELLOW, font_size=18).move_to([0.3*(e1x + e2x), 0.3*(e1y + e2y), 0.3*(e1z + e2z)]).rotate(3*PI/4)

    # shape (n, 3)
    e1_vec = np.asarray((e1x, e1y, e1z))
    e2_vec = np.asarray((e2x, e2y, e2z))
    e3_vec = np.asarray((e3x, e3y, e3z))
    arc_alpha = np.asarray([(t*e2_vec + (1-t)*e3_vec) for t in np.linspace(0, 1, 16)])
    arc_alpha *= 0.2 / np.sqrt(np.sum(arc_alpha*arc_alpha, axis=1, keepdims=True))
    arc_alpha = Polygon(*arc_alpha, [0, 0, 0], color=YELLOW, stroke_width=0.5)
    arc_beta = np.asarray([(t*e1_vec + (1-t)*e3_vec) for t in np.linspace(0, 1, 16)])
    arc_beta *= 0.2 / np.sqrt(np.sum(arc_beta*arc_beta, axis=1, keepdims=True))
    arc_beta = Polygon(*arc_beta, [0, 0, 0], color=YELLOW, stroke_width=0.5)
    arc_gamma = np.asarray([(t*e1_vec + (1-t)*e2_vec) for t in np.linspace(0, 1, 16)])
    arc_gamma *= 0.2 / np.sqrt(np.sum(arc_gamma*arc_gamma, axis=1, keepdims=True))
    arc_gamma = Polygon(*arc_gamma, [0, 0, 0], color=YELLOW, stroke_width=0.5)

    return (
        Group(surface, e1, e2, e3, e1_text, e2_text, e3_text),
        Group(
            a_text, b_text, c_text,
            Group(alpha_text, arc_alpha), Group(beta_text, arc_beta), Group(gamma_text, arc_gamma),
        ),
    )


def get_base():
    return Group(
        Arrow3D([0, 0, 0], [1, 0, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([0, 0, 0], [0, 1, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([0, 0, 0], [0, 0, 1], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        MathTex("C_1", color=GREEN, font_size=18).move_to([1.2, 0, 0]).rotate(3*PI/4),
        MathTex("C_2", color=GREEN, font_size=18).move_to([0, 1.2, 0]).rotate(3*PI/4),
        MathTex("C_3", color=GREEN, font_size=18).move_to([0, 0, 1.2]).rotate(3*PI/4),
    )


class IMGLattice(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/4, focal_distance=10, frame_center=[0.0, 0.0, 0.1])
        lattice_ref, text_ref = get_lattice(1.0, 1.0, 1.0, 75*DEGREES, 90*DEGREES, 75*DEGREES)
        self.add(lattice_ref, text_ref)


class IMGLatticeBc(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/4, focal_distance=10, frame_center=[0.0, 0.0, 0.1])
        base = get_base()
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, 75*DEGREES, 90*DEGREES, 75*DEGREES)
        self.add(base, lattice_ref, text_ref)


class ANIMLatticeBc(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=35*DEGREES, focal_distance=10, frame_center=[0.0, 0.0, 0.1])
        base = get_base()
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.add(base, lattice_ref)

        self.begin_ambient_camera_rotation(20*DEGREES/9, about="theta")

        lattice_a, text = get_lattice(0.8, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(Write(text_ref[0]), run_time=0.5)
        self.play(ReplacementTransform(lattice_ref, lattice_a), ReplacementTransform(text_ref[0], text[0]), run_time=2)
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(ReplacementTransform(lattice_a, lattice_ref), ReplacementTransform(text[0], text_ref[0]), run_time=1.5)
        self.remove(text_ref[0])

        lattice_b, text = get_lattice(0.6, 0.8, 0.6, PI/2, PI/2, PI/2)
        self.play(Write(text_ref[1]), run_time=0.5)
        self.play(ReplacementTransform(lattice_ref, lattice_b), ReplacementTransform(text_ref[1], text[1]), run_time=2)
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(ReplacementTransform(lattice_b, lattice_ref), ReplacementTransform(text[1], text_ref[1]), run_time=1.5)
        self.remove(text_ref[1])

        lattice_c, text = get_lattice(0.6, 0.6, 0.8, PI/2, PI/2, PI/2)
        self.play(Write(text_ref[2]), run_time=0.5)
        self.play(ReplacementTransform(lattice_ref, lattice_c), ReplacementTransform(text_ref[2], text[2]), run_time=2)
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(ReplacementTransform(lattice_c, lattice_ref), ReplacementTransform(text[2], text_ref[2]), run_time=1.5)
        self.remove(text_ref[2])

        self.stop_ambient_camera_rotation(about="theta")
        self.begin_ambient_camera_rotation(-20*DEGREES/9, about="theta")

        lattice_alpha, text = get_lattice(0.6, 0.6, 0.6, 60*DEGREES, PI/2, PI/2)
        self.play(Write(text_ref[3][0]), Write(text_ref[3][1]), run_time=0.5)
        self.play(ReplacementTransform(lattice_ref, lattice_alpha), ReplacementTransform(text_ref[3], text[3]), run_time=2)
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(ReplacementTransform(lattice_alpha, lattice_ref), ReplacementTransform(text[3], text_ref[3]), run_time=1.5)
        self.remove(text_ref[3])

        lattice_beta, text = get_lattice(0.6, 0.6, 0.6, PI/2, 60*DEGREES, PI/2)
        self.play(Write(text_ref[4][0]), Write(text_ref[4][1]), run_time=0.5)
        self.play(ReplacementTransform(lattice_ref, lattice_beta), ReplacementTransform(text_ref[4], text[4]), run_time=2)
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(ReplacementTransform(lattice_beta, lattice_ref), ReplacementTransform(text[4], text_ref[4]), run_time=1.5)
        self.remove(text_ref[4])

        lattice_gamma, text = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, 60*DEGREES)
        self.play(Write(text_ref[5][0]), Write(text_ref[5][1]), run_time=0.5)
        self.play(ReplacementTransform(lattice_ref, lattice_gamma), ReplacementTransform(text_ref[5], text[5]), run_time=2)
        lattice_ref, text_ref = get_lattice(0.6, 0.6, 0.6, PI/2, PI/2, PI/2)
        self.play(ReplacementTransform(lattice_gamma, lattice_ref), ReplacementTransform(text[5], text_ref[5]), run_time=1.5)
        self.remove(text_ref[5])

        self.stop_ambient_camera_rotation(about="theta")
