# clear && manim -p -a --media_dir /tmp reciprocal.py
from manim import *
import torch

from laueimproc.diffraction import lattice_to_primitive, primitive_to_reciprocal

config.frame_width = 3
config.frame_height = 3
config.pixel_height, config.pixel_width = 1024, 1024
config.frame_rate = 25

config.output_file = "_"  # solve strage name with -a option


def draw_primitive(primitive):
    ((e1x, e2x, e3x), (e1y, e2y, e3y), (e1z, e2z, e3z)) = primitive.tolist()

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
        Group(e1, e2, e3, e1_text, e2_text, e3_text),
        Group(
            a_text, b_text, c_text,
            Group(alpha_text, arc_alpha), Group(beta_text, arc_beta), Group(gamma_text, arc_gamma),
        ),
    )


def draw_reciprocal(reciprocal):
    ((e1x, e2x, e3x), (e1y, e2y, e3y), (e1z, e2z, e3z)) = reciprocal.tolist()
    e1 = Arrow3D([0, 0, 0], [e1x, e1y, e1z], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04)
    e2 = Arrow3D([0, 0, 0], [e2x, e2y, e2z], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04)
    e3 = Arrow3D([0, 0, 0], [e3x, e3y, e3z], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04)
    e1_text = MathTex("e^*_1", color=GREEN, font_size=18).move_to([1.1*e1x, 1.1*e1y, 1.1*e1z + 0.05]).rotate(3*PI/4)
    e2_text = MathTex("e^*_2", color=GREEN, font_size=18).move_to([1.1*e2x, 1.1*e2y, 1.1*e2z + 0.05]).rotate(3*PI/4)
    e3_text = MathTex("e^*_3", color=GREEN, font_size=18).move_to([1.1*e3x, 1.1*e3y, 1.1*e3z + 0.05]).rotate(3*PI/4)
    return Group(e1, e2, e3, e1_text, e2_text, e3_text)


class IMGPrimitiveReciprocal(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/4, focal_distance=10, frame_center=[0.0, 0.0, 0.1])
        lattice = torch.tensor([0.8, 0.8, 0.8, 75*DEGREES, 90*DEGREES, 75*DEGREES])
        primitive = lattice_to_primitive(lattice)

        prim_ref, text_ref = draw_primitive(primitive)
        rec_ref = draw_reciprocal(primitive_to_reciprocal(primitive))
        self.add(prim_ref, text_ref, rec_ref)


class ANIMPrimitiveReciprocal(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=35*DEGREES, focal_distance=10, frame_center=[0.0, 0.0, 0.1])
        primitive_ref = lattice_to_primitive(torch.tensor([0.8, 0.8, 0.8, PI/2, PI/2, PI/2]))
        prim_ref, text_ref = draw_primitive(primitive_ref)
        rec_ref = draw_reciprocal(primitive_to_reciprocal(primitive_ref))
        self.add(prim_ref, rec_ref)

        self.begin_ambient_camera_rotation(20*DEGREES/9, about="theta")

        for i, lattice in enumerate([
            (1.1, 0.8, 0.8, PI/2, PI/2, PI/2),
            (0.8, 1.1, 0.8, PI/2, PI/2, PI/2),
            (0.8, 0.8, 1.1, PI/2, PI/2, PI/2),
        ]):
            primitive = lattice_to_primitive(torch.tensor(lattice))
            prim_a, text = draw_primitive(primitive)
            rec_a = draw_reciprocal(primitive_to_reciprocal(primitive))
            self.play(Write(text_ref[i]), run_time=0.5)
            self.play(
                ReplacementTransform(prim_ref, prim_a),
                ReplacementTransform(text_ref[i], text[i]),
                ReplacementTransform(rec_ref, rec_a),
                run_time=2,
            )
            prim_ref, text_ref = draw_primitive(primitive_ref)
            rec_ref = draw_reciprocal(primitive_to_reciprocal(primitive_ref))
            self.play(
                ReplacementTransform(prim_a, prim_ref),
                ReplacementTransform(text[i], text_ref[i]),
                ReplacementTransform(rec_a, rec_ref),
                run_time=1.5,
            )
            self.remove(text_ref[i])

        self.stop_ambient_camera_rotation(about="theta")
        self.begin_ambient_camera_rotation(-20*DEGREES/9, about="theta")

        for i, lattice in enumerate([
            (0.8, 0.8, 0.8, PI/3, PI/2, PI/2),
            (0.8, 0.8, 0.8, PI/2, PI/3, PI/2),
            (0.8, 0.8, 0.8, PI/2, PI/2, PI/3),
        ]):
            primitive = lattice_to_primitive(torch.tensor(lattice))
            prim_a, text = draw_primitive(primitive)
            rec_a = draw_reciprocal(primitive_to_reciprocal(primitive))
            self.play(Write(text_ref[i+3][0]), Write(text_ref[i+3][1]), run_time=0.5)
            self.play(
                ReplacementTransform(prim_ref, prim_a),
                ReplacementTransform(text_ref[i+3][0], text[i+3][0]),
                ReplacementTransform(rec_ref, rec_a),
                run_time=2,
            )
            prim_ref, text_ref = draw_primitive(primitive_ref)
            rec_ref = draw_reciprocal(primitive_to_reciprocal(primitive_ref))
            self.play(
                ReplacementTransform(prim_a, prim_ref),
                ReplacementTransform(text[i+3][0], text_ref[i+3][0]),
                ReplacementTransform(rec_a, rec_ref),
                run_time=1.5,
            )
            self.remove(text_ref[i+3])

        self.stop_ambient_camera_rotation(about="theta")
