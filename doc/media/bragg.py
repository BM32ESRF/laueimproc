# clear && manim -p -a --media_dir /tmp bragg.py
from manim import *

config.frame_width = 2.5
config.frame_height = 2.5
config.pixel_height, config.pixel_width = 1024, 1024
config.frame_rate = 25

config.output_file = "_"  # solve strange name with -a option


def get_base():
    return Group(
        Arrow3D([0, 0, 0], [1, 0, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([0, 0, 0], [0, -1, 0], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        Arrow3D([0, 0, 0], [0, 0, 1], color=GREEN, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
        MathTex("L_3", color=GREEN, font_size=18).move_to([1.2, 0, 0]).rotate(5*PI/6),
        MathTex("L_2", color=GREEN, font_size=18).move_to([0, -1.2, 0]).rotate(5*PI/6),
        MathTex("L_1", color=GREEN, font_size=18).move_to([0, 0, 1.2]).rotate(5*PI/6),
    )


class IMGBragg(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/3, focal_distance=10, frame_center=[0.0, -0.2, 0.1])
        base = get_base()
        u_i = Group(
            Arrow3D([-1, 0, 0], [0, 0, 0], color=YELLOW, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
            MathTex("u_i", color=YELLOW, font_size=18).move_to([-1.2, 0, 0]).rotate(5*PI/6),
        )
        self.add(base, u_i)


class IMGThetaChi(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI/3, theta=PI/6, focal_distance=10, frame_center=[0.0, -0.2, 0.1])
        base = get_base()
        l1l3 = Polyhedron(
            [[0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, 0, -0.5]],
            [[0, 1, 2], [2, 3, 0]],
            faces_config={"fill_opacity": 0.25, "stroke_opacity": 1, "stroke_width": 0, "color": GREEN},
            graph_config={
                "edge_config": {"stroke_opacity": 0},
                "vertex_config": {"radius": 0},
            },
        )
        u_i = Group(
            Arrow3D([-1, 0, 0], [0, 0, 0], color=YELLOW, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
            MathTex("u_i", color=YELLOW, font_size=18).move_to([-1.2, 0, 0]).rotate(2*PI/3),
        )
        uf = np.array([0.5, 0.433, 0.75])
        l3 = np.array([1.0, 0.0, 0.0])
        u_f = Group(
            Arrow3D([0, 0, 0], uf, color=YELLOW, resolution=16, thickness=0.01, height=0.1, base_radius=0.04),
            MathTex("u_f", color=YELLOW, font_size=18).move_to([0.7, 0.633, 0.95]).rotate(2*PI/3),
        )

        l1l3 = Polyhedron(
            [[0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, 0, -0.5]],
            [[0, 1, 2], [2, 3, 0]],
            faces_config={"fill_opacity": 0.25, "stroke_opacity": 1, "stroke_width": 0, "color": GREEN},
            graph_config={
                "edge_config": {"stroke_opacity": 0},
                "vertex_config": {"radius": 0},
            },
        )

        arc_twicetheta = np.asarray([(t*l3 + (1-t)*uf) for t in np.linspace(0, 1, 16)])
        arc_twicetheta *= 0.4 / np.sqrt(np.sum(arc_twicetheta*arc_twicetheta, axis=1, keepdims=True))
        arc_twicetheta = Polygon(*arc_twicetheta, [0, 0, 0], color=RED, stroke_width=0.5)
        pos = l3+uf
        pos *= 0.6 / np.sqrt(pos.sum())
        twicetheta_text = MathTex("2\\theta", color=RED, font_size=18).move_to(pos).rotate(2*PI/3)
        uf_proj = np.array([uf[0], 0.0, uf[2]])
        arc_chi = np.asarray([(t*uf_proj + (1-t)*uf) for t in np.linspace(0, 1, 16)])
        arc_chi *= 0.4 / np.sqrt(np.sum(arc_chi*arc_chi, axis=1, keepdims=True))
        arc_chi = Polygon(*arc_chi, [0, 0, 0], color=PURPLE, stroke_width=0.5)
        pos = uf_proj+uf
        pos *= 0.6 / np.sqrt(pos.sum())
        chi_text = MathTex("\\chi", color=PURPLE, font_size=18).move_to(pos).rotate(2*PI/3)

        self.add(base, l1l3, u_i, u_f, arc_twicetheta, twicetheta_text, arc_chi, chi_text)
