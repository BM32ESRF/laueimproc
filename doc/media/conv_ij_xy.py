# clear && manim -p -a --media_dir /tmp conv_ij_xy.py
from manim import *

config.frame_width = 5
config.frame_height = 5
config.pixel_height, config.pixel_width = 1024, 1024
config.frame_rate = 25

config.output_file = "_"  # solve strage name with -a option


class IMGConvIJXY(Scene):
    def get_base_ij(self):
        ij_dot = Dot([-2, 2, 0], color=GREEN)
        ij_text = MathTex("(0, 0)", color=GREEN, font_size=18).move_to([-2.1, 2.25, 0])
        i_arrow = Arrow(ij_dot, [-2, 1, 0], buff=1, color=GREEN)
        i_text = MathTex("i", color=GREEN, font_size=18).next_to(i_arrow, UR)
        j_arrow = Arrow(ij_dot, [-1, 2, 0], buff=1, color=GREEN)
        j_text = MathTex("j", color=GREEN, font_size=18).next_to(j_arrow, DL)
        base_ij = Group(i_arrow, i_text, j_arrow, j_text, ij_dot, ij_text)
        return base_ij

    def get_base_xy(self):
        xy_dot = Dot([-1.5, 1.5, 0], color=RED)
        xy_text = MathTex("(1, 1)", color=RED, font_size=18).move_to([-1.6, 1.75, 0])
        y_arrow = Arrow(xy_dot, [-1.5, 0.5, 0], buff=1, color=RED)
        y_text = MathTex("y", color=RED, font_size=18).next_to(y_arrow, DOWN)
        x_arrow = Arrow(xy_dot, [-0.5, 1.5, 0], buff=1, color=RED)
        x_text = MathTex("x", color=RED, font_size=18).next_to(x_arrow, RIGHT)
        base_xy = Group(y_arrow, y_text, x_arrow, x_text, xy_dot, xy_text)
        return base_xy

    def construct(self):
        grid = Rectangle(width=4, height=4, grid_xstep=1, grid_ystep=1)
        title = Text("Convention ij and xy", font_size=16).move_to([0, 2.25, 0])

        point = Dot([0.2, 0.2, 0], color=YELLOW)
        point_text = MathTex("p", color=YELLOW, font_size=18).move_to([0.45, 0.45, 0])
        point_coord = MathTex("p", " = ", "(2.2, 1.8)_{ij}", " = ", "(2.7, 2.3)_{xy}", font_size=18).move_to([0, -2.25, 0])
        point_coord[0].set_color(YELLOW)
        point_coord[2].set_color(GREEN)
        point_coord[4].set_color(RED)
        p_group = Group(point, point_text, point_coord)

        self.add(grid, title, self.get_base_ij(), self.get_base_xy(), p_group)


class ANIMConvIJNumpyContinuity(IMGConvIJXY):
    def construct(self):
        # introduction
        msg1 = Text("Numpy convention (ij)", font_size=16).move_to([0, 2.25, 0])
        self.play(Write(msg1))
        grid = Rectangle(width=4, height=4, grid_xstep=1, grid_ystep=1)
        self.play(Create(grid))
        self.play(FadeIn(self.get_base_ij()))

        # numpy slicing
        slice1 = MathTex(*"img[1:3, 1:4]", color=BLUE, font_size=18).move_to([0, -2.25, 0])
        self.play(Create(slice1))
        framebox1 = SurroundingRectangle(slice1[4], buff=0, color=YELLOW)
        framebox2 = SurroundingRectangle(slice1[9], buff=0, color=YELLOW)
        self.play(Create(framebox1), Create(framebox2))
        point1 = Dot([-1, 1, 0], color=YELLOW)
        self.play(ReplacementTransform(framebox1, point1), ReplacementTransform(framebox2, point1))
        framebox1 = SurroundingRectangle(slice1[6], buff=0, color=YELLOW)
        framebox2 = SurroundingRectangle(slice1[10], buff=0, color=YELLOW)
        point2 = Dot([2, -1, 0], color=YELLOW)
        self.play(ReplacementTransform(framebox1, point2), ReplacementTransform(framebox2, point2))
        rect1 = Rectangle(width=3, height=2, fill_opacity=0.5, color=BLUE, fill_color=BLUE).move_to([0.5, 0, 0])
        self.play(FadeIn(rect1))
        slice2 = MathTex("img[1:2, 1:2]", color=BLUE, font_size=18).move_to([0, -2.25, 0])
        point3 = Dot([0, 0, 0], color=YELLOW)
        rect2 = Rectangle(width=1, height=1, fill_opacity=0.5, color=BLUE, fill_color=BLUE).move_to([-0.5, 0.5, 0])
        self.play(
            FadeOut(slice1), FadeIn(slice2),
            ReplacementTransform(point2, point3),
            ReplacementTransform(rect1, rect2),
        )
        slice3 = MathTex("img[1, 1]", color=BLUE, font_size=18).move_to([0, -2.25, 0])
        self.play(FadeOut(slice2), FadeIn(slice3), FadeOut(point2), FadeOut(point3))
        self.wait()

        # prolongation by continuity
        msg2 = Text("Continuity extension", font_size=16).move_to([0, 2.25, 0])
        self.play(FadeOut(msg1), FadeIn(msg2), FadeOut(rect2))
        p_text1 = MathTex("p", color=YELLOW, font_size=18).move_to([-0.75, 1.25, 0])
        slice4 = MathTex("p = (1.0, 1.0)_{ij}", color=GREEN, font_size=18).move_to([0, -2.25, 0])
        self.play(Write(p_text1), ReplacementTransform(slice3, slice4))
        point2 = point1.copy().move_to([0.2, 0.2, 0])
        p_text2 = p_text1.copy().move_to([0.45, 0.45, 0])
        slice5 = MathTex("p = (2.2, 1.8)_{ij}", color=GREEN, font_size=18).move_to([0, -2.25, 0])
        self.play(ReplacementTransform(slice4, slice5), ReplacementTransform(point1, point2), ReplacementTransform(p_text1, p_text2))
        self.wait()
