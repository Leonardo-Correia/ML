from manim import *
import numpy as np

class PlotExample(Scene):
    
    def construct(self):
        title = Tex(r"Bucklery-Leverett").to_corner(UP+LEFT)
        self.play(Create(title))
        self.wait(1)
        plot_axes = Axes(
            x_range=[0, 1, 0.05],
            y_range=[0, 1, 0.05],
            x_length=9,
            y_length=5,
            axis_config={
                "numbers_to_include": np.arange(0, 1 + 0.1, 0.1),
                "font_size": 24,
            },
            tips=False,
        )
        
        t1 = 0.1
        
        u1 = FunctionGraph(
        lambda x: 1 if x<0 else (0.5*(np.sqrt(((((-2*(x/t1)+np.sqrt(1+4*(x/t1))-1)/(x/t1)+ 1)))+1)) if((x/t1)<(1+np.sqrt(2))*1/2) else 0),
        x_range = ([0.001,1.3]),
        ).scale(7).move_to([0,0,0]).set_color(RED)
        
        t2 = 0.2
        
        u2 = FunctionGraph(
        lambda x: 1 if x<0 else (0.5*(np.sqrt(((((-2*(x/t2)+np.sqrt(1+4*(x/t2))-1)/(x/t2)+ 1)))+1)) if((x/t2)<(1+np.sqrt(2))*1/2) else 0),
        x_range = ([0.001,1.3]),
        ).scale(7).move_to([0,0,0]).set_color(RED)
        
        t3 = 0.4
        
        u3 = FunctionGraph(
        lambda x: 1 if x<0 else (0.5*(np.sqrt(((((-2*(x/t3)+np.sqrt(1+4*(x/t3))-1)/(x/t3)+ 1)))+1)) if((x/t3)<(1+np.sqrt(2))*1/2) else 0),
        x_range = ([0.001,1.3]),
        ).scale(7).move_to([0,0,0]).set_color(RED)
        
        t4 = 0.8
        
        u4 = FunctionGraph(
        lambda x: 1 if x<0 else (0.5*(np.sqrt(((((-2*(x/t4)+np.sqrt(1+4*(x/t4))-1)/(x/t4)+ 1)))+1)) if((x/t4)<(1+np.sqrt(2))*1/2) else 0),
        x_range = ([0.001,1.3]),
        ).scale(7).move_to([0,0,0]).set_color(RED)
        
        self.play(Create(plot_axes),run_time = 3)
        
        t_1,t_2,t_3,t_4 = MathTex("t=0.1").shift(3*RIGHT+3*UP),MathTex("t=0.2").shift(3*RIGHT+3*UP),MathTex("t=0.4").shift(3*RIGHT+3*UP),MathTex("t=0.8").shift(3*RIGHT+3*UP)
        
        self.play(Create(u1),Write(t_1),run_time=2)
        
        self.wait(2)
        
        self.play(Transform(u1,u2),Transform(t_1,t_2),run_time = 3)
        self.wait(2)
        self.play(Transform(u1,u3),Transform(t_1,t_3),run_time = 3)
        self.wait(2)
        self.play(Transform(u1,u4),Transform(t_1,t_4),run_time = 3)
        self.wait(2)