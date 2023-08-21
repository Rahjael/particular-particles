import cProfile
import json
import math
import numpy as np
import os
import pstats
import pygame
import random
import subprocess
import time
from tqdm import tqdm


from engine import Engine
from renderer import Renderer
from particle import Particle
from particle_factory import ParticleFactory
import utils as Utils



# TODO explain this class and why I prefer it over a dictionary
class Config():
    def __init__(self):
        # Manual settings
        seed = None
        animation_duration = 30 # in seconds
        animation_fps = 60
        frame_dimensions = (500, 500)
        particle_groups = 4 # Odd values have produced better animations so far
        background_color = (255, 255, 255)
        max_particles = 5
        starting_particles = 5
        influence_radius_range = (50, 500) # 70 - 200
        interaction_force_range = (0, 100) # 0 - 100
        recenter_particles_out_of_screen = True
        seconds_of_nothing_at_the_end = 1
        max_frames_in_array_before_saving_to_file = 5000
        # These are percentage values. They determine at which point
        # the animation should stop spawning particles, or should start
        # de-spawning them. 
        core_phase_start_percent = 1 # % of max_frames
        core_phase_end_percent = 10 # % of max_frames
        particles_max_speed_radius_multiplier = 1 # 1 - 3?
        particles_default_body_radius = 6


        # Derived settings
        frames_to_generate = int(animation_duration * animation_fps)
        core_phase_start = int(frames_to_generate * core_phase_start_percent / 100)
        core_phase_end = int(frames_to_generate * core_phase_end_percent / 100)
        increase_phase_duration = core_phase_start
        tail_duration = int(animation_fps * seconds_of_nothing_at_the_end)
        decrease_phase_duration = frames_to_generate - core_phase_end - tail_duration
        particles_should_spawn_x_at_a_time = max(math.ceil(max_particles / increase_phase_duration), 1) if core_phase_start != 0 else max_particles
        particles_should_despawn_x_at_a_time = max(math.ceil(max_particles / decrease_phase_duration), 1)

        self.seed = seed
        self.animation_duration = animation_duration
        self.animation_fps = animation_fps
        self.frame_dimensions = frame_dimensions
        self.particle_groups = particle_groups
        self.background_color = background_color
        self.max_particles = max_particles
        self.starting_particles = starting_particles
        self.influence_radius_range = influence_radius_range
        self.interaction_force_range = interaction_force_range
        self.recenter_particles_out_of_screen = recenter_particles_out_of_screen
        self.seconds_of_nothing_at_the_end = seconds_of_nothing_at_the_end
        self.max_frames_in_array_before_saving_to_file = max_frames_in_array_before_saving_to_file
        self.frames_to_generate = frames_to_generate
        self.zeros_fill_size = len(str(frames_to_generate))
        self.display_center = (frame_dimensions[0] // 2, frame_dimensions[1] // 2)
        self.temp_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'./temp/{seed}')
        self.initial_particles_state_filename = 'initial_particles_state.json'
        self.particles_max_speed_radius_multiplier = particles_max_speed_radius_multiplier
        self.particles_default_body_radius = particles_default_body_radius
        self.core_phase_start_percent = core_phase_start_percent
        self.core_phase_end_percent = core_phase_end_percent
        self.core_phase_start = core_phase_start
        self.core_phase_end = core_phase_end
        self.increase_phase_duration = increase_phase_duration
        self.tail_duration = tail_duration
        self.decrease_phase_duration = decrease_phase_duration
        self.particles_should_spawn_x_at_a_time = particles_should_spawn_x_at_a_time
        self.particles_should_despawn_x_at_a_time = particles_should_despawn_x_at_a_time

        print(f'Frames to generate: {frames_to_generate}')
        print(f'Core phase start: {core_phase_start}')
        print(f'Core phase end: {core_phase_end}')
        print(f'Spawn rate: {particles_should_spawn_x_at_a_time}')
        print(f'De-spawn rate: {particles_should_despawn_x_at_a_time}')
        print(f'Increase phase delta: {particles_should_spawn_x_at_a_time * core_phase_start}')
        print(f'Decrease phase delta: {particles_should_despawn_x_at_a_time * decrease_phase_duration}')

    def reseed(self, seed):      
        self.seed = seed  
        self.temp_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'./temp/{seed}')
        os.mkdir(self.temp_folder_path)
        return seed


class GlobalState():
    def __init__(self):
        self.script_start_time = time.time()
        self.frames_saved = 0





def main():
    global global_state
    # Creating config and seed as necessary
    config = Config()
    session_seed = config.seed if config.seed != None else Utils.generate_seed()
    random.seed(config.reseed(session_seed))

    

    print(f'config.temp_folder_path: {config.temp_folder_path}')

    pf = ParticleFactory(config.frame_dimensions, config.particles_default_body_radius)
    particles: list[Particle] = pf.get_random_particles(config.max_particles, config.particle_groups, False)

    # Get rid of the output of previous instances
    Utils.delete_file(config.initial_particles_state_filename)

    # Save initial state of the generated particles
    with open(config.initial_particles_state_filename, 'a') as file:
        dict_particles = [p.get_as_dict() for p in particles]
        json.dump(dict_particles, file, indent=4)

    # Setup the engine for calculations
    engine = Engine(config.seed, config.particle_groups, config.influence_radius_range, config.interaction_force_range, config.temp_folder_path)

    # Calculate the positions for every frame
    # engine.compute_frames(particles, config)



    # renderer = Renderer(config.seed, config.frame_dimensions)
    # renderer.draw_and_save_frame(particles, config.background_color, 'test_img.png')


    elapsed_time = time.time() - global_state.script_start_time
    print(f'Generated {global_state.frames_saved} images in {int(elapsed_time)} seconds')

    if global_state.frames_saved > 0:
        avg_time_per_frame = round(elapsed_time / global_state.frames_saved, 2)
        print(f'Average time to generate 1 frame with {Config.max_particles} particles: {avg_time_per_frame} seconds')
        print('\n')

    
    # TODO save video
    

    # Quitting program
    # renderer.cleanup()




global_state = GlobalState()
if __name__ == '__main__':
    print(global_state.script_start_time, global_state.frames_saved)
    main()
