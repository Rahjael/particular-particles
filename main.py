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


from renderer import Renderer
from particle import Particle
from particle_factory import ParticleFactory
import utils as Utils



# TODO explain this class and why I prefer it over a dictionary
class Config():
    seed = None
    temp_folder = './temp/'
    animation_duration = 15 # in seconds
    frame_dimensions = (1080, 1080)
    background_color = (255, 255, 255)
    particle_groups = 13 # Odd values have produced better animations so far
    max_particles = 5
    # min_particles = 5

    initial_particles_state_filename = 'initial_particles_state.json'
    frames_to_draw_filename = 'frames_to_draw.json'









def main():
    session_seed = Config.seed if Config.seed != None else Utils.generate_seed()
    random.seed(session_seed)

    pf = ParticleFactory(Config.frame_dimensions)
    particles = pf.get_random_particles(Config.max_particles, Config.particle_groups, False)

    # Get rid of the output of previous instances
    Utils.delete_file(Config.initial_particles_state_filename)

    # Save initial state of the generated particles
    with open(Config.initial_particles_state_filename, 'a') as file:
        objects = [p.get_as_dict() for p in particles]
        json.dump(objects, file, indent=4)


    # Do stuff with the initial state data.
    # TODO compute all the frames


    renderer = Renderer(Config.seed, Config.frame_dimensions)
    renderer.draw_and_save_frame(particles, Config.background_color, 'test_img.png')

    

    






if __name__ == '__main__':
    main()




























quit()




#
#
# ALREADY TRANSFERED (start)
#
#

# Settings
random_seed = None # this must be left as None to generate new sets. Otherwise use the name of a previously generated set as a seed

# save_performance_stats = True

temp_folder = './temp/'
animation_duration = 15 # in seconds

frame_dimensions = (1080, 1080)

particle_groups = 13 # Odd values have produced better animations so far

# Rough worst case estimate for 59 seconds:
# 1500: 2 hours
# 2000: 5 hours
# 3000: 

# 10 seconds @5000 particles: 2 hours



max_particles = 1000
min_particles = 5

# These are percentage values. They determine at which point
# the animation should stop spawning particles, or should start
# de-spawning them. 
core_phase_start_percent = 1 # % of max_frames
core_phase_end_percent = 10 # % of max_frames

particles_should_spawn_from_around_center = True

particles_max_speed_radius_multiplier = 1 # 1 - 3?
particle_default_body_radius = 6

influence_radius_range = (50, 600) # 70 - 200
interaction_force_range = (0, 100) # 0 - 100

recenter_particles_out_of_screen = True

FPS = 60
seconds_of_nothing_at_the_end = 1

max_frames_in_array_before_saving_to_file = 5000 # How many frames to calculate before saving them all to file to clear RAM


# Automatic settings
frames_to_generate = int(animation_duration * FPS)
zeros_fill_size = len(str(frames_to_generate))
display_center = (frame_dimensions[0] // 2, frame_dimensions[1] // 2)
script_start_time = time.time()
current_temp_path = None
current_frame_num = 0

core_phase_start = int(frames_to_generate * core_phase_start_percent / 100)
core_phase_end = int(frames_to_generate * core_phase_end_percent / 100)

increase_phase_duration = core_phase_start
tail_duration = int(FPS * seconds_of_nothing_at_the_end)
decrease_phase_duration = frames_to_generate - core_phase_end - tail_duration

particles_should_spawn_x_at_a_time = max(math.ceil(max_particles / increase_phase_duration), 1) if core_phase_start != 0 else max_particles
particles_should_despawn_x_at_a_time = max(math.ceil(max_particles / decrease_phase_duration), 1)


print(f'Frames to generate: {frames_to_generate}')
print(f'Core phase start: {core_phase_start}')
print(f'Core phase end: {core_phase_end}')
print(f'Spawn rate: {particles_should_spawn_x_at_a_time}')
print(f'De-spawn rate: {particles_should_despawn_x_at_a_time}')
print(f'Increase phase delta: {particles_should_spawn_x_at_a_time * core_phase_start}')
print(f'Decrease phase delta: {particles_should_despawn_x_at_a_time * decrease_phase_duration}')



#
#
# FUNCTIONS DEFINITIONS
#
#

def delete_png_images(folder):
  for filename in os.listdir(folder):
    if filename.lower().endswith('.png'):
      os.remove(os.path.join(folder, filename))
  print('Temp folder cleared')

def draw_gradient_frame(surface, surface_width, surface_height, position):
  if position == 'left':
    width = int(min(surface_width, surface_height) * 0.2)
    height = surface_height
    for x in range(width):
      alpha = 255 - int((x / width) * 255) # Calculate alpha value based on x position
      strip = pygame.Surface((1, height), pygame.SRCALPHA)
      strip.fill((255, 255, 255, alpha))  # Set the color with alpha
      surface.blit(strip, (x, 0))  # Draw the rectangle onto the screen
  if position == 'right':
    width = int(min(surface_width, surface_height) * 0.2)
    height = surface_height
    for x in range(width):
      alpha = int((x / width) * 255) # Calculate alpha value based on x position
      strip = pygame.Surface((1, height), pygame.SRCALPHA)
      strip.fill((255, 255, 255, alpha))  # Set the color with alpha
      surface.blit(strip, ((surface_width - width) + x, 0))  # Draw the rectangle onto the screen
  if position == 'top':
    width = surface_width
    height = int(min(surface_width, surface_height) * 0.2)
    for y in range(height):
      alpha = 255 - int((y / height) * 255)  # Calculate alpha value based on x position
      strip = pygame.Surface((width, 1), pygame.SRCALPHA)
      strip.fill((255, 255, 255, alpha))  # Set the color with alpha
      surface.blit(strip, (0, y))  # Draw the rectangle onto the screen
  if position == 'bottom':
    width = surface_width
    height = int(min(surface_width, surface_height) * 0.2)
    for y in range(height):
      alpha = int((y / height) * 255)  # Calculate alpha value based on x position
      strip = pygame.Surface((width, 1), pygame.SRCALPHA)
      strip.fill((255, 255, 255, alpha))  # Set the color with alpha
      surface.blit(strip, (0, (surface_height - height) + y))  # Draw the rectangle onto the screen

def get_positions_from_file(file):
  with open(file, "r") as f:
    content = f.read()
  content = content.strip().split(",")
  result = [tuple(map(float, s.strip("()").split(","))) for s in content]
  return result

def generate_particles_positions(particles, interaction_handler, frame_dimensions, display_center, num_frames):
  start_time = time.time()
  active_particles_num = particles_should_spawn_x_at_a_time

  # Convert the list of particles into NumPy arrays
  positions = np.array([p.position for p in particles], dtype=np.float64)
  velocities = np.array([p.velocity for p in particles], dtype=np.float64)
  accelerations = np.array([p.acceleration for p in particles], dtype=np.float64)
  groups = np.array([p.group for p in particles], dtype=np.int32)
  colors = [tuple(p.color) for p in particles]
  
  with open(f'{current_temp_path}groups.txt', 'a') as f:
    for g in groups:
      f.write(str(g) + '\n')

  with open(f'{current_temp_path}colors.txt', 'a') as f:
    for i in range(len(colors)):
      f.write(', '.join(str(value) for value in colors[i]) + '\n' )

  frames = []
  frames.append(positions)

  active_particles_mask = np.zeros(max_particles, dtype=bool)
  active_particles_mask[:active_particles_num] = True


  # Use tqdm for a progress bar
  for frame_num in tqdm(range(num_frames), desc="Calculating positions"):
    position_diffs = positions[active_particles_mask][:, np.newaxis] - positions[active_particles_mask]
    distances_sqrd = np.sum(position_diffs ** 2, axis=-1)
    np.fill_diagonal(distances_sqrd, np.inf)

    # for i, p1 in enumerate(particles):
    #   for j, p2 in enumerate(particles):
    active_indices = np.arange(len(positions))[active_particles_mask]
    # for i in active_indices:
    #   for j in active_indices:
    #     if i == j:
    #       continue

    #     p1_p2_distance_sqrd = distances_sqrd[i, j]

    for idx_i, i in enumerate(active_indices):
      for idx_j, j in enumerate(active_indices):
        if idx_i == idx_j:
          continue

        p1_p2_distance_sqrd = distances_sqrd[idx_i, idx_j]

        if p1_p2_distance_sqrd == 0:
          continue

        interaction = interaction_handler.interaction_scheme[groups[i]][groups[j]]
        interaction_distance = interaction[1]

        if p1_p2_distance_sqrd > interaction_distance ** 2:
          continue

        interaction_force = interaction[0]
        interaction_sign = interaction[2]

        direction_vector = position_diffs[idx_j, idx_i] / np.sqrt(p1_p2_distance_sqrd)
        force_magnitude = max(1, interaction_force / max(1, p1_p2_distance_sqrd))
        force_vector = (interaction_sign * force_magnitude) * direction_vector

        accelerations[j] += force_vector

    # Update velocities and positions
    velocities[active_particles_mask] += accelerations[active_particles_mask]

    # Adjust for max_speed
    max_speed = particles_max_speed_radius_multiplier * particle_default_body_radius
    # velocities_magnitudes = np.linalg.norm(velocities[active_particles_mask], axis=1) # Calculate the magnitudes of the velocities
    # excess_speed_mask = velocities_magnitudes > max_speed # Create a boolean mask for velocities exceeding the maximum speed
    # excess_speed_indices = np.where(excess_speed_mask)
    # velocities[active_particles_mask][excess_speed_indices] = (velocities[active_particles_mask][excess_speed_indices].T / velocities_magnitudes[excess_speed_indices] * max_speed).T

    velocities_magnitudes = np.linalg.norm(velocities, axis=1) # Calculate the magnitudes of the velocities for all particles
    excess_speed_mask = velocities_magnitudes > max_speed # Create a boolean mask for velocities exceeding the maximum speed
    excess_speed_indices = np.where(excess_speed_mask)
    velocities[excess_speed_indices] = (velocities[excess_speed_indices].T / velocities_magnitudes[excess_speed_indices] * max_speed).T # Adjust the velocities of the relevant particles






    # Calculate new positions
    positions[active_particles_mask] += velocities[active_particles_mask]

    if recenter_particles_out_of_screen:
      # out_of_bounds_x = np.logical_or(positions[active_particles_mask][:, 0] < 0, positions[active_particles_mask][:, 0] > frame_dimensions[0])
      # out_of_bounds_y = np.logical_or(positions[active_particles_mask][:, 1] < 0, positions[active_particles_mask][:, 1] > frame_dimensions[1])
      # out_of_bounds = np.logical_or(out_of_bounds_x, out_of_bounds_y)
      # positions[active_particles_mask][out_of_bounds] = display_center

      out_of_bounds_x = np.logical_or(positions[:, 0] < 0, positions[:, 0] > frame_dimensions[0])
      out_of_bounds_y = np.logical_or(positions[:, 1] < 0, positions[:, 1] > frame_dimensions[1])
      out_of_bounds = np.logical_and(active_particles_mask, np.logical_or(out_of_bounds_x, out_of_bounds_y))
      positions[out_of_bounds] = display_center


    # Reset accelerations for the next frame
    accelerations.fill(0)
    frames.append(np.copy(positions[active_particles_mask]))

    # Resize active particles mask
    if frame_num < core_phase_start:
      active_particles_num = min(active_particles_num + particles_should_spawn_x_at_a_time, max_particles)
      active_particles_mask[:active_particles_num] = True
      active_indices = np.arange(len(positions))[active_particles_mask]
    elif frame_num > core_phase_end:
      active_particles_num = max(active_particles_num - particles_should_despawn_x_at_a_time, min_particles)
      active_particles_mask[active_particles_num:] = False
      active_indices = np.arange(len(positions))[active_particles_mask]



    if len(frames) >= max_frames_in_array_before_saving_to_file:
      # print('\nAbout to save to file: ', len(frames), ' frames')
      with open(f'{current_temp_path}frames.txt', 'a') as f:
        for i in range(len(frames)):
          line = ", ".join(str(f) for f in frames[i]) + '\n'
          f.write(line)
      print(f'{frame_num} / {frames_to_generate} frames saved to file. (Elapsed time: {time.time() - start_time})')
      frames = []


    


  with open(f'{current_temp_path}frames.txt', 'a') as f:
    for i in range(len(frames)):
      line = ", ".join(str(f) for f in frames[i]) + '\n'
      f.write(line)
    print(f'{frame_num} / {frames_to_generate} frames saved to file. (Elapsed time: {time.time() - start_time})')

  elapsed_time = time.time() - start_time
  print('Elapsed time: ', elapsed_time)
  print(f'On average, it took {round(elapsed_time / frames_to_generate, 2)} seconds to generate 1 frame')



def generate_pngs(positions, colors):
  global current_frame_num
  display.fill((255, 255, 255))
  for i in range(len(positions)):
    if positions[i][0] > 0:
      pygame.draw.circle(display, colors[i], positions[i], particle_default_body_radius)

  draw_gradient_frame(display, frame_dimensions[0], frame_dimensions[1], 'left')
  draw_gradient_frame(display, frame_dimensions[0], frame_dimensions[1], 'right')
  draw_gradient_frame(display, frame_dimensions[0], frame_dimensions[1], 'top')
  draw_gradient_frame(display, frame_dimensions[0], frame_dimensions[1], 'bottom')
  clock.tick(FPS)
  pygame.display.update()
  pygame.image.save(display, f'{current_temp_path}{interaction_handler.set_name}_{str(current_frame_num).zfill(zeros_fill_size)}.png')
  current_frame_num = current_frame_num + 1
  # print(f'\rSaved frame {current_frame_num} of {frames_to_generate}', end='...', flush=True)

























  #
  #
  # ALREADY TRANSFERED (start)
  #
  #
if __name__ == '__main__':
  
  interaction_handler = InteractionHandler(random_seed, particle_groups, influence_radius_range, interaction_force_range)
  current_temp_path = temp_folder + interaction_handler.set_name + '/'
  
  if not os.path.exists(current_temp_path):
    os.makedirs(current_temp_path)

  with open(f'{current_temp_path}interactions.txt', 'a') as f:
    for line in interaction_handler.interaction_scheme:
      f.write(str(line) + '\n')

  particle_factory = ParticleFactory(particle_default_body_radius=particle_default_body_radius)
  particles = particle_factory.get_random_particles(particle_groups=particle_groups, plane_dimensions=frame_dimensions, n=max_particles, start_from_center=particles_should_spawn_from_around_center)
  #
  #
  # ALREADY TRANSFERED (end)
  #
  #






































  generate_particles_positions(particles, interaction_handler, frame_dimensions, display_center, frames_to_generate)
  print(f'Frames generation completed after {round(time.time() - script_start_time, 2)} seconds')

  pygame.init()

  # Setup
  display = pygame.display.set_mode((frame_dimensions[0], frame_dimensions[1]))
  clock = pygame.time.Clock()
  pygame.display.set_caption(f"Seed: {interaction_handler.set_name}")
  pygame.display.iconify()

  # cProfile.run('run_game_loop()', 'profile_stats')
  # pstats.Stats('profile_stats').strip_dirs().sort_stats('tottime').print_stats()

  colors = []
  frames = []
  with open(f'{current_temp_path}colors.txt', 'r') as f:
    for line in f:
      color_values = [int(value.strip()) for value in line.split(',')]
      color_tuple = tuple(color_values)
      colors.append(color_tuple)
  print(f'{len(colors)} colors loaded. ')

  with open(f'{current_temp_path}frames.txt', 'r') as f:
    for line in f:
      positions_str = line.strip().split(',')
      positions_float = []
      
      for i, p in enumerate(positions_str):
        if p == '':
          print('Empty line found. Skipping.')
          continue
        pair = p.strip('[] ').split()
        positions_float.append(float(pair[0]))
        positions_float.append(float(pair[1]))

      position_tuples = [(positions_float[i], positions_float[i+1]) for i in range(0, len(positions_float), 2)]
      frames.append(position_tuples)
  print(f'{len(frames)} frames loaded.')

  for i in tqdm(range(len(frames)), desc="Generating pngs"):
    generate_pngs(frames[i], colors)

  pygame.quit()

  elapsed_time = time.time() - script_start_time
  avg_time_per_frame = round(elapsed_time / current_frame_num, 2)
  print(f'Generated {current_frame_num} images in {int(elapsed_time)} seconds')
  print(f'Average time to generate 1 frame with {max_particles} particles: {avg_time_per_frame} seconds')
  print('\n')


  print('Saving video...')
  # ffmpeg -i uxoxgid_%04d.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p video.mp4
  # ffmpeg -i ./temp_frames/uxoxgid_%04d.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p video.mp4
  video_name = f'{interaction_handler.set_name}'
  video_name += f'_{frame_dimensions[0]}x{frame_dimensions[1]}'
  video_name += f'_p{max_particles}'
  video_name += f'_g{particle_groups}'
  video_name += f'_p{core_phase_start_percent}-{core_phase_end_percent}'
  video_name += f'_r{influence_radius_range[0]}-{influence_radius_range[1]}'
  video_name += f'_s{particle_default_body_radius}'
  video_name += f'.mp4'

  subprocess.call(f'ffmpeg -framerate {FPS} -i {current_temp_path}{interaction_handler.set_name}_%{len(str(frames_to_generate))}d.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p -loglevel warning {video_name}')
  delete_png_images(current_temp_path)
  
  print(f'End of script. Elapsed time from script start: {round(time.time() - script_start_time, 2)} seconds')
