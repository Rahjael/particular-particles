import json
import numpy as np
import os
import random
import time
import tqdm


class Engine:
    def __init__(self, seed, groups, influence_radius_range, interaction_force_range, temp_folder_path):
        print(f'Generating new interaction scheme with seed: {seed}')
        self.interactions_scheme = Engine.generate_interactions_scheme(
            groups, influence_radius_range, interaction_force_range, temp_folder_path)
        self.temp_folder_path = temp_folder_path

        
    def __init__(self, seed, groups, influence_radius_range, interaction_force_range, temp_folder_path="."):
        print(f'Generating new interaction scheme with seed: {seed}')
        self.interactions_scheme = Engine.generate_interactions_scheme(
            groups, influence_radius_range, interaction_force_range, temp_folder_path)
        self.temp_folder_path = temp_folder_path
        self.particles = []  # List to store the particles


    @staticmethod
    def generate_interactions_scheme(groups, influence_radius_range, interaction_force_range, temp_folder_path):
        interactions_scheme = []

        for i in range(groups):
            interactions_scheme.append([])
            for _ in range(groups):
                interaction_max_force = random.randint(
                    interaction_force_range[0], interaction_force_range[1])
                sign = random.choice([-1, 1])
                radius = random.randint(
                    influence_radius_range[0], influence_radius_range[1])
                interaction = (interaction_max_force, radius, sign)
                interactions_scheme[i].append(interaction)

        print(f'received: {temp_folder_path}')
        path = os.path.join(temp_folder_path, './interactions_scheme.json')
        with open(path, 'w') as file:
            json.dump(interactions_scheme, file, indent=4)            

        # print('Interaction scheme generated: \n', interaction_scheme)
        return interactions_scheme



    def dummy_compute_frames(self, particles, config):
        current_temp_path = config.current_temp_path
        particles_max_speed_radius_multiplier = config.particles_max_speed_radius_multiplier
        particle_default_body_radius = config.particle_default_body_radius
        recenter_particles_out_of_screen = config.recenter_particles_out_of_screen
        frame_dimensions = config.frame_dimensions
        display_center = config.display_center
        max_frames_in_array_before_saving_to_file = config.max_frames_in_array_before_saving_to_file
        frames_to_generate = config.frames_to_generate

        interactions_scheme = self.interactions_scheme


        start_time = time.time()

        # Convert the list of particles into NumPy arrays
        positions = np.array([p.position for p in particles], dtype=np.float64)
        velocities = np.array([p.velocity for p in particles], dtype=np.float64)
        accelerations = np.array([p.acceleration for p in particles], dtype=np.float64)
        groups = np.array([p.group for p in particles], dtype=np.int32)
        colors = [tuple(p.color) for p in particles]

        frames = []
        frames.append(positions)


        # Use tqdm for a progress bar
        for frame_num in tqdm(range(frames_to_generate), desc="Calculating positions"):
            



            # ALGORITHM GOES HERE



            if len(frames) >= max_frames_in_array_before_saving_to_file:
                # print('\nAbout to save to file: ', len(frames), ' frames')
                with open(f'{current_temp_path}frames.txt', 'a') as f:
                    for i in range(len(frames)):
                        line = ", ".join(str(f) for f in frames[i]) + '\n'
                        f.write(line)
                print(
                    f'{frame_num} / {frames_to_generate} frames saved to file. (Elapsed time: {time.time() - start_time})')
                frames = []

        with open(f'{current_temp_path}frames.txt', 'a') as f:
            for i in range(len(frames)):
                line = ", ".join(str(f) for f in frames[i]) + '\n'
                f.write(line)
            print(f'{frame_num} / {frames_to_generate} frames saved to file. (Elapsed time: {time.time() - start_time})')

        elapsed_time = time.time() - start_time
        print('Elapsed time: ', elapsed_time)
        print(
            f'On average, it took {round(elapsed_time / frames_to_generate, 2)} seconds to generate 1 frame')









    def compute_frames_old_version(self, particles, config):
        particles_should_spawn_x_at_a_time = config.particles_should_spawn_x_at_a_time
        current_temp_path = config.current_temp_path
        max_particles = config.max_particles
        starting_particles = config.starting_particles
        num_frames = config.num_frames
        particles_max_speed_radius_multiplier = config.particles_max_speed_radius_multiplier
        particle_default_body_radius = config.particle_default_body_radius
        recenter_particles_out_of_screen = config.recenter_particles_out_of_screen
        frame_dimensions = config.frame_dimensions
        display_center = config.display_center
        core_phase_start = config.core_phase_start
        core_phase_end = config.core_phase_end
        particles_should_despawn_x_at_a_time = config.particles_should_despawn_x_at_a_time
        min_particles = config.min_particles
        max_frames_in_array_before_saving_to_file = config.max_frames_in_array_before_saving_to_file
        frames_to_generate = config.frames_to_generate



        start_time = time.time()
        active_particles_num = particles_should_spawn_x_at_a_time

        # Convert the list of particles into NumPy arrays
        positions = np.array([p.position for p in particles], dtype=np.float64)
        velocities = np.array([p.velocity for p in particles], dtype=np.float64)
        accelerations = np.array(
            [p.acceleration for p in particles], dtype=np.float64)
        groups = np.array([p.group for p in particles], dtype=np.int32)
        colors = [tuple(p.color) for p in particles]

        with open(f'{current_temp_path}groups.txt', 'a') as f:
            for g in groups:
                f.write(str(g) + '\n')

        with open(f'{current_temp_path}colors.txt', 'a') as f:
            for i in range(len(colors)):
                f.write(', '.join(str(value) for value in colors[i]) + '\n')

        frames = []
        frames.append(positions)

        active_particles_mask = np.zeros(max_particles, dtype=bool)
        active_particles_mask[:active_particles_num] = True

        # Use tqdm for a progress bar
        for frame_num in tqdm(range(num_frames), desc="Calculating positions"):
            position_diffs = positions[active_particles_mask][:,
                                                            np.newaxis] - positions[active_particles_mask]
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

                    interaction = self.interaction_scheme[groups[i]][groups[j]]
                    interaction_distance = interaction[1]

                    if p1_p2_distance_sqrd > interaction_distance ** 2:
                        continue

                    interaction_force = interaction[0]
                    interaction_sign = interaction[2]

                    direction_vector = position_diffs[idx_j,
                                                    idx_i] / np.sqrt(p1_p2_distance_sqrd)
                    force_magnitude = max(
                        1, interaction_force / max(1, p1_p2_distance_sqrd))
                    force_vector = (interaction_sign *
                                    force_magnitude) * direction_vector

                    accelerations[j] += force_vector

            # Update velocities and positions
            velocities[active_particles_mask] += accelerations[active_particles_mask]

            # Adjust for max_speed
            max_speed = particles_max_speed_radius_multiplier * particle_default_body_radius
            # velocities_magnitudes = np.linalg.norm(velocities[active_particles_mask], axis=1) # Calculate the magnitudes of the velocities
            # excess_speed_mask = velocities_magnitudes > max_speed # Create a boolean mask for velocities exceeding the maximum speed
            # excess_speed_indices = np.where(excess_speed_mask)
            # velocities[active_particles_mask][excess_speed_indices] = (velocities[active_particles_mask][excess_speed_indices].T / velocities_magnitudes[excess_speed_indices] * max_speed).T

            # Calculate the magnitudes of the velocities for all particles
            velocities_magnitudes = np.linalg.norm(velocities, axis=1)
            # Create a boolean mask for velocities exceeding the maximum speed
            excess_speed_mask = velocities_magnitudes > max_speed
            excess_speed_indices = np.where(excess_speed_mask)
            # Adjust the velocities of the relevant particles
            velocities[excess_speed_indices] = (
                velocities[excess_speed_indices].T / velocities_magnitudes[excess_speed_indices] * max_speed).T

            # Calculate new positions
            positions[active_particles_mask] += velocities[active_particles_mask]

            if recenter_particles_out_of_screen:
                # out_of_bounds_x = np.logical_or(positions[active_particles_mask][:, 0] < 0, positions[active_particles_mask][:, 0] > frame_dimensions[0])
                # out_of_bounds_y = np.logical_or(positions[active_particles_mask][:, 1] < 0, positions[active_particles_mask][:, 1] > frame_dimensions[1])
                # out_of_bounds = np.logical_or(out_of_bounds_x, out_of_bounds_y)
                # positions[active_particles_mask][out_of_bounds] = display_center

                out_of_bounds_x = np.logical_or(
                    positions[:, 0] < 0, positions[:, 0] > frame_dimensions[0])
                out_of_bounds_y = np.logical_or(
                    positions[:, 1] < 0, positions[:, 1] > frame_dimensions[1])
                out_of_bounds = np.logical_and(
                    active_particles_mask, np.logical_or(out_of_bounds_x, out_of_bounds_y))
                positions[out_of_bounds] = display_center

            # Reset accelerations for the next frame
            accelerations.fill(0)
            frames.append(np.copy(positions[active_particles_mask]))

            # Resize active particles mask
            if frame_num < core_phase_start:
                active_particles_num = min(
                    active_particles_num + particles_should_spawn_x_at_a_time, max_particles)
                active_particles_mask[:active_particles_num] = True
                active_indices = np.arange(len(positions))[active_particles_mask]
            elif frame_num > core_phase_end:
                active_particles_num = max(
                    active_particles_num - particles_should_despawn_x_at_a_time, min_particles)
                active_particles_mask[active_particles_num:] = False
                active_indices = np.arange(len(positions))[active_particles_mask]

            if len(frames) >= max_frames_in_array_before_saving_to_file:
                # print('\nAbout to save to file: ', len(frames), ' frames')
                with open(f'{current_temp_path}frames.txt', 'a') as f:
                    for i in range(len(frames)):
                        line = ", ".join(str(f) for f in frames[i]) + '\n'
                        f.write(line)
                print(
                    f'{frame_num} / {frames_to_generate} frames saved to file. (Elapsed time: {time.time() - start_time})')
                frames = []

        with open(f'{current_temp_path}frames.txt', 'a') as f:
            for i in range(len(frames)):
                line = ", ".join(str(f) for f in frames[i]) + '\n'
                f.write(line)
            print(f'{frame_num} / {frames_to_generate} frames saved to file. (Elapsed time: {time.time() - start_time})')

        elapsed_time = time.time() - start_time
        print('Elapsed time: ', elapsed_time)
        print(
            f'On average, it took {round(elapsed_time / frames_to_generate, 2)} seconds to generate 1 frame')









class Engine:
    def __init__(self, seed, groups, influence_radius_range, interaction_force_range, temp_folder_path="."):
        print(f'Generating new interaction scheme with seed: {seed}')
        self.interactions_scheme = Engine.generate_interactions_scheme(
            groups, influence_radius_range, interaction_force_range, temp_folder_path)
        self.temp_folder_path = temp_folder_path
        self.particles = []  # List to store the particles

    def add_particle(self, position, group, velocity=(0,0)):
        """Method to add a particle to the engine."""
        self.particles.append(Particle(position, group, velocity))

    def generate_frames(self, num_frames):
        """Generates the frames based on particle interactions."""
        return generate_frames(self.particles, self.interactions_scheme, num_frames)

    @staticmethod
    def generate_interactions_scheme(groups, influence_radius_range, interaction_force_range, temp_folder_path="."):
        interactions_scheme = []

        for i in range(groups):
            interactions_scheme.append([])
            for _ in range(groups):
                interaction_max_force = random.randint(
                    interaction_force_range[0], interaction_force_range[1])
                sign = random.choice([-1, 1])
                radius = random.randint(
                    influence_radius_range[0], influence_radius_range[1])
                interaction = (interaction_max_force, radius, sign)
                interactions_scheme[i].append(interaction)

        path = os.path.join(temp_folder_path, 'interactions_scheme.json')
        with open(path, 'w') as file:
            json.dump(interactions_scheme, file, indent=4)

        return interactions_scheme

# Testing the integrated Engine
engine = Engine("test_seed", 2, (1, 10), (-10, 10))
engine.add_particle((0, 0), group=0)
engine.add_particle((1, 1), group=1)
frames = engine.generate_frames(3)
frames
