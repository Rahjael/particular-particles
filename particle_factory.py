import colorsys
import random

from particle import Particle

class ParticleFactory:
  def __init__(self, 
    plane_dimensions: tuple,
    particle_default_body_radius=4,
    color_saturation_range = (20, 100),
    color_lightness_range = (20, 100)
  ):
    self.plane_dimensions = plane_dimensions
    self.particle_default_body_radius = particle_default_body_radius
    self.color_saturation_range = color_saturation_range
    self.color_lightness_range = color_lightness_range

  def get_group_color(self, group_num, max_groups):
    # Compute the hue range for each group
    hue_range = 360 // max_groups
    hue_min = group_num * hue_range
    hue_max = (group_num + 1) * hue_range
    
    # Generate a random hue within the group's range
    hue = random.randint(hue_min, hue_max)
    saturation = random.randint(self.color_saturation_range[0], self.color_saturation_range[1])
    lightness = random.randint(self.color_lightness_range[0], self.color_lightness_range[1])

    hsla = (hue, saturation, lightness, 255)
    h, s, l, a = hsla
    r, g, b = [int(255 * x) for x in colorsys.hls_to_rgb(h/360, l/100, s/100)]
    return (r, g, b, a)


  def get_random_particles(self, n: int, particle_groups: int, start_from_center=True):
    w = self.plane_dimensions[0]
    h = self.plane_dimensions[1]
    particles = []

    for i in range(n):
      if start_from_center:
        pos_x = w // 2 + random.randint(-(int(self.plane_dimensions[0] * 0.01)), int(self.plane_dimensions[0] * 0.01))
        pos_y = h // 2 + random.randint(-(int(self.plane_dimensions[1] * 0.01)), int(self.plane_dimensions[1] * 0.01))
      else:
        pos_x = random.randint(w * 0.1, w * 0.9)
        pos_y = random.randint(h * 0.1, h * 0.9)

      vel_x = random.uniform(0, 1) * random.choice([-1, 1])
      vel_y = random.uniform(0, 1) * random.choice([-1, 1])

      position = (pos_x, pos_y)
      velocity = (vel_x, vel_y)
      acceleration = (0, 0)

      group = random.choice(range(particle_groups))
      color = self.get_group_color(group, particle_groups)

      p = Particle(position, velocity, acceleration, color, group, self.particle_default_body_radius)
      particles.append(p)

    return particles



