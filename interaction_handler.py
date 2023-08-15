import random

class InteractionHandler:
  def __init__(self, random_seed, groups, influence_radius_range, interaction_force_range):
    print(f'Generating new interaction scheme with seed: {self.set_name}')
    self.interaction_scheme = InteractionHandler.generate_interaction_scheme(groups, influence_radius_range, interaction_force_range)

  @staticmethod
  def generate_interaction_scheme(groups, influence_radius_range, interaction_force_range):
    interaction_scheme = []

    for i in range(groups):
      interaction_scheme.append([])
      for j in range(groups):
        multiplier = random.randint(interaction_force_range[0], interaction_force_range[1])
        sign = random.choice([-1, 1])
        radius = random.randint(influence_radius_range[0], influence_radius_range[1])
        interaction = (multiplier, radius, sign)
        interaction_scheme[i].append(interaction)

    # print('Interaction scheme generated: \n', interaction_scheme)
    return interaction_scheme
