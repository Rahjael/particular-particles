class Particle:
    def __init__(self,
                position,
                velocity,
                acceleration,
                color,
                group,
                body_radius
                ):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.color = color
        self.body_radius = body_radius
        self.group = group
        # print('Particle created: ', self)

    def __repr__(self):
        string = f'Particle: \n'
        string += f'position: {self.position}\n'
        string += f'velocity: {self.velocity}\n'
        string += f'acceleration: {self.acceleration}\n'
        string += f'color: {self.color}\n'
        string += f'body_radius: {self.body_radius}\n'
        string += f'group: {self.group}\n'
        return string
    
    def get_as_dict(self):
        particle = {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'color': self.color,
            'body_radius': self.body_radius,
            'group': self.group,
        }
        return particle
