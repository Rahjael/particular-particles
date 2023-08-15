import pygame


class Renderer():
    def __init__(self, seed, frame_dimensions):
        self.seed = seed
        self.frame_dimensions = frame_dimensions

        pygame.init()
        self.display = pygame.display.set_mode(
            (frame_dimensions[0], frame_dimensions[1]))
        pygame.display.set_caption(f"Seed: {seed}")
        pygame.display.iconify()

    def draw_and_save_frame(self, particles: list, background_color: (int, int, int), filename: str):        
        self.display.fill(background_color)

        for p in particles:
            pygame.draw.circle(self.display, p.color, p.position, p.body_radius)

        # draw_gradient_frame(
        #     self.display, self.frame_dimensions[0], self.frame_dimensions[1], 'left')
        # draw_gradient_frame(
        #     self.display, self.frame_dimensions[0], self.frame_dimensions[1], 'right')
        # draw_gradient_frame(
        #     self.display, self.frame_dimensions[0], self.frame_dimensions[1], 'top')
        # draw_gradient_frame(
        #     self.display, self.frame_dimensions[0], self.frame_dimensions[1], 'bottom')

        pygame.display.update()
        pygame.image.save(self.display, filename)
        # pygame.image.save(self.display, f'{self.temp_path}{self.seed}_{str(iteration_num).zfill(8)}.png')


def draw_gradient_frame(surface, surface_width, surface_height, position):
    if position == 'left':
        width = int(min(surface_width, surface_height) * 0.2)
        height = surface_height
        for x in range(width):
            # Calculate alpha value based on x position
            alpha = 255 - int((x / width) * 255)
            strip = pygame.Surface((1, height), pygame.SRCALPHA)
            strip.fill((255, 255, 255, alpha))  # Set the color with alpha
            surface.blit(strip, (x, 0))  # Draw the rectangle onto the screen
    if position == 'right':
        width = int(min(surface_width, surface_height) * 0.2)
        height = surface_height
        for x in range(width):
            # Calculate alpha value based on x position
            alpha = int((x / width) * 255)
            strip = pygame.Surface((1, height), pygame.SRCALPHA)
            strip.fill((255, 255, 255, alpha))  # Set the color with alpha
            # Draw the rectangle onto the screen
            surface.blit(strip, ((surface_width - width) + x, 0))
    if position == 'top':
        width = surface_width
        height = int(min(surface_width, surface_height) * 0.2)
        for y in range(height):
            # Calculate alpha value based on x position
            alpha = 255 - int((y / height) * 255)
            strip = pygame.Surface((width, 1), pygame.SRCALPHA)
            strip.fill((255, 255, 255, alpha))  # Set the color with alpha
            surface.blit(strip, (0, y))  # Draw the rectangle onto the screen
    if position == 'bottom':
        width = surface_width
        height = int(min(surface_width, surface_height) * 0.2)
        for y in range(height):
            # Calculate alpha value based on x position
            alpha = int((y / height) * 255)
            strip = pygame.Surface((width, 1), pygame.SRCALPHA)
            strip.fill((255, 255, 255, alpha))  # Set the color with alpha
            # Draw the rectangle onto the screen
            surface.blit(strip, (0, (surface_height - height) + y))
