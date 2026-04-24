import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from pygame import RLEACCEL

pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.init()

scr_size = (width, height) = (600, 150)
FPS = 60
gravity = 0.6
background_col = (235, 235, 235)
GROUND_Y = int(0.98 * height)

_screen = None
_clock = None
_display_caption = "T-Rex Rush"
_display_ready = False
SPRITES_DIR = Path(__file__).resolve().parent / "sprites"


def ensure_pygame_display():
    global _screen, _clock, _display_ready
    if not _display_ready or pygame.display.get_surface() is None:
        _screen = pygame.display.set_mode(scr_size)
        pygame.display.set_caption(_display_caption)
        _display_ready = True
    else:
        _screen = pygame.display.get_surface()
    if _clock is None:
        _clock = pygame.time.Clock()
    return _screen


def get_screen():
    return ensure_pygame_display()


def get_clock():
    global _clock
    if _clock is None:
        _clock = pygame.time.Clock()
    return _clock


def load_sound(name):
    if pygame.mixer.get_init() is None:
        return None

    fullname = SPRITES_DIR / name
    try:
        return pygame.mixer.Sound(str(fullname))
    except (pygame.error, FileNotFoundError):
        return None


jump_sound = load_sound("jump.wav")
die_sound = load_sound("die.wav")
checkPoint_sound = load_sound("checkPoint.wav")


def load_image(name, sizex=-1, sizey=-1, colorkey=None):
    fullname = SPRITES_DIR / name
    image = pygame.image.load(str(fullname))
    if pygame.display.get_surface() is not None:
        image = image.convert_alpha() if image.get_alpha() is not None else image.convert()
    else:
        image = image.copy()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return image, image.get_rect()


def load_sprite_sheet(sheetname, nx, ny, scalex=-1, scaley=-1, colorkey=None):
    fullname = SPRITES_DIR / sheetname
    sheet = pygame.image.load(str(fullname))
    if pygame.display.get_surface() is not None:
        sheet = sheet.convert_alpha() if sheet.get_alpha() is not None else sheet.convert()
    else:
        sheet = sheet.copy()

    sheet_rect = sheet.get_rect()
    sprites = []
    sizex = sheet_rect.width // nx
    sizey = sheet_rect.height // ny

    for i in range(0, ny):
        for j in range(0, nx):
            rect = pygame.Rect((j * sizex, i * sizey, sizex, sizey))
            image = pygame.Surface(rect.size, pygame.SRCALPHA)
            image.blit(sheet, (0, 0), rect)

            if colorkey is not None:
                if colorkey == -1:
                    colorkey = image.get_at((0, 0))
                image.set_colorkey(colorkey, RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image, (scalex, scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()
    return sprites, sprite_rect


def disp_gameOver_msg(retbutton_image, gameover_image):
    screen = get_screen()
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width // 2
    retbutton_rect.top = int(height * 0.52)

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width // 2
    gameover_rect.centery = int(height * 0.35)

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)


def extractDigits(number):
    if number < 0:
        return [0, 0, 0, 0, 0]

    digits = []
    while number // 10 != 0:
        digits.append(number % 10)
        number = number // 10

    digits.append(number % 10)
    for _ in range(len(digits), 5):
        digits.append(0)
    digits.reverse()
    return digits


class Dino(pygame.sprite.Sprite):
    def __init__(self, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self)
        self.images, self.rect = load_sprite_sheet("dino.png", 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet("dino_ducking.png", 2, 1, 59, sizey, -1)
        self.rect.bottom = GROUND_Y
        self.rect.left = width // 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        get_screen().blit(self.image, self.rect)

    def checkbounds(self):
        if self.rect.bottom > GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.isJumping = False
            self.movement[1] = 0

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2
        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2

        if self.isDead:
            self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[self.index % 2]
            self.rect.width = self.duck_pos_width

        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and not self.isBlinking:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0 and checkPoint_sound is not None:
                checkPoint_sound.play()

        self.counter += 1


class Cactus(pygame.sprite.Sprite):
    def __init__(self, rng, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet("cacti-small.png", 3, 1, sizex, sizey, -1)
        self.rect.bottom = GROUND_Y
        self.rect.left = width + self.rect.width
        self.image = self.images[rng.randrange(0, 3)]
        self.mask = pygame.mask.from_surface(self.image)
        self.movement = [-1 * speed, 0]

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Ptera(pygame.sprite.Sprite):
    def __init__(self, rng, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet("ptera.png", 2, 1, sizex, sizey, -1)
        self.ptera_height = [int(height * 0.82), int(height * 0.75), int(height * 0.60)]
        self.rect.centery = self.ptera_height[rng.randrange(0, 3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.rect.move(self.movement)
        self.counter += 1
        if self.rect.right < 0:
            self.kill()


class Ground:
    def __init__(self, speed=-5):
        self.image, self.rect = load_image("ground.png", -1, -1, -1)
        self.image1, self.rect1 = load_image("ground.png", -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen = get_screen()
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right
        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("cloud.png", int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard:
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.tempimages, self.temprect = load_sprite_sheet("numbers.png", 12, 1, 11, int(11 * 6 / 5), -1)
        self.image = pygame.Surface((55, int(11 * 6 / 5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = int(width * 0.89)
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = int(height * 0.1)
        else:
            self.rect.top = y

    def draw(self):
        get_screen().blit(self.image, self.rect)

    def update(self, score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


class TrexEnv(gym.Env):
    metadata = {"render_modes": [None, "human"], "render_fps": FPS}
    action_meanings = {0: "noop", 1: "jump", 2: "duck"}

    def __init__(self, render_mode=None, max_episode_steps=5000, max_steps=None):
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps if max_steps is None else max_steps
        self.observation_size = 15
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([
                0.0, -20.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32),
            high=np.array([
                float(height), 20.0, 1.0, 1.0, 100.0,
                1.0, float(width), float(width), float(height), float(height),
                1.0, float(width), float(width), float(height), float(height),
            ], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = random.Random()
        self.high_score = 0
        self.playerDino = None
        self.new_ground = None
        self.scb = None
        self.highsc = None
        self.cacti = None
        self.pteras = None
        self.clouds = None
        self.last_obstacle = None
        self.gamespeed = 4
        self.counter = 0
        self.terminated = False
        self.truncated = False
        self.last_score = 0
        self.obstacles_cleared = 0

        if self.render_mode == "human":
            ensure_pygame_display()
        self.replay_button_image, _ = load_image("replay_button.png", 35, 31, -1)
        self.gameover_image, _ = load_image("game_over.png", 190, 11, -1)
        number_images, number_rect = load_sprite_sheet("numbers.png", 12, 1, 11, int(11 * 6 / 5), -1)
        self.hi_image = pygame.Surface((22, int(11 * 6 / 5)))
        self.hi_rect = self.hi_image.get_rect()
        self.hi_image.fill(background_col)
        self.hi_image.blit(number_images[10], number_rect)
        number_rect.left += number_rect.width
        self.hi_image.blit(number_images[11], number_rect)
        self.hi_rect.top = int(height * 0.1)
        self.hi_rect.left = int(width * 0.73)

    def _reset_round(self, start_speed):
        self.playerDino = Dino(44, 47)
        self.new_ground = Ground(-1 * start_speed)
        self.scb = Scoreboard()
        self.highsc = Scoreboard(int(width * 0.78))
        self.cacti = pygame.sprite.Group()
        self.pteras = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()
        self.last_obstacle = pygame.sprite.Group()

        Cactus.containers = self.cacti
        Ptera.containers = self.pteras
        Cloud.containers = self.clouds

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        self.gamespeed = 4
        self.counter = 0
        self.terminated = False
        self.truncated = False
        self.last_score = 0
        self.obstacles_cleared = 0
        self._reset_round(self.gamespeed)
        observation = self._get_obs()
        info = self._make_info()
        return observation, info

    def _play_die_sound(self):
        if die_sound is not None and self.render_mode == "human":
            die_sound.play()

    def _jump(self):
        if self.playerDino.rect.bottom == GROUND_Y:
            self.playerDino.isJumping = True
            self.playerDino.movement[1] = -1 * self.playerDino.jumpSpeed
            if jump_sound is not None and self.render_mode == "human":
                jump_sound.play()

    def _apply_action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Unsupported action: {action}")

        self.playerDino.isDucking = action == 2 and not self.playerDino.isDead
        if action == 1:
            self._jump()

    def _update_obstacles(self):
        for cactus in self.cacti:
            cactus.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.playerDino, cactus):
                self.playerDino.isDead = True
                self._play_die_sound()
                return

        for ptera in self.pteras:
            ptera.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.playerDino, ptera):
                self.playerDino.isDead = True
                self._play_die_sound()
                return

    def _spawn_obstacles(self):
        if len(self.cacti) < 2:
            if len(self.cacti) == 0:
                self.last_obstacle.empty()
                self.last_obstacle.add(Cactus(self.rng, self.gamespeed, 40, 40))
            else:
                for obstacle in self.last_obstacle:
                    if obstacle.rect.right < int(width * 0.7) and self.rng.randrange(0, 50) == 10:
                        self.last_obstacle.empty()
                        self.last_obstacle.add(Cactus(self.rng, self.gamespeed, 40, 40))
                        break

        if len(self.pteras) == 0 and self.rng.randrange(0, 200) == 10 and self.counter > 500:
            for obstacle in self.last_obstacle:
                if obstacle.rect.right < int(width * 0.8):
                    self.last_obstacle.empty()
                    self.last_obstacle.add(Ptera(self.rng, self.gamespeed, 46, 40))
                    break

        if len(self.clouds) < 5 and self.rng.randrange(0, 300) == 10:
            Cloud(width, self.rng.randrange(height // 5, height // 2))

    def _count_cleared_obstacles(self):
        cleared = 0
        dino_left = self.playerDino.rect.left
        for obstacle in list(self.cacti) + list(self.pteras):
            if not getattr(obstacle, "counted_for_reward", False) and obstacle.rect.right < dino_left:
                obstacle.counted_for_reward = True
                cleared += 1
        return cleared

    def _update_entities(self):
        self.playerDino.update()
        self.cacti.update()
        self.pteras.update()
        self.clouds.update()
        self.new_ground.update()
        self.scb.update(self.playerDino.score)
        self.highsc.update(self.high_score)

    def _nearest_obstacles(self, count=2):
        obstacles = []
        dino_right = self.playerDino.rect.right
        for obstacle in list(self.cacti) + list(self.pteras):
            if obstacle.rect.right >= dino_right:
                obstacles.append(obstacle)
        obstacles.sort(key=lambda obstacle: obstacle.rect.left)
        return obstacles[:count]

    def _obstacle_features(self, obstacle):
        if obstacle is None:
            return [0.0, float(width), 0.0, 0.0, 0.0]

        obstacle_type = 1.0 if isinstance(obstacle, Ptera) else 0.0
        distance = float(max(0, obstacle.rect.left - self.playerDino.rect.right))
        obstacle_width = float(obstacle.rect.width)
        obstacle_height = float(obstacle.rect.height)
        obstacle_center_y = float(obstacle.rect.centery)
        return [obstacle_type, distance, obstacle_width, obstacle_height, obstacle_center_y]

    def _build_observation(self):
        nearest = self._nearest_obstacles(2)
        first = nearest[0] if len(nearest) > 0 else None
        second = nearest[1] if len(nearest) > 1 else None

        obs = [
            float(GROUND_Y - self.playerDino.rect.bottom),
            float(self.playerDino.movement[1]),
            float(int(self.playerDino.isJumping)),
            float(int(self.playerDino.isDucking)),
            float(self.gamespeed),
        ]
        obs.extend(self._obstacle_features(first))
        obs.extend(self._obstacle_features(second))
        return obs

    def _get_obs(self):
        return np.clip(
            np.array(self._build_observation(), dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )

    def _make_info(self):
        return {
            "score": self.playerDino.score,
            "high_score": self.high_score,
            "speed": self.gamespeed,
            "obstacles_cleared": self.obstacles_cleared,
            "counter": self.counter,
        }

    def _compute_reward(self, newly_cleared):
        reward = 0.1
        if newly_cleared > 0:
            reward += 1.0 * newly_cleared
            self.obstacles_cleared += newly_cleared
        if self.playerDino.isDead:
            reward -= 10.0
        self.last_score = self.playerDino.score
        return reward

    def step(self, action):
        if self.playerDino is None:
            raise RuntimeError("Call reset() before step().")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        self._apply_action(action)
        self._update_obstacles()
        if not self.playerDino.isDead:
            self._spawn_obstacles()
        self._update_entities()
        newly_cleared = self._count_cleared_obstacles()

        if self.playerDino.isDead:
            self.terminated = True
            if self.playerDino.score > self.high_score:
                self.high_score = self.playerDino.score

        if self.counter % 700 == 699:
            self.new_ground.speed -= 1
            self.gamespeed += 1

        self.counter += 1
        if self.counter >= self.max_episode_steps:
            self.truncated = True

        observation = self._get_obs()
        reward = self._compute_reward(newly_cleared)
        info = self._make_info()

        if self.render_mode == "human":
            self.render()
            get_clock().tick(FPS)

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        screen = get_screen()
        screen.fill(background_col)
        self.new_ground.draw()
        self.clouds.draw(screen)
        self.scb.draw()
        if self.high_score != 0:
            self.highsc.draw()
            screen.blit(self.hi_image, self.hi_rect)
        self.cacti.draw(screen)
        self.pteras.draw(screen)
        self.playerDino.draw()
        pygame.display.update()

    def render_game_over(self):
        self.highsc.update(self.high_score)
        disp_gameOver_msg(self.replay_button_image, self.gameover_image)
        if self.high_score != 0:
            self.highsc.draw()
            get_screen().blit(self.hi_image, self.hi_rect)
        pygame.display.update()

    def get_action_meanings(self):
        return [self.action_meanings[index] for index in range(self.action_space.n)]

    def close(self):
        pygame.quit()


def register_trex_envs():
    env_id = "TrexEnv-v0"
    if env_id in registry:
        return

    register(
        id=env_id,
        entry_point="DQN.envs.dino.env:TrexEnv",
        max_episode_steps=None,
    )


def introscreen():
    ensure_pygame_display()
    temp_dino = Dino(44, 47)
    temp_dino.isBlinking = True
    gameStart = False

    callout, callout_rect = load_image("call_out.png", 196, 45, -1)
    callout_rect.left = int(width * 0.05)
    callout_rect.top = int(height * 0.4)

    temp_ground, temp_ground_rect = load_sprite_sheet("ground.png", 15, 1, -1, -1, -1)
    temp_ground_rect.left = width // 20
    temp_ground_rect.bottom = height

    logo, logo_rect = load_image("logo.png", 240, 40, -1)
    logo_rect.centerx = int(width * 0.6)
    logo_rect.centery = int(height * 0.6)
    while not gameStart:
        if pygame.display.get_surface() is None:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_UP):
                temp_dino.isJumping = True
                temp_dino.isBlinking = False
                temp_dino.movement[1] = -1 * temp_dino.jumpSpeed

        temp_dino.update()

        screen = get_screen()
        screen.fill(background_col)
        screen.blit(temp_ground[0], temp_ground_rect)
        if temp_dino.isBlinking:
            screen.blit(logo, logo_rect)
            screen.blit(callout, callout_rect)
        temp_dino.draw()
        pygame.display.update()

        get_clock().tick(FPS)
        if not temp_dino.isJumping and not temp_dino.isBlinking:
            gameStart = True

    return False


def play_human():
    env = TrexEnv(render_mode="human")
    env.reset()
    duck_pressed = False

    while True:
        action = 2 if duck_pressed else 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return
                if event.key == pygame.K_SPACE:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    duck_pressed = True
                    action = 2
            if event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
                duck_pressed = False
                action = 0

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            duck_pressed = False
            waiting_for_restart = True
            while waiting_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            waiting_for_restart = False
                env.render_game_over()
                get_clock().tick(FPS)
            env.reset()


register_trex_envs()
