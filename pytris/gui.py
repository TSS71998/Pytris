"""Pygame front-end: play Tetris yourself, or watch the trained AI play."""

import os

import pygame

from .game import HEIGHT, ROTATIONS, WIDTH, PIECE_ID, Tetris

CELL = 30
MINI = 18
PAD = 20
SIDE_L = 6 * MINI + 2 * PAD          # hold panel
SIDE_R = 7 * MINI + 2 * PAD + 40     # next queue + stats
BOARD_X = PAD + SIDE_L
BOARD_Y = PAD
WIN_W = BOARD_X + WIDTH * CELL + SIDE_R
WIN_H = HEIGHT * CELL + 2 * PAD

BG = (18, 20, 28)
GRID = (34, 38, 50)
PANEL = (26, 29, 40)
TEXT = (225, 228, 238)
DIM = (130, 136, 155)
COLORS = {
    1: (64, 214, 230),   # I
    2: (245, 210, 70),   # O
    3: (176, 98, 230),   # T
    4: (100, 210, 110),  # S
    5: (235, 82, 90),    # Z
    6: (72, 120, 235),   # J
    7: (245, 150, 60),   # L
}

LOCK_DELAY = 0.5
MAX_LOCK_RESETS = 15


def _shade(color, f):
    return tuple(max(0, min(255, int(v * f))) for v in color)


def draw_cell(surf, x, y, size, color, ghost=False):
    rect = pygame.Rect(x, y, size, size)
    if ghost:
        pygame.draw.rect(surf, _shade(color, 0.55), rect.inflate(-4, -4), 2, border_radius=3)
        return
    pygame.draw.rect(surf, color, rect.inflate(-2, -2), border_radius=4)
    pygame.draw.rect(surf, _shade(color, 1.25), (x + 3, y + 3, size - 8, 3), border_radius=2)
    pygame.draw.rect(surf, _shade(color, 0.7), rect.inflate(-2, -2), 1, border_radius=4)


def draw_mini(surf, name, cx, top):
    """Draw a small preview of piece `name` centered on cx."""
    cells = ROTATIONS[name][0]
    minr = min(r for r, _ in cells); maxr = max(r for r, _ in cells)
    minc = min(c for _, c in cells); maxc = max(c for _, c in cells)
    w = (maxc - minc + 1) * MINI
    h = (maxr - minr + 1) * MINI
    ox, oy = cx - w // 2, top + (2 * MINI - h) // 2
    for r, c in cells:
        draw_cell(surf, ox + (c - minc) * MINI, oy + (r - minr) * MINI, MINI,
                  COLORS[PIECE_ID[name]])


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("dejavusans,arial", 18)
        self.big = pygame.font.SysFont("dejavusans,arial", 30, bold=True)
        self.small = pygame.font.SysFont("dejavusans,arial", 14)

    def text(self, s, x, y, font=None, color=TEXT, center=False):
        img = (font or self.font).render(s, True, color)
        r = img.get_rect()
        if center:
            r.center = (x, y)
        else:
            r.topleft = (x, y)
        self.screen.blit(img, r)

    def draw(self, game, title, stats, help_lines, overlay=None):
        s = self.screen
        s.fill(BG)
        # board
        pygame.draw.rect(s, PANEL, (BOARD_X - 4, BOARD_Y - 4, WIDTH * CELL + 8,
                                    HEIGHT * CELL + 8), border_radius=6)
        for r in range(HEIGHT):
            for c in range(WIDTH):
                x, y = BOARD_X + c * CELL, BOARD_Y + r * CELL
                v = game.board[r][c]
                if v:
                    draw_cell(s, x, y, CELL, COLORS[v])
                else:
                    pygame.draw.rect(s, GRID, (x, y, CELL, CELL), 1)
        if not game.game_over:
            p = game.current
            color = COLORS[PIECE_ID[p.name]]
            gy = game.ghost_y()
            for r, c in p.cells(y=gy):
                if r >= 0:
                    draw_cell(s, BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, color, ghost=True)
            for r, c in p.cells():
                if r >= 0:
                    draw_cell(s, BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, color)

        # left panel: title + hold + help
        lx = PAD
        self.text(title, lx, PAD, self.big)
        self.text("HOLD", lx, PAD + 50, color=DIM)
        pygame.draw.rect(s, PANEL, (lx, PAD + 75, SIDE_L - 2 * PAD + 10, 3 * MINI + 10),
                         border_radius=6)
        if game.hold_name:
            draw_mini(s, game.hold_name, lx + (SIDE_L - 2 * PAD + 10) // 2, PAD + 80)
        y = PAD + 160
        for line in help_lines:
            self.text(line, lx, y, self.small, DIM)
            y += 20

        # right panel: next queue + stats
        rx = BOARD_X + WIDTH * CELL + PAD
        pw = SIDE_R - 2 * PAD
        self.text("NEXT", rx, PAD, color=DIM)
        pygame.draw.rect(s, PANEL, (rx, PAD + 25, pw, 5 * (2 * MINI + 12) + 10),
                         border_radius=6)
        for i, name in enumerate(game.queue[:5]):
            draw_mini(s, name, rx + pw // 2, PAD + 32 + i * (2 * MINI + 12))
        y = PAD + 25 + 5 * (2 * MINI + 12) + 30
        for label, value in stats:
            self.text(label, rx, y, self.small, DIM)
            self.text(str(value), rx, y + 16)
            y += 46

        if overlay:
            shade = pygame.Surface((WIDTH * CELL, HEIGHT * CELL), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 170))
            s.blit(shade, (BOARD_X, BOARD_Y))
            cx = BOARD_X + WIDTH * CELL // 2
            cy = BOARD_Y + HEIGHT * CELL // 2
            self.text(overlay[0], cx, cy - 20, self.big, center=True)
            for i, line in enumerate(overlay[1:]):
                self.text(line, cx, cy + 20 + i * 24, self.font, DIM, center=True)


def _init(caption):
    pygame.init()
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    return screen, Renderer(screen), pygame.time.Clock()


def _basic_stats(game):
    return [("SCORE", f"{game.score:,}"), ("LINES", game.lines),
            ("LEVEL", game.level), ("PIECES", game.pieces)]


# ------------------------------------------------------------------ human
def run_human():
    screen, ui, clock = _init("Pytris")
    pygame.key.set_repeat(170, 45)  # auto-repeat for held keys
    game = Tetris()
    fall = lock_timer = 0.0
    lock_resets = 0
    paused = False
    help_lines = ["Left/Right  move", "Down  soft drop", "Space  hard drop",
                  "Up / X  rotate", "Z  rotate left", "C / Shift  hold",
                  "P  pause", "R  restart", "Esc  quit"]

    def moved():
        nonlocal lock_timer, lock_resets
        if game.on_ground() and lock_resets < MAX_LOCK_RESETS:
            lock_timer = 0.0
            lock_resets += 1

    running = True
    while running:
        dt = clock.tick(60) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_r:
                    game = Tetris()
                    fall = lock_timer = 0.0
                    lock_resets = 0
                    paused = False
                elif k == pygame.K_p and not game.game_over:
                    paused = not paused
                elif paused or game.game_over:
                    continue
                elif k == pygame.K_LEFT:
                    game.move(-1) and moved()
                elif k == pygame.K_RIGHT:
                    game.move(1) and moved()
                elif k == pygame.K_DOWN:
                    if game.soft_drop():
                        fall = 0.0
                elif k in (pygame.K_UP, pygame.K_x):
                    game.rotate(1) and moved()
                elif k == pygame.K_z:
                    game.rotate(-1) and moved()
                elif k == pygame.K_SPACE:
                    game.hard_drop()
                    fall = lock_timer = 0.0
                    lock_resets = 0
                elif k in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT):
                    if game.hold():
                        fall = lock_timer = 0.0
                        lock_resets = 0

        if not paused and not game.game_over:
            if game.on_ground():
                lock_timer += dt
                if lock_timer >= LOCK_DELAY:
                    game.lock()
                    fall = lock_timer = 0.0
                    lock_resets = 0
            else:
                fall += dt
                while fall >= game.gravity_interval() and game.gravity_step():
                    fall -= game.gravity_interval()
                    lock_timer = 0.0

        overlay = None
        if game.game_over:
            overlay = ("GAME OVER", f"Score {game.score:,}", "R to restart")
        elif paused:
            overlay = ("PAUSED", "P to resume")
        ui.draw(game, "PYTRIS", _basic_stats(game), help_lines, overlay)
        pygame.display.flip()
    pygame.quit()


# ------------------------------------------------------------------ AI
SPEEDS = [0.25, 0.12, 0.06, 0.03, 0.012, 0.0]  # seconds per AI action


def run_ai(agent, seed=None, speed=2, label="trained"):
    screen, ui, clock = _init("Pytris - AI")
    game = Tetris(seed=seed)
    target = None
    timer = 0.0
    paused = False
    games_played, best_lines = 0, 0
    help_lines = ["The AI is playing.", "", "+ / -  speed",
                  "P  pause", "R  new game", "Esc  quit", "", f"Weights: {label}"]
    running = True
    while running:
        dt = clock.tick(60) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed + 1, len(SPEEDS) - 1)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed - 1, 0)
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_r:
                    game, target = Tetris(), None

        if not paused and not game.game_over:
            timer += dt
            delay = SPEEDS[speed]
            # at max speed do many actions per frame
            budget = 200 if delay == 0 else 0
            while (delay == 0 and budget > 0) or (delay > 0 and timer >= delay):
                timer = timer - delay if delay > 0 else 0
                budget -= 1
                if target is None:
                    target = agent.choose(game)
                    if target is None:  # no legal move: drop and lose
                        game.hard_drop()
                        break
                rot, x = target
                p = game.current
                ok = True
                if p.rot != rot:
                    ok = game.rotate(1)
                elif p.x < x:
                    ok = game.move(1)
                elif p.x > x:
                    ok = game.move(-1)
                else:
                    ok = False  # in position
                if not ok:
                    game.hard_drop()
                    target = None
                if game.game_over:
                    break

        if game.game_over and target is not None:
            target = None
        overlay = None
        if game.game_over:
            overlay = ("GAME OVER", f"{game.lines} lines", "R for a new game")
        elif paused:
            overlay = ("PAUSED", "P to resume")
        stats = _basic_stats(game) + [("TETRISES", game.tetrises),
                                      ("SPEED", f"{speed + 1}/{len(SPEEDS)}")]
        ui.draw(game, "PYTRIS AI", stats, help_lines, overlay)
        pygame.display.flip()
    pygame.quit()


def screenshot(game, path, title="PYTRIS", help_lines=()):
    """Render a single frame to a PNG (used for testing without a display)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    screen, ui, _ = _init(title)
    ui.draw(game, title, _basic_stats(game), list(help_lines))
    pygame.image.save(screen, path)
    pygame.quit()