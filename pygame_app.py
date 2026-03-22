import pygame
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
#  Model definition
# =======================
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =======================
#  Model loading
# =======================
device = torch.device("cpu")  # CPU is enough here
model = DigitNet().to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()
print("Model loaded on", device)

# =======================
#  PyGame setup
# =======================
pygame.init()

WIDTH, HEIGHT = 800, 600
CONSOLE_HEIGHT = 140

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DIGI")

# Colors
BLACK      = (15, 15, 15)
PURE_BLACK = (0, 0, 0)      # canvas background (like MNIST)
WHITE      = (240, 240, 240)
GRAY       = (70, 70, 70)
DARK_GRAY  = (40, 40, 40)
GREEN      = (0, 200, 0)

# Drawing area rect (where the canvas will be shown)
DRAW_AREA = pygame.Rect(20, 20, WIDTH - 40, HEIGHT - CONSOLE_HEIGHT - 40)

# --- Dedicated canvas just for drawing ---
canvas = pygame.Surface(DRAW_AREA.size)  # width x height
canvas.fill(PURE_BLACK)                  # black background like MNIST

# Fonts
pygame.font.init()
small_font   = pygame.font.SysFont("Menlo", 16)
info_font    = pygame.font.SysFont("Menlo", 20, bold=True)
pred_font    = pygame.font.SysFont("Avenir", 80, bold=True)
percent_font = pygame.font.SysFont("Avenir", 32, bold=True)

DRAW_RADIUS = 7
clock = pygame.time.Clock()

# =======================
#  Console logging
# =======================
console_lines = []
MAX_LINES = 5

def log(msg: str):
    print(msg)
    console_lines.append(msg)
    if len(console_lines) > MAX_LINES:
        console_lines.pop(0)

# =======================
#  Pre-processing (from canvas)
# =======================
def preprocess_canvas():
    """
    Take the drawing canvas (only the drawing, no UI), preprocess to 28x28 tensor.
    """
    # canvas is a Surface of size DRAW_AREA.size
    arr = pygame.surfarray.array3d(canvas)  # (W, H, 3)
    arr = np.transpose(arr, (1, 0, 2))      # -> (H, W, 3)
    log(f">> canvas array shape: {arr.shape}")

    # Convert to grayscale
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    log(">> converted to grayscale")

    # OPTIONAL: crop tight around the digit to help centering (simple bbox)
    # Threshold to decide "ink"
    thresh = 10  # since background is 0, white strokes ~255
    ys, xs = np.where(gray > thresh)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        gray = gray[y_min:y_max+1, x_min:x_max+1]
        log(f">> cropped to bbox: {gray.shape}")
    else:
        log(">> WARNING: nothing drawn, input is blank")

    # Resize to 28x28
    img = Image.fromarray(gray.astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    log(">> resized to 28x28")

    img_arr = np.array(img).astype(np.float32) / 255.0

    # MNIST style: white digit on black background, so this is fine

    img_arr = img_arr.reshape(1, 1, 28, 28)
    tensor = torch.from_numpy(img_arr).to(device)
    tensor = (tensor - 0.5) / 0.5  # same normalization as training
    log(">> normalized tensor ready")

    return tensor

# =======================
#  Prediction
# =======================
last_prediction = None  # (digit, confidence)

def predict_digit():
    global last_prediction
    with torch.no_grad():
        x = preprocess_canvas()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()
        last_prediction = (pred, conf)
        log(f">> predicted digit: {pred}")
        log(f">> confidence: {conf:.2f}")


# =======================
#  Drawing UI / Console
# =======================
def draw_layout():
    # Background
    screen.fill(PURE_BLACK)

    # Draw background panel for drawing (rounded)
    pygame.draw.rect(screen, DARK_GRAY, DRAW_AREA, border_radius=24)

    # Blit the drawing canvas on top of that panel
    screen.blit(canvas, DRAW_AREA.topleft)

    # Console panel
    console_rect = pygame.Rect(10, HEIGHT - CONSOLE_HEIGHT - 10,
                               WIDTH - 20, CONSOLE_HEIGHT)
    pygame.draw.rect(screen, GRAY, console_rect, border_radius=10)

    inner = console_rect.inflate(-8, -8)
    pygame.draw.rect(screen, DARK_GRAY, inner, border_radius=6)

    # Console text
    x_text = inner.x + 14
    y_text = inner.y + 10
    for line in console_lines:
        surf = small_font.render(line, True, WHITE)
        screen.blit(surf, (x_text, y_text))
        y_text += 22

    # Prediction display
    if last_prediction is not None:
        digit, conf = last_prediction
        pct = int(conf * 100)

        digit_surf = pred_font.render(str(digit), True, WHITE)
        digit_rect = digit_surf.get_rect()
        digit_rect.bottomright = (inner.right - 100, inner.bottom - 10)
        screen.blit(digit_surf, digit_rect)

        pct_surf = percent_font.render(f"{pct}%", True, GREEN)
        pct_rect = pct_surf.get_rect()
        pct_rect.midleft = (digit_rect.right + 20, digit_rect.centery + 15)
        screen.blit(pct_surf, pct_rect)

    # info_text = "Draw in the top panel.  (Predictions auto while drawing)   C: clear   ESC: quit"
    # info_surf = info_font.render(info_text, True, WHITE)
    # screen.blit(info_surf, (30, 25))


# =======================
#  Main loop
# =======================
running = True
drawing = False

# --- prediction timer ---
PRED_INTERVAL_MS = 100          # <<< NEW: 100 ms between predictions
last_pred_time   = 0            # <<< NEW

log("App started")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                canvas.fill(PURE_BLACK)  # clear drawing only
                last_prediction = None
                log("canvas cleared")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and DRAW_AREA.collidepoint(event.pos):
                drawing = True
            elif event.button == 3:
                canvas.fill(PURE_BLACK)  # clear drawing only
                last_prediction = None
                log("canvas cleared")
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False

    # Draw on the canvas (NOT on screen)
    if drawing:
        mx, my = pygame.mouse.get_pos()
        # convert screen coords -> canvas coords
        if DRAW_AREA.collidepoint((mx, my)):
            cx = mx - DRAW_AREA.x
            cy = my - DRAW_AREA.y
            # your multiple circles (kept exactly as you wrote)
            pygame.draw.circle(canvas, WHITE, (cx, cy), DRAW_RADIUS)

        # --- auto prediction every 100 ms while drawing ---
        now = pygame.time.get_ticks()                     # <<< NEW
        if now - last_pred_time >= PRED_INTERVAL_MS:      # <<< NEW
            predict_digit()                               # <<< NEW
            last_pred_time = now                          # <<< NEW

    # Render everything
    draw_layout()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()