import cv2
import mediapipe as mp
import numpy as np
import random
import pygame
import pickle
import sys
import time

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("World's Hardest Rock, Paper, Scissors!")
clock = pygame.time.Clock() 

custom_font = pygame.font.Font('assets/pricedown_bl.otf', 60) 
small_font = pygame.font.Font('assets/pricedown_bl.otf', 40)  

rock_img = pygame.image.load('assets/rock.png')
paper_img = pygame.image.load('assets/paper.png')
scissors_img = pygame.image.load('assets/scissors.png')
player_outline = pygame.image.load('assets/player_outline.png') 

with open('data/gesture_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

GESTURES = ['rock', 'paper', 'scissors']

# Game State
game_state = 'menu' 

camera = cv2.VideoCapture(0)

# Functions
def draw_text(text, font, color, x, y, stroke_color=(0, 0, 0), stroke_width=3):
    """Draw centered text with a stroke around it."""
    for dx in [-stroke_width, stroke_width]:
        for dy in [-stroke_width, stroke_width]:
            img = font.render(text, True, stroke_color)
            screen.blit(img, (x - img.get_width() // 2 + dx, y + dy))

    
    img = font.render(text, True, color)
    screen.blit(img, (x - img.get_width() // 2, y))

def show_camera_background():
    """Display the camera feed as the background."""
    success, img = camera.read()
    if success:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.rot90(img) 
        img = pygame.surfarray.make_surface(img)
        screen.blit(pygame.transform.scale(img, (800, 600)), (0, 0))

def show_menu():
    """Show the main menu with camera feed background."""
    show_camera_background()
    draw_text("Rock, Paper, Scissors!", custom_font, (255, 255, 255), 400, 100)
    
    # Play button
    play_button = pygame.Rect(300, 300, 200, 50)
    pygame.draw.rect(screen, (0, 128, 255), play_button)
    draw_text("Play!", small_font, (255, 255, 255), 400, 295)
    
    # Rules button
    rules_button = pygame.Rect(300, 400, 200, 50)
    pygame.draw.rect(screen, (0, 128, 255), rules_button)
    draw_text("Rules!", small_font, (255, 255, 255), 400, 395)  
    
    return play_button, rules_button

def show_rules():
    """Display the rules screen."""
    screen.fill((30, 30, 30)) 
    draw_text("Rules", custom_font, (255, 255, 255), 400, 100)
    draw_text("1. Play with Right-Hand", small_font, (255, 255, 255), 400, 200)
    draw_text("2. Follow on-screen instructions", small_font, (255, 255, 255), 400, 250)
    draw_text("3. Try your best!", small_font, (255, 255, 255), 400, 300)

    return_button = pygame.Rect(300, 500, 200, 50)
    pygame.draw.rect(screen, (0, 128, 255), return_button)
    draw_text("Return", small_font, (255, 255, 255), 400, 495)

    pygame.display.update()

    return return_button

def countdown_shake():
    """Smooth countdown with centered text and real-time updates."""
    countdown_steps = ["Rock", "Paper", "Scissors", "Shoot!"]
    start_time = time.time()

    for i, count in enumerate(countdown_steps):
        end_time = start_time + (i + 1)
        while time.time() < end_time:
            show_camera_background()
            draw_text(count, custom_font, (255, 255, 255), 400, 200) 
            draw_text("Make your action on Shoot!", small_font, (255, 255, 255), 400, 300)
            pygame.display.update()
            clock.tick(60) 

def get_ai_choice():
    """Random AI choice."""
    return random.choice(GESTURES)

def display_player_choice(player_choice):
    """Display the player's gesture on the left side using the same images as AI."""
    if player_choice == 'rock':
        screen.blit(rock_img, (50, 150)) 
    elif player_choice == 'paper':
        screen.blit(paper_img, (50, 150))
    elif player_choice == 'scissors':
        screen.blit(scissors_img, (50, 150))

def determine_winner(player, ai):
    """Determine winner based on player and AI choice."""
    if player == ai:
        return 'Draw'
    if (player == 'rock' and ai == 'scissors') or \
       (player == 'scissors' and ai == 'paper') or \
       (player == 'paper' and ai == 'rock'):
        return 'Player wins!'
    else:
        return 'AI wins!'

def countdown_orientation():
    """Show countdown with player outline for 5 seconds to help player orient."""
    font = pygame.font.Font('assets/pricedown_bl.otf', 50)  
    start_time = time.time()
    while time.time() - start_time < 5:
        show_camera_background()
        screen.blit(player_outline, (50, 150))
        countdown = 5 - int(time.time() - start_time)
        draw_text(f"Get Ready! {countdown}", font, (255, 255, 255), 400, 100)  
        pygame.display.update()
        clock.tick(60)  

def play_game():
    """Main gameplay loop."""
    ai_choice = get_ai_choice()
    first_round = True  
    countdown_orientation() 
    
    running = True
    while running:
        show_camera_background()

        # Display player on the left side
        success, img = camera.read()
        player_choice = None
        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_lms in results.multi_hand_landmarks:
                    for lm in hand_lms.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                landmarks = np.array(landmarks).reshape(1, -1)
                player_choice = model.predict(landmarks)[0]
                display_player_choice(player_choice) 

        if first_round:
            countdown_shake()
            first_round = False     

        if ai_choice == 'rock':
            screen.blit(rock_img, (500, 150)) 
        elif ai_choice == 'paper':
            screen.blit(paper_img, (500, 150))
        elif ai_choice == 'scissors':
            screen.blit(scissors_img, (500, 150))
        
        pygame.display.update()

        # If player made a gesture, determine the winner
        if player_choice:
            screen.fill((0, 0, 0))  
            show_camera_background()  
            display_player_choice(player_choice)  
            if ai_choice == 'rock':
                screen.blit(rock_img, (500, 150))
            elif ai_choice == 'paper':
                screen.blit(paper_img, (500, 150))
            elif ai_choice == 'scissors':
                screen.blit(scissors_img, (500, 150))

            # Display the winner along with "Your Move" and "AI's Move"
            draw_text(f"Your Move: {player_choice.capitalize()}", small_font, (255, 255, 255), 200, 400)
            draw_text(f"AI's Move: {ai_choice.capitalize()}", small_font, (255, 255, 255), 600, 400)
            winner = determine_winner(player_choice, ai_choice)
            draw_text(winner, small_font, (255, 255, 255), 400, 50)
            draw_text("Press Enter to play again!", small_font, (255, 255, 255), 400, 500)
            draw_text("Press Esc to quit", small_font, (255, 255, 255), 400, 550)
            pygame.display.update()

            # Wait for user input
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            waiting = False
                            play_game() 
                        elif event.key == pygame.K_ESCAPE:
                            waiting = False
                            running = False  
                            pygame.quit()  
                            sys.exit()  
        
        pygame.display.update()
        clock.tick(60)  

# Main game loop
running = True
while running:
    screen.fill((0, 0, 0))
    
    if game_state == 'menu':
        play_button, rules_button = show_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.collidepoint(event.pos):
                    game_state = 'play'
                    play_game()
                elif rules_button.collidepoint(event.pos):
                    game_state = 'rules'
    
    elif game_state == 'rules':
        return_button = show_rules()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if return_button.collidepoint(event.pos):
                    game_state = 'menu'
    
    pygame.display.update()
    clock.tick(60) 
