Rock Paper Scissors Game

This is a fun Python-based Rock Paper Scissors game created using Pygame and OpenCV. The game allows the player to play Rock Paper Scissors using hand gestures tracked by a camera, making it interactive without the need for a mouse or keyboard input.

This game was built as a fun project for my little brother, who constantly asked me to play Rock Paper Scissors with him. Now, with this game, he can play anytime without needing me around!

Features:
Data Collection Phase: The game also includes a data collection phase that allows the player to train the model. This phase collects hand gesture data for "rock", "paper", and "scissors" using camera input. The collected data is used to improve the model's accuracy in recognizing the player's gestures.
Hand Gesture Tracking: The game uses OpenCV and Mediapipe for real-time hand gesture tracking to recognize the player's move (rock, paper, or scissors).
Opponent: The game includes an AI that randomly selects its move, offering a new challenge with each round.
Countdown System: The game features a countdown ("Rock... Paper... Scissors... Shoot!") to mimic real-world timing.
Interactive Feedback: The player's move is displayed on the left side of the screen, while the AI’s move is displayed on the right side.
On-Screen Instructions: During the countdown, the message "Make your action on shoot!" appears to guide the player.

Files and Directory Structure:
game_logic.py: The main game file that runs the Rock Paper Scissors game.
data_collection.py: A separate file used to collect hand gesture data for rock, paper, and scissors. During the data collection phase, you perform each gesture multiple times, and the coordinates of your hand landmarks are stored.
train_model.py: A script that processes the collected hand gesture data and trains a classifier to distinguish between the gestures. The resulting model is stored in gesture_classifier.pkl.
gesture_classifier.pkl: The trained model that is used by the game to recognize the player's gestures during gameplay.

How to Play:
Start the Game: Run the game using Python. You’ll be greeted by a menu with two buttons:
Play: Start the Rock Paper Scissors game.
Rules: View the game instructions.

Game Mechanics:
After clicking "Play," the camera feed will display, and the game will begin.
A countdown will appear ("Rock... Paper... Scissors... Shoot!").
Make your move during the "Shoot!" phase using your right hand. The game will detect whether you chose rock, paper, or scissors.
The AI will also make a random move, and the game will display who won the round.
Play Again: After each round, you’ll have the option to play again or quit.

Requirements:
Python3
Pygame
OpenCV
Mediapipe

To install the required libraries, run:
pip install pygame opencv-python mediapipe

Project Structure
/RockPaperScissorsProject
│
├── /assets
│   ├── rock.png
│   ├── paper.png
│   ├── scissors.png
│   ├── player_outline.png
│   └── pricedown_bl.otf 
│
├── /data
│   └── gesture_classifier.pkl 
│
├── game_logic.py  
└── README.md  

Run the game:
python game_logic.py

Future Plans:

-Add different difficulty levels for the AI.
-Improve hand tracking accuracy.
-Introduce new game modes.
