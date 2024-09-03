# Pac-Man AI

## Overview
This project is an implementation of AI algorithms to enhance the gameplay of Pac-Man. Developed as part of UC Berkeley's CS 188 Pac-Man AI Projects, this repository explores various search algorithms and adversarial techniques to optimize Pac-Man's navigation and decision-making within a dynamic environment.

## Features
- **Search Algorithms**: Implementation of A* (A-star), Depth-First Search (DFS), and Breadth-First Search (BFS) to find the optimal paths for Pac-Man to reach its goals.
- **Adversarial Search**: Integration of Expectimax and Minimax algorithms to develop intelligent adversarial agents that challenge Pac-Man.
- **Reinforcement Learning**: Application of reinforcement learning techniques to enable Pac-Man to adapt and improve its strategies over time based on previous experiences.

## Project Structure
- `search.py`: Contains the implementation of various search algorithms including A*, DFS, and BFS.
- `adversarial.py`: Implements the adversarial search techniques such as Expectimax and Minimax.
- `pacman.py`: The main script that runs the game, utilizing the implemented algorithms to control Pac-Man's behavior.
- `ghostAgents.py`: Defines the behavior of ghost agents that act as adversaries to Pac-Man.
- `game.py`: Handles the game logic, including the rules, score calculations, and the state of the game environment.
